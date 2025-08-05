import itertools
import json
import numpy as np
import openai
import os
import pandas as pd
import copy
from argparse import ArgumentParser
from dotenv import load_dotenv
from num2words import num2words
from openai import OpenAI
from pathlib import Path
from random import randint
from tqdm import tqdm
from typing import List, Tuple

from datageneration.data_model import (RelSpatial, LocPoint, Area, Entity, Property, Relation, Relations,
                                       GeneratedPrompt, GeneratedIMRSentence, Distance)
from datageneration.utils import (add_yaml_to_filename, write_output, write_dict_output, write_output_csv,
                                  translate_queries_to_yaml)

load_dotenv(override=True)

# imports
import random
import time


# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specified errors
            except errors as e:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper


@retry_with_exponential_backoff
def chatcompletions_with_backoff(**kwargs):
    '''
    Helper function to deal with the "openai.error.RateLimitError". If not used, the script will simply
    stop once the limit is reached, not saving any of the data generated until then. This method will wait
    and then try again, hence preventing the error.

    :param kwargs: List of arguments passed to the OpenAI API for completion.
    '''
    return CLIENT.chat.completions.create(**kwargs)


# OpenAI parameters
MODEL = os.getenv('MODEL', 'azure-gpt-4.1')# #gpt-4.1-mini  #gpt-4.1-nano
# MODEL = os.getenv('MODEL', 'gpt-4.1-mini')
    # https://openai.com/index/gpt-4-1/
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.4))
TOP_P = float(os.getenv('TEMPERATURE', 0.9))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 4096))

# CLIENT = OpenAI(
#     api_key=os.environ["OPENAI_API_KEY"], organization=os.environ["OPENAI_ORG"]
# )
CLIENT = OpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url="https://llm-hub.dw.com/openai"
)

def request_openai(prompt):
    response = chatcompletions_with_backoff(
        model=MODEL,  # "gpt-4",
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    text = response.choices[0].message.content
    return text
# def request_openai(prompt):
#     response = CLIENT.chat.completions.create(
#         model="azure-gpt-4o",  # Must match deployment name, not OpenAI model name
#         messages=[{"role": "user", "content": prompt}],
#         temperature=TEMPERATURE,
#         max_tokens=MAX_TOKENS,
#     )
#     return response.choices[0].message.content

def is_number(s):
    if not s:
        return False
    try:
        float(s)  # Try converting the string to a float
        return True
    except ValueError:
        return False


def remove_surrounding_double_quotes(text):
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def post_processing(text):
    text = text.replace('\r', '').strip()
    text = text.replace("User:", "")
    text = remove_surrounding_double_quotes(text)
    return text


def load_rel_spatial_terms(relative_spatial_terms_path: str) -> List[RelSpatial]:
    relative_spatial_terms = pd.read_csv(relative_spatial_terms_path, sep=',').to_dict(orient='records')
    processed_rel_spatial_terms = []
    for relative_spatial_term in relative_spatial_terms:
        values = list(map(lambda x: x.rstrip().strip(), relative_spatial_term['Vals'].split(',')))
        relative_spatial_term_dist = relative_spatial_term['Dist'].split()
        processed_rel_spatial_terms.append(RelSpatial(distance=Distance(magnitude=relative_spatial_term_dist[0], metric=relative_spatial_term_dist[1]), values=values))
    return processed_rel_spatial_terms

def load_contains_terms(contains_terms_path: str) -> List[str]:
    df = pd.read_csv(contains_terms_path, sep=',')  # assuming tab-separated
    contains_terms = df.iloc[0, 1]  # row 0 (after header), column 1
    return contains_terms.split(', ')

def load_list_of_strings(list_of_strings_path: str) -> List[str]:
    '''
    Helper function for personas and styles for data generation. Loads a list of strings from a text file.
    params:
    list_of_strings_path: Path to the personas or styles text file.
    return:
    list_of_strings: List of strings, either personas or styles.
    '''
    with open(list_of_strings_path, 'r') as f:
        list_of_strings = f.readlines()
        list_of_strings = list(map(lambda x: x.rstrip().strip(), list_of_strings))
    return list_of_strings


def normalize_entity_name(entity_name):
    # if 'brand:' in entity_name:
    #     entity_name = entity_name.replace('brand:', '')

    return entity_name


class PromptHelper:
    '''
    It is a helper class for prompt generation. It has templates and functions for paraphrasing prompts.
    '''

    def __init__(self, relative_spatial_terms, contains_terms, prob_usage_of_relative_spatial_terms,
                 prob_usage_of_written_numbers, prob_distance_writing_with_full_metric,
                 prob_distance_writing_no_whitespace):
        self.relative_spatial_terms = relative_spatial_terms
        self.contains_terms = contains_terms
        self.prob_usage_of_relative_spatial_terms = prob_usage_of_relative_spatial_terms
        self.prob_usage_of_written_numbers = prob_usage_of_written_numbers
        self.prob_distance_writing_no_whitespace = prob_distance_writing_no_whitespace
        self.prob_distance_writing_with_full_metric = prob_distance_writing_with_full_metric
        # self.beginning_template = (
        #     "You are an assistant that generates short, natural-sounding user queries based on structured geographic "
        #     "data in the form of scene descriptions.\n\n"
        #     "Imagine you're an investigative journalist or fact-checker, looking at an image or video of a real-world scene, "
        #     "and you're trying to describe what you see — the objects, places, and how they relate to one another. Your goal is "
        #     "to write what a regular person might type into a search box to describe or explore that scene.\n\n"
        #     "The scene description provides a list of entities (e.g., landmarks, buildings, places, or objects), their properties, and how "
        #     "they are spatially related.\n"
        #     "Your task is to turn this into a natural, casual sentence or query — not a literal translation "
        #     "of the scene description.\n\n"
        #     "Here’s how to approach it:\n"
        #     "- Focus on the scene, not the data format. Don't copy the structure or terminology of the input. Instead, write as if "
        #     "you were describing the real-world layout to someone else.\n"
        #     "- Use casual, human phrasing. Avoid technical terms like \"entity\", \"property\", or \"OSM key\".\n"
        #     "- Make sure to correctly use the entity information in the sentence and use ALL available information:"
        #     "  - Entities can either be a single entity (e.g. \"- Obj. 0: viewpoint\", i.e. a viewpoint), or a cluster of multiple "
        #     "of one type (e.g. \"- Obj. 1: 3 x bench\", i.e. three benches).\n"
        #     "  - If a cluster has no distance value, just use it like an entity in the sentence with the number of "
        #     " occurences mentioned (e.g. \"three benches\"). \n"
        #     "  - A cluster can also have a specified distance value between the entities (e.g. \"- Obj. 0: 2 x house, "
        #     "at max 50 m to another\" -> In the sentence (example phrasing): \"two houses within 50 m\") \n"
        #     "  - Important: If there is a distance specified for a cluster, the distance value MUST be used in the "
        #     "sentence!! This can either be a distance value (see example above), or relative spatial terms (e.g. "
        #     " \"five foutains next to another\"). \n"
        #     "  - A cluster distance is different from a distance relation. Distance relations (if used) come in a separate "
        #     "section marked as \"Distance\", cluster distances are part of object definitions. The cluster distance is only "
        #     "between the multiple instances of the same object. A cluster can have a distance between its instances, and separately "
        #     "relations that define the distance of the cluster to other objects/cluster. The phrase must include both "
        #     "if both are given (e.g. \"3 houses in a radius of 30 m, which are 100 m from a fountain\" or \"a church next to two "
        #     "parks that are nearby another). \n"
        #     "- Translate tags into natural language. For example:\n"
        #     "  - Entity \"brand:Thalia\" → \"a Thalia\"\n"
        #     "  - Entity \"cafe\" + Property \"brand~Eiffel\" → \"an Eiffel café\"\n"
        #     "  - Entity \"restaurant\" + Property \"cuisine~italian\" → \"an Italian restaurant\"\n"
        #     "  - Property \"building:material=wood\" → \"made from wood\"\n"
        #     "  - Property \"roof:colour=red\" → \"with a red roof\"\n\n"
        #     "- Always reflect spatial relationships exactly as stated in the scene description:\n"
        #     "  - If a distance is given, treat it as a maximum.\n"
        #     "  - If a relation has a specified phrase (e.g., \"next to\", \"surrounded by\"), use that exact phrase — don’t invent alternatives.\n"
        #     "  - If a contains relation is given, use phrases like \"containing\", \"with\" and \"in(side)\" to describe the spatial relation.\n"
        #     "  - If no relation is provided, do not imply one (in general, avoid terms like \"with\", \"near\" or \"close to\" if not explicitly mentioned).\n\n"
        #     "- Use number formatting like this: {thousands} for thousands separators and {decimal} for decimals. Example: {example}.\n"
        #     "- Avoid repetition in phrasing across outputs. In general be direct, but include natural variation — "
        #     "some sentences can be a bit longer or have more detail; while the tendency is to be short and to the point.\n"
        #     "- Only use the provided information about the scene — and use **all** of it! Double check that all details, including cluster "
        #     "distances and properties, are used in the generated sentence!\n"
        #     "- You must use the same alphabet as used in the provided data. Do not change them to their english version in generated sentence if the "
        #     "original used a non-latin alphabet, or the other way around.\n"
        #     "- Always use metrics exactly as specified in distance information, do not convert them to a different unit.\n"
        #     "- Do not generate why/what/how type questions, only instructions.\n\n"
        #     # "- If an entity/property combo is obviously nonsensical (e.g., a toilet or a street with a cuisine, a cliff "
        #     # "with a brand etc.), no sentence should be generated. This is only related to the entity and property names, "
        #     # "unrealistic numeric values like height or number of floor are acceptable. In nonsensical cases, return only:\n "
        #     # "`UNREALISTIC COMBINATION`\n\n"

        self.instruction_template = """
###  SYSTEM  ###
You convert scene descriptions into a single, casual search query in English.

# RULES
{rules}

# EXAMPLES
Below are examples of how to convert a structured scene description into a natural-sounding query. Use them as inspiration, but vary the style, phrasing, and structure in your own outputs.

## Example 1
Objects:
- Obj. 0: public toilet
- Obj. 1: church
- Obj. 2: 2 x tower, use this phrase to describe the spatial relation between the objects: side by side (from/to/of) another     
Distances:
- The public toilet is inside the church, use this term to describe their relation: in  
- The tower is contained in the church, use this term to describe their relation: with 

-> Find me a public toilet in a church with two towers of 100 meters height side by side. 
## Example 2
Search area:
- Amphoe Phu Pha Man, Khon Kaen, Thailand
Objects:
- Obj. 0: beach
- Obj. 1: 2 x tavern | Properties -> cuisine: greek, outdoor seating
- Obj. 2: scuba center
Distances:
- The tavern is inside the beach, use this term to describe their relation: with 
- The tavern is 1.2 km distance away from the scuba center

-> Searching for a beach in Amphoe Phu Pha Man, Khon Kaen, Thailand, with 2 greek taverns with outdoor seating and a scuba center 1.2 km away. 
## Example 3
Search area:
- Petro-Slavyanka, Saint Petersburg
Objects:
- Obj. 0: 8 x street lamp, at max 1.5m distance between each other
- Obj. 1: police station
- Obj. 2: fire hydrant
Distances:
- The police station is 125.20m from the street lamps
- The fire hydrant is more or less 3 m distance away from the police station

-> Find 8 street lamps 125.20m from a police station. The lamps are at max 1.5m apart from each other. The location is Petro-Slavyanka, Saint Petersburg, Russia. There is a fire hydrant 3 m away from the station. 
## Example 4
Objects:
- Obj. 0: gas station | Properties -> brand: "Esso"
- Obj. 1: railway | Properties -> in a cutting
Distances:
- No distance is given.

-> Show me all Esso gas stations and railways that are in a cutting.
## Example 5
Search area:
- အင်္ဂလန်၊ ယူနိုက်တက်ကင်းဒမ်း
Objects:
- Obj. 0: speed camera
- Obj. 1: 3 x dovecote
- Obj. 2: fountain
Distances:
- All objects are two hundred miles from each other.

-> I is look for speed camera within two hundret miles from 3 dovecotes and a fontain within that are in the selected area. My search area is အင်္ဂလန်၊ ယူနိုက်တက်ကင်းဒမ်း.

# TASK
Generate one natural search query and double-check that it follows all rules for the scene below.
Think step-by-step, but only output the query, nothing else.
- Persona: {persona}
- Style: {style}
{typos}
# INPUT
{input}
        """

        self.core_rules = [
            "Use all information from the input scene. Do not invent or omit anything.",
            "Translate the scene into a natural, casual, descriptive sentence - not a literal data dump.",
            "Use the same alphabet and script as provided in the input. Never translate or switch alphabets.",
            "Always preserve exact units. Do not convert (e.g. 'ft' must remain 'ft').",
            "Translate tags into natural human phrases (e.g. 'gas station' + 'brand: Esso' → 'an Esso gas station', 'brand:DM' → 'a DM').",
            "Avoid technical terms like 'entity', 'property', 'key', or 'OSM'.",
            "Use number formatting like this: {thousands} for thousands separators and {decimal} for decimals. Example: {example}.",
            "Use short, direct, user-style queries. Avoid overly formal, explanatory, or question-style phrasings.",
            "Include natural linguistic variety across outputs - vary beginning, structure and phrasing. Follow the persona, style and typo instructions.",
            "If the input includes multiple entities, mention them in the order they appear - unless a spatial relationship requires reordering."
        ]
        self.optional_rules = {
            "cluster": "If multiple instances of an object are listed (e.g. '3 x bench'), mention the count, and describe how they're grouped (e.g. 'three benches').",
            "cluster_distance": "If a cluster specifies a distance between items (e.g. 'at max 50 m'), you must include that distance in the sentence. Use natural phrasing like 'within 50 m' or the provided phrase (e.g. 'next to each other').",
           # "distance_relation": "????????.",
            "relspat": "If a specific distance term is provided (e.g. 'side by side', 'next to', 'enclosed by'), use it exactly as stated. Do not invent new phrasing.",
            "contains_relation": "If a 'contains' relation is used, describe it using natural language like 'inside', 'within', 'containing', 'enclosed by', or 'part of'.",
            "properties": "If an entity has descriptive properties (e.g. 'cuisine', 'roof color', 'building material'), express these in a natural way, not as tag names. Example: 'material=wood' → 'wooden'; 'restaurant' + 'cuisine=italian' → 'Italian restaurant'.",
            "no_distances": "If no distances are given, do not imply any spatial relationships or proximity. Avoid words like 'next to', 'near', 'in' or 'close to'. Just search for the objects independently (e.g. 'show me all X, Y and Z').",
        }

        self.typo_templates = [
            "- Typos: Introduce a small number of realistic typos or minor grammar mistakes to simulate a rushed user query.\n",
            "- Typos: Introduce a moderate amount of typos and informal grammar mistakes, as if typed quickly or casually.\n",
            "- Typos: Introduce a large number of typos and grammar issues, making the sentence feel informal, rushed, or even broken in parts.\n",
        ]

        # self.typo_templates = [
        #     "\n\n==Other specifications==\nThe text should contain a {amount} amount of typos.",
        #     "\n\n==Other specifications==\nThe text should contain a {amount} amount of grammar mistakes.",
        #     "\n\n==Other specifications==\nThe text should contain a {amount} amount of typos and grammar mistakes."
        # ]
        # self.typo_amounts = ["small", "medium", "large"]
#         self.ending_template = (
#             "\nPlease take your time and make sure all the provided information is contained in the sentence. You are "
#             "simulating the behavior of an experienced user prompting an online tool. Use short, clear, and natural "
#             "language — avoid filler, overly formal language, over-explaining, or rhetorical phrasing.\n"
#             "Think of how real users would prompt after using the system for a while: concise, factual, and slightly "
#             "varied, but always focused on the core facts. Double check again if all the provided information is used in the "
#             "generated sentence!"
# )

        self.name_regex_templates = ["", "", "", "contains the letters", "begins with the letters",
                                     "ends with the letters"]
        self.phrases_for_numerical_comparison = {
            "<": ["less than", "smaller than", "lower than", "beneath", "under"],
            ">": ["greater than", "more than", "larger than", "above", "over", "at least"]
        }

        self.phrases_desc = ["", "", "", "", "", "", "", " more or less", " approximately", " less than",
                             " no more than", " no less than", " around", " at max", " about", " at least"]

        self.phrases_away = ["away from", "from"]
        self.phrases_radius = ["within DIST", "in a radius of DIST", "no more than DIST from another",
                               "DIST from each other"]
        self.phrases_contains = ["within", "in", "inside", "contained in"]

        self.dist_lookup = {"cm": "centimeters", "m": "meters", "km": "kilometers", "in": "inches", "ft": "feet",
                            "yd": "yards", "mi": "miles"}

        self.distance_writing_styles = ["default", "with_full_metric"]
        self.distance_writing_styles_probs = [1.0 - self.prob_distance_writing_with_full_metric, self.prob_distance_writing_with_full_metric]

    def get_instructions(self, persona: str, writing_style: str, rules: str, input: str, typos: str) -> str:
        '''
        Create the beginning of a prompt by using the beginning template
        '''
        finished_instructions = self.instruction_template.format(persona=persona, style=writing_style, rules=rules, typos=typos, input=input)
        seps = [["comma", "period", "10,000.00"], ["period", "comma", "10.000,00"]][np.random.choice([0, 1])]
        finished_instructions = finished_instructions.format(thousands=seps[0], decimal=seps[1], example=seps[2])
        return finished_instructions

    def typo(self, prob_of_typos: float) -> str:
        '''
        Add specifications for inclusion of typos if randomly selected
        '''
        if np.random.choice([True, False], p=[prob_of_typos, 1 - prob_of_typos]):
            typo_text = np.random.choice(self.typo_templates)
        else:
            typo_text = ""

        return typo_text

    def add_area_prompt(self, area: Area) -> str:
        '''
        Helper to generate area prompt that is appended to search_prompt
        '''
        area_prompt = ""
        if area.type not in ["bbox", "polygon"]:
            area_prompt = "Search area:\n- " + area.value + "\n"
        return area_prompt

    def add_numerical_prompt(self, entity_property: Property) -> str:
        '''
        This helper generates a numerical prompt for numerical properties and properties such as height
        '''
        if not is_number(entity_property.value) and np.random.choice([True, False]):
            metric = self.dist_lookup[entity_property.value.rsplit(" ", 1)[-1]]
            value = entity_property.value.rsplit(" ", 1)[0] + " " + metric
            if np.random.choice([True, False]):
                value = value.replace(" ", "")
        else:
            value = entity_property.value

        if entity_property.operator not in self.phrases_for_numerical_comparison:
            return f": {value}"
        else:
            numerical_phrases = self.phrases_for_numerical_comparison[entity_property.operator]
            np.random.shuffle(numerical_phrases)
            selected_numerical_phrase = numerical_phrases[0]
            return f": {selected_numerical_phrase} {value}"

    def add_name_regex_prompt(self, entity_property: Property) -> str:
        '''
        It is a helper function for name properties such as name, street names
        '''
        selected_name_regex = np.random.choice(self.name_regex_templates)

        if len(entity_property.value) > 1 and len(selected_name_regex) > 0:
            len_substring = np.random.choice(np.arange(1, len(entity_property.value)))
            idx = random.randrange(0, len(entity_property.value) - len_substring + 1)
            entity_property.value = entity_property.value[idx: (idx + len_substring)].strip()

        return f": {selected_name_regex} \"{entity_property.value}\""

    def add_other_non_numerical_prompt(self, entity_property: Property) -> str:
        '''
        handler for core/prop type of properties having no value and properties such as cuisine
        '''
        return f": {entity_property.value}" if entity_property.value else ""

    def add_property_prompt(self, core_prompt: str, entity_properties: List[Property]) -> str:
        for entity_property in entity_properties:
            core_prompt = core_prompt + entity_property.name

            if entity_property.name == 'height' or is_number(entity_property.value):
                core_prompt = core_prompt + self.add_numerical_prompt(entity_property=entity_property)
            elif entity_property.operator == '~':
                core_prompt = core_prompt + self.add_name_regex_prompt(entity_property=entity_property)
            else:
                core_prompt = core_prompt + self.add_other_non_numerical_prompt(entity_property=entity_property)
            core_prompt = core_prompt + ", "

        return core_prompt[:-2]

    def add_relative_spatial_terms(self, relation: Relation, entities: List[Entity]) -> Tuple[str, Distance]:
        '''
        Randomly selects relative spatial term
        '''
        selected_relative_spatial = np.random.choice(self.relative_spatial_terms)

        # select randomly descriptor of relative special
        descriptors_of_relative_spatial_terms = selected_relative_spatial.values
        np.random.shuffle(descriptors_of_relative_spatial_terms)
        selected_relative_spatial_term = descriptors_of_relative_spatial_terms[0]
        generated_prompt, overwritten_distance = self.add_relative_spatial_term_helper(
            selected_relative_spatial_term, relation, selected_relative_spatial, entities)

        return (generated_prompt, overwritten_distance)

    def add_relative_spatial_term_helper(self, selected_relative_spatial_term: str, relation: Relation,
                                         selected_relative_spatial: RelSpatial, entities: List[Entity]):
        for entity in entities:
            if entity.id == relation.target:
                target_ent = normalize_entity_name(entity.name)
            if entity.id == relation.source:
                source_ent = normalize_entity_name(entity.name)
        generated_prompt = (f"- Use this term to describe the spatial relation between the {source_ent} and the "
                            f"{target_ent} (similar to \"X is _ Y\"): {selected_relative_spatial_term}\n")
        overwritten_distance = selected_relative_spatial.distance
        return generated_prompt, overwritten_distance

    def generate_written_word_distance(self, distance: Distance, max_digits: int) -> tuple[Distance, Distance]:
        """
        Generates a random number starting at 100, with the last two digits always being zero, and returns it as both
        a scalar and written number.

        :param metric: the metric of the old distance
        :param max_digits: maximum number of digits allowed
        :return: numeric - new numeric value, written - corresponding number in written words
        """
        digits = randint(1, max_digits - 2)
        low = np.power(10, digits - 1)
        high = np.power(10, digits) - 1

        modified_magnitude = randint(low, high) * 100

        written_magnitude = num2words(modified_magnitude)
        if np.random.choice([True, False]):
            written_magnitude = written_magnitude.replace(",", "")

        numeric = Distance(magnitude=str(modified_magnitude), metric=distance.metric)
        written = Distance(magnitude=written_magnitude, metric=distance.metric)
        return numeric, written

    def add_desc_away_prompt_helper(self, relation: Relation, selected_phrases_desc: str, selected_phrases_away: str,
                                    entities: List[Entity]):
        '''Helper function for generating desc away prompts'''
        distance = self.rewrite_distance(relation.value)

        for entity in entities:
            if entity.id == relation.target:
                target_ent = normalize_entity_name(entity.name)
            if entity.id == relation.source:
                source_ent = normalize_entity_name(entity.name)

        generated_prompt = (f"- The {source_ent} is{selected_phrases_desc} {distance} {selected_phrases_away} "
                            f"the {target_ent}\n")
        return generated_prompt

    def rewrite_distance(self, distance: Distance) -> str:
        magnitude = distance.magnitude
        metric = distance.metric
        selected_task = np.random.choice(self.distance_writing_styles, p=self.distance_writing_styles_probs)

        if magnitude.isdigit():
            remove_whitespace = np.random.choice([False, True], p=[
                1.0 - self.prob_distance_writing_no_whitespace, self.prob_distance_writing_no_whitespace])
        else:
            remove_whitespace = False
        if selected_task == "with_full_metric":
            metric = self.dist_lookup[metric]
            distance_as_str = f"{magnitude}{metric}" if remove_whitespace else f'{magnitude} {metric}'

        else:
            distance_as_str = f"{magnitude}{metric}" if remove_whitespace else f'{magnitude} {metric}'
        return distance_as_str

    def add_desc_away_prompt(self, relation: Relation, entities: List[Entity]) -> str:
        selected_phrases_desc = np.random.choice(self.phrases_desc)
        selected_phrases_away = np.random.choice(self.phrases_away)

        generated_prompt = self.add_desc_away_prompt_helper(relation, selected_phrases_desc, selected_phrases_away,
                                                            entities)
        return generated_prompt

    def add_prompt_for_within_radius_relation(self, distance: Distance) -> str:
        distance = self.rewrite_distance(distance)
        selected_phrase = np.random.choice(self.phrases_radius)
        selected_phrase = selected_phrase.replace('DIST', distance)
        generated_prompt = f"- All objects are {selected_phrase}.\n"
        return generated_prompt

    def add_relation_with_contain(self, relations: List[Relation], entities: List[Entity]) -> Tuple[str, Relations]:
        '''
        This function identifies the objects having containing relationship, collect the remaining ones which have
        individual rels with the other ones.
        :param relations:
        :return: generated_prompt, List[Relation]: list of individual relations
        '''
        contains_phrase = np.random.choice(self.phrases_contains)
        contains_term = np.random.choice(self.contains_terms)

        generated_prompts = []
        positions = []
        for id, relation in enumerate(relations):
            if relation.type == "contains":
                for entity in entities:
                    if entity.id == relation.target:
                        target_ent = normalize_entity_name(entity.name)
                    if entity.id == relation.source:
                        source_ent = normalize_entity_name(entity.name)
                generated_prompts.append(f"- The {target_ent} is {contains_phrase} the {source_ent}, use this term to "
                                         f"describe their relation: {contains_term}\n")
                positions.append(id)

        return (generated_prompts, positions)

class GPTDataGenerator:
    def __init__(self, relative_spatial_terms: List[RelSpatial], contains_terms: List[str],
                 personas: List[str],
                 styles: List[str],
                 prob_no_cluster_distance = 0.5,
                 prob_usage_of_relative_spatial_terms: float = 0.4,
                 prob_usage_of_written_numbers: float = 0.3,
                 prob_of_typos: float = 0.3,
                 prob_distance_writing_with_full_metric: float = 0.1,
                 prob_distance_writing_no_whitespace: float = 0.8,
                 max_dist_digits: int = 5):

        self.relative_spatial_terms = relative_spatial_terms
        self.contains_terms = contains_terms
        self.prob_no_cluster_distance = prob_no_cluster_distance
        self.prob_usage_of_relative_spatial_terms = prob_usage_of_relative_spatial_terms
        self.prob_usage_of_written_numbers = prob_usage_of_written_numbers
        self.prob_of_typos = prob_of_typos
        self.prob_distance_writing_with_full_metric = prob_distance_writing_with_full_metric
        self.prob_distance_writing_no_whitespace = prob_distance_writing_no_whitespace
        self.max_dist_digits = max_dist_digits

        self.phrases_desc = ["", "", "", "", "", "", "", " more or less", " approximately", " less than",
                                 " no more than", " no less than", " around", " at max", " about", " at least"]
        self.phrases_away = ["away from", "from"]
        self.phrases_dist = ["", "distance "]
        self.phrases_dist_relspat = ["spatial relation", "distance"]
        self.phrases_anoth = ["between each other", "to each other", "to another", "between the objects"]

        self.personas = personas
        self.styles = styles
        self.prompt_helper = PromptHelper(relative_spatial_terms=self.relative_spatial_terms,
                                          contains_terms=self.contains_terms,
                                          prob_usage_of_relative_spatial_terms=self.prob_usage_of_relative_spatial_terms,
                                          prob_usage_of_written_numbers=self.prob_usage_of_written_numbers,
                                          prob_distance_writing_with_full_metric=self.prob_distance_writing_with_full_metric,
                                          prob_distance_writing_no_whitespace=self.prob_distance_writing_no_whitespace)

    def update_relation_distance(self, relations: Relations, relation_to_be_updated: Relation, distance: str):
        updated_relations = []
        for relation in relations.relations:
            if relation == relation_to_be_updated:
                relation.value = distance
                updated_relations.append(relation)
            else:
                updated_relations.append(relation)
        return relations.update(relations=updated_relations)

    def edit_cluster_distance(self, entity):
        if np.random.choice([False, True], p=[
            1.0 - self.prob_no_cluster_distance, self.prob_no_cluster_distance]):
            # use_cluster_distance = False
            # use_relative_spatial_terms = False
            # use_written_distance = False

            type = 'none'
            entity_value = None
            written_value = None
        else:
            # use_cluster_distance = True
            use_relative_spatial_terms = np.random.choice([False, True], p=[
                1.0 - self.prob_usage_of_relative_spatial_terms, self.prob_usage_of_relative_spatial_terms])
            use_written_distance = np.random.choice([False, True], p=[
                1.0 - self.prob_usage_of_written_numbers, self.prob_usage_of_written_numbers])

            # In case both relative term and written word are selected, randomly only select one of them
            if use_relative_spatial_terms and use_written_distance:
                if random.choice([True, False]):
                    use_relative_spatial_terms = False
                else:
                    use_written_distance = False
            if use_relative_spatial_terms:
                selected_relative_spatial = np.random.choice(self.relative_spatial_terms)

                # select randomly descriptor of relative special
                descriptors_of_relative_spatial_terms = selected_relative_spatial.values
                np.random.shuffle(descriptors_of_relative_spatial_terms)
                selected_relative_spatial_term = descriptors_of_relative_spatial_terms[0]

                type = 'relspat'
                entity_value = selected_relative_spatial.distance
                written_value = selected_relative_spatial_term
            elif use_written_distance:
                type = 'written'
                entity_value, written_value = self.prompt_helper.generate_written_word_distance(
                    distance=entity.maxDistance, max_digits=self.max_dist_digits)
                written_value = written_value.magnitude + " " + written_value.metric
            else:
                type = 'number'
                entity_value = entity.maxDistance
                written_value = self.prompt_helper.rewrite_distance(entity.maxDistance)

        return entity_value, written_value, type

    def generate_prompt(self, loc_point: LocPoint, persona: str, style: str) -> str:
        '''
        A method that takes the intermediate query representation, and uses it to generate a natural language prompt for
        the GPT API. Different sentence structures are required for the different tasks, for the special tag "count",
        as well as for the different substring searches (beginning, ending, containing, equals).

        :param dict loc_point: The dictionary containing all relevant information for the query
        '''
        feature_tracker = {
            "cluster": False,
            "cluster_distance": False,
            "distance_relation": False,
            "relspat": False,
            "contains_relation": False,
            "properties": False,
            "no_distances": False,
        }
        area = loc_point.area
        entities = loc_point.entities
        relations = loc_point.relations

        core_relation = ''

        cont_prompts = []
        cont_pos = []
        ind_prompts = []
        ind_pos = []
        if relations.type in ["individual_distances_with_contains", "contains_relation"]:
            feature_tracker["contains_relation"] = True

            cont_prompts, cont_pos = self.prompt_helper.add_relation_with_contain(relations.relations, entities)

        if relations.type in ["individual_distances", "individual_distances_with_contains"]:
            feature_tracker["distance_relation"] = True
            ind_prompts, ind_pos, use_relspat = self.individual_prompt_generation(relations, entities)
            if use_relspat:
                feature_tracker["relspat"] = True

        if relations.type in ["individual_distances", "individual_distances_with_contains", "contains_relation"]:
            for pos in range(len(relations.relations)):
                if pos in cont_pos:
                    core_relation += cont_prompts.pop(0)
                elif pos in ind_pos:
                    core_relation += ind_prompts.pop(0)

        elif relations.type == "within_radius":
            feature_tracker["distance_relation"] = True
            core_relation += self.radius_prompt_generation(relations)

        if len(core_relation) > 0:
            core_relation = "Distances:\n" + core_relation
        else:
            feature_tracker["no_distances"] = True
            core_relation = "Distances:\nNo distance is given.\n"

        # Generate object prompt lines, must be after relations so cluster can adapt the possibly updated relation distances
        core_prompt = self.prompt_helper.add_area_prompt(area)
        core_prompt += "Objects:\n"

        for entity_id, entity in enumerate(entities):
            entity_name = normalize_entity_name(entity.name)
            if entity.type == 'nwr':
                core_prompt = core_prompt + "- Obj. " + str(entity_id) + ": " + entity_name
            elif entity.type == 'cluster':
                feature_tracker["cluster"] = True
                entity_value, written_value, type = self.edit_cluster_distance(entity)

                # In case no distance was provided, use distance from relation before (target), unless this is not
                # possible, then fall back to the relation after (source). Don't mention the maxdist in the prompt.
                if type == 'none':
                    if relations.relations:
                        for rel in relations.relations:
                            if rel.target == entity.id and rel.value:
                                entity_value = copy.deepcopy(rel.value)
                                break
                        if not entity_value:
                            for rel in relations.relations:
                                if rel.source == entity.id and rel.value:
                                    entity_value = copy.deepcopy(rel.value)
                                    break
                    if not entity_value:
                        entity_value = Distance(magnitude="50", metric="m")

                    entity.maxDistance = entity_value
                    core_prompt = (core_prompt + "- Obj. " + str(entity_id) + ": " + str(entity.minPoints) + " x " +
                                   entity_name)
                elif type == 'relspat':
                    feature_tracker["relspat"] = True
                    entity.maxDistance = entity_value
                    phrase_dist_relspat = np.random.choice(self.phrases_dist_relspat)
                    phrase_anoth = np.random.choice(self.phrases_anoth)
                    core_prompt = (core_prompt + "- Obj. " + str(entity_id) + ": " + str(entity.minPoints) + " x " +
                                   entity_name + ", use this phrase to describe the " + phrase_dist_relspat + " " +
                                   phrase_anoth + ": " + written_value + " (from/to/of) another")
                else:
                    feature_tracker["cluster_distance"] = True
                    entity.maxDistance = entity_value
                    selected_phrases_desc = np.random.choice(self.phrases_desc)
                    phrases_dist = np.random.choice(self.phrases_dist)
                    phrase_anoth = np.random.choice(self.phrases_anoth)
                    core_prompt = (core_prompt + "- Obj. " + str(entity_id) + ": " + str(entity.minPoints) + " x " +
                                   entity_name + "," + selected_phrases_desc + " " + written_value + " " +
                                   phrases_dist + phrase_anoth)
            if len(entity.properties) > 0:
                feature_tracker["properties"] = True
                core_prompt += " | Properties -> "
                core_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                     entity_properties=entity.properties)
            core_prompt += '\n'

        core_prompt = core_prompt + core_relation

        rules = copy.deepcopy(self.prompt_helper.core_rules)
        for rk in self.prompt_helper.optional_rules:
            if feature_tracker[rk]:
                rules.append(self.prompt_helper.optional_rules[rk])
        rules = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rules))

        instructions = self.prompt_helper.get_instructions(persona=persona, writing_style=style, rules=rules,
                                                typos=self.prompt_helper.typo(self.prob_of_typos), input=core_prompt)

        return loc_point, instructions

    def assign_persona_styles_to_queries(self, num_of_all_persona_style, num_tag_queries):
        persona_style_ids = list(range(num_of_all_persona_style))
        num_tag_queries_ids = list(range(num_tag_queries))

        cycled_persona_style_ids = itertools.cycle(persona_style_ids)
        persona_style_tag_pairs = [(x, next(cycled_persona_style_ids)) for x in num_tag_queries_ids]
        return persona_style_tag_pairs

    def individual_prompt_generation(self, relations, entities):
        use_relspat = False
        indiv_prompt = []
        positions = []
        for id, relation in enumerate(relations.relations):
            if relation.type == "distance":
                cluster_uses_rel_distance = False
                for entity in entities:
                    if entity.type == "cluster" and (entity.id == relation.source or entity.id == relation.target):
                        if entity.maxDistance == relation.value:
                            cluster_uses_rel_distance = True

                use_relative_spatial_terms = np.random.choice([False, True], p=[
                    1.0 - self.prob_usage_of_relative_spatial_terms, self.prob_usage_of_relative_spatial_terms])
                use_written_distance = np.random.choice([False, True], p=[
                    1.0 - self.prob_usage_of_written_numbers, self.prob_usage_of_written_numbers])
                # In case both relative term and written word are selected, randomly only select one of them
                if use_relative_spatial_terms and use_written_distance:
                    if random.choice([True, False]):
                        use_relative_spatial_terms = False
                    else:
                        use_written_distance = False
                if use_relative_spatial_terms and not cluster_uses_rel_distance:
                    use_relspat = True
                    generated_prompt, overwritten_distance = self.prompt_helper.add_relative_spatial_terms(relation,
                                                                                                           entities)
                    indiv_prompt.append(generated_prompt)
                    positions.append(id)
                    self.update_relation_distance(relations=relations,
                                                  relation_to_be_updated=relation,
                                                  distance=overwritten_distance)
                elif use_written_distance and not cluster_uses_rel_distance:
                    numeric_distance, written_distance = self.prompt_helper.generate_written_word_distance(
                        distance=relation.value, max_digits=self.max_dist_digits)

                    written_distance_relation = Relation(type=relation.type, source=relation.source,
                                                         target=relation.target, value=written_distance)
                    indiv_prompt.append(self.prompt_helper.add_desc_away_prompt(written_distance_relation, entities))
                    positions.append(id)
                    self.update_relation_distance(relations=relations,
                                                  relation_to_be_updated=relation,
                                                  distance=numeric_distance)
                else:
                    indiv_prompt.append(self.prompt_helper.add_desc_away_prompt(relation, entities))
                    positions.append(id)
        return (indiv_prompt, positions, use_relspat)

    def radius_prompt_generation(self, relations):
        radius_prompt = ""
        distance = relations.relations[0].value
        use_written_distance = np.random.choice([False, True], p=[
            1.0 - self.prob_usage_of_written_numbers, self.prob_usage_of_written_numbers])
        if use_written_distance:
            numeric_distance, written_distance = self.prompt_helper.generate_written_word_distance(
                distance, max_digits=self.max_dist_digits)
            for relation in relations.relations:
                self.update_relation_distance(relations=relations,
                                              relation_to_be_updated=relation,
                                              distance=numeric_distance)
            radius_prompt += self.prompt_helper.add_prompt_for_within_radius_relation(written_distance)
        else:
            radius_prompt += self.prompt_helper.add_prompt_for_within_radius_relation(distance)
        return radius_prompt

    def generate_prompts(self, tag_queries: List[LocPoint]) -> List[GeneratedPrompt]:
        '''
        A method that takes the intermediate query representation, and uses it to generate a natural language prompt for
        the GPT API. Different sentence structures are required for the different tasks, for the special tag "count",
        as well as for the different substring searches (beginning, ending, containing, equals).

        :param dict tag_queries: The dictionary containing all relevant information for the query
        '''
        all_possible_persona_and_styles = list(itertools.product(self.personas, self.styles))
        random.shuffle(all_possible_persona_and_styles)
        num_tag_queries = len(tag_queries)
        num_of_all_persona_style = len(all_possible_persona_and_styles)
        persona_style_tag_pairs = self.assign_persona_styles_to_queries(num_of_all_persona_style, num_tag_queries)

        results = []
        for tag_id, persona_style_pair in tqdm(persona_style_tag_pairs, total=num_tag_queries):
            persona, style = all_possible_persona_and_styles[persona_style_pair]
            imr_sample = tag_queries[tag_id]
            imr_sample, prompt = self.generate_prompt(imr_sample, persona, style)
            results.append(GeneratedPrompt(query=imr_sample, prompt=prompt, style=style, persona=persona))

        return results

    def generate_sentences(self, generated_prompts, output_gpt_generations_temp) -> List[GeneratedIMRSentence]:
        generated_sentences = []
        for generated_prompt in tqdm(generated_prompts, total=len(generated_prompts)):
            generated_sentence = self.generate_sentence(generated_prompt)

            generated_imr_sentence = dict(
                query=generated_prompt["query"],
                prompt=generated_prompt["prompt"],
                style=generated_prompt["style"],
                persona=generated_prompt["persona"],
                sentence=generated_sentence
            )
            generated_sentences.append(generated_imr_sentence)

            write_dict_output(generated_sentences, output_gpt_generations_temp, True)
        return generated_sentences

    def generate_sentence(self, generated_prompt: GeneratedPrompt) -> str:
        generated_sentence = request_openai(prompt=generated_prompt["prompt"])
        return generated_sentence


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    parser = ArgumentParser()
    parser.add_argument('--relative_spatial_terms_path', help='Path for the relative spatial term definitions', required=True)
    parser.add_argument('--contains_terms_path', help='Path for the contains terms', required=True)
    parser.add_argument('--tag_query_file', required=True)
    parser.add_argument('--output_gpt_generations', required=True)
    parser.add_argument('--output_prompt_generations', required=True)
    parser.add_argument('--persona_path', required=True)
    parser.add_argument('--styles_path', required=True)
    parser.add_argument('--prob_no_cluster_distance', type=float, default=0.5)
    parser.add_argument('--prob_usage_of_relative_spatial_terms', type=float, default=0.4)
    parser.add_argument('--prob_usage_of_written_numbers', type=float, default=0.3)
    parser.add_argument('--prob_of_typos', type=float, default=0.3)
    parser.add_argument('--prob_distance_writing_no_whitespace', type=float, default=0.3)
    parser.add_argument('--prob_distance_writing_with_full_metric', type=float, default=0.3)
    parser.add_argument('--max_dist_digits', type=int, default=5)
    parser.add_argument('--generate_prompts', action='store_true',
                        help='Activate it if you want to generate prompts that will be sent to LLM')
    parser.add_argument('--generate_sentences', action='store_true',
                        help='Activate it if you want to generate sentences with GPT')
    parser.add_argument('--translate_to_yaml', action='store_true',
                        help='Activate it if you want to translate the queries to the final YAML format')
    parser.add_argument('--save_yaml_csv', action='store_true',
                        help='Activate if you want to save a CSV file on top of the JSONL for better readability')

    args = parser.parse_args()

    output_prompt_generations = args.output_prompt_generations
    output_gpt_generations = args.output_gpt_generations
    relative_spatial_terms_path = args.relative_spatial_terms_path
    contains_terms_path = args.contains_terms_path
    persona_path = args.persona_path
    styles_path = args.styles_path
    tag_query_file = args.tag_query_file

    # probabilities
    prob_no_cluster_distance = args.prob_no_cluster_distance
    prob_usage_of_relative_spatial_terms = args.prob_usage_of_relative_spatial_terms
    prob_usage_of_written_numbers = args.prob_usage_of_written_numbers
    prob_of_typos = args.prob_of_typos
    prob_distance_writing_with_full_metric = args.prob_distance_writing_with_full_metric
    prob_distance_writing_no_whitespace = args.prob_distance_writing_no_whitespace

    max_dist_digits = args.max_dist_digits
    generate_sentences = args.generate_sentences
    generate_prompts = args.generate_prompts
    translate_to_yaml = args.translate_to_yaml
    save_yaml_csv = args.save_yaml_csv

    rel_spatial_terms = load_rel_spatial_terms(relative_spatial_terms_path=relative_spatial_terms_path)
    contains_terms = load_contains_terms(contains_terms_path=contains_terms_path)
    personas = load_list_of_strings(list_of_strings_path=persona_path)
    styles = load_list_of_strings(list_of_strings_path=styles_path)

    gen = GPTDataGenerator(relative_spatial_terms=rel_spatial_terms,
                           contains_terms=contains_terms,
                           personas=personas,
                           styles=styles,
                           prob_no_cluster_distance=prob_no_cluster_distance,
                           prob_usage_of_relative_spatial_terms=prob_usage_of_relative_spatial_terms,
                           prob_usage_of_written_numbers=prob_usage_of_written_numbers,
                           prob_distance_writing_no_whitespace=prob_distance_writing_no_whitespace,
                           prob_distance_writing_with_full_metric=prob_distance_writing_with_full_metric,
                           prob_of_typos=prob_of_typos,
                           max_dist_digits=max_dist_digits)

    generated_queries = None
    generated_queries_yaml = None
    if generate_prompts:
        with open(tag_query_file, "r") as f:
            candidate_loc_points = [LocPoint(**json.loads(each_line)) for each_line in f]
        generated_queries = gen.generate_prompts(candidate_loc_points)
        write_output(generated_queries, output_prompt_generations)

        generated_queries_yaml = translate_queries_to_yaml(generated_queries)
        write_dict_output(generated_queries_yaml, output_prompt_generations, True)

        if save_yaml_csv:
            write_output_csv(generated_queries_yaml, output_prompt_generations)

    if generate_sentences:
        if not generated_queries_yaml:
            with open(output_prompt_generations, "r") as f:
                generated_queries = [GeneratedPrompt(**json.loads(each_line)) for each_line in f]
            generated_queries_yaml = translate_queries_to_yaml(generated_queries)
            
        parent_dir = Path(output_gpt_generations).parent
        filename_without_extension = Path(output_gpt_generations).stem
        file_extension = Path(output_gpt_generations).suffix
        output_gpt_generations_temp = parent_dir / (filename_without_extension + "_temp" + file_extension)

        generated_sentences = gen.generate_sentences(generated_queries_yaml, output_gpt_generations_temp)
        write_dict_output(generated_sentences, output_gpt_generations, True)

        if os.path.exists(output_gpt_generations_temp):
            os.remove(output_gpt_generations_temp)

        if save_yaml_csv:
            write_output_csv(generated_sentences, output_gpt_generations, True)
