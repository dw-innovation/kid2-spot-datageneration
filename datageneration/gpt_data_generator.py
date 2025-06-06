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
MODEL = os.getenv('MODEL', 'gpt-4.1-mini') #gpt-4.1-mini  #gpt-4.1-nano
    # https://openai.com/index/gpt-4-1/
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.3))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 4096))

CLIENT = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"], organization=os.environ["OPENAI_ORG"]
)
# CLIENT = OpenAI(
#     api_key=os.getenv("LLM_API_KEY"),
#     base_url="https://llm-hub.dw.com/openai"
# )

def request_openai(prompt):
    response = chatcompletions_with_backoff(
        model=MODEL,  # "gpt-4",
        temperature=TEMPERATURE,
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
    relative_spatial_terms = pd.read_csv(relative_spatial_terms_path, sep=';').to_dict(orient='records')
    processed_rel_spatial_terms = []
    for relative_spatial_term in relative_spatial_terms:
        values = list(map(lambda x: x.rstrip().strip(), relative_spatial_term['Vals'].split(',')))
        relative_spatial_term_dist = relative_spatial_term['Dist'].split()
        processed_rel_spatial_terms.append(RelSpatial(distance=Distance(magnitude=relative_spatial_term_dist[0], metric=relative_spatial_term_dist[1]), values=values))
    return processed_rel_spatial_terms


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

    def __init__(self, relative_spatial_terms, prob_usage_of_written_numbers, prob_distance_writing_with_full_metric, prob_distance_writing_no_whitespace):
        self.prob_usage_of_written_numbers = prob_usage_of_written_numbers
        self.prob_distance_writing_no_whitespace = prob_distance_writing_no_whitespace
        self.prob_distance_writing_with_full_metric = prob_distance_writing_with_full_metric
        self.relative_spatial_terms = relative_spatial_terms
        self.beginning_template = (
            # "Generate one or more sentences simulating a user using a natural language interface "
            # "for an AI geolocation search tool that finds locations based on descriptions of "
            # "objects and their spatial relations. Each object has one main descriptor and "
            # "optionally additional properties. All properties must be put in a logical connection "
            # "to the object. Objects can either be single instances, or clusters of multiple of one object "
            # "which are located in a specific distance radius (e.g. \"three houses next to/within 10m of "
            # "each other\").\n"
            # "Mention the area, cover all entities and their respective properties, and describe "
            # "the respective relations. Stick to the descriptions of entities and relations "
            # "provided and don’t add anything. When describing names or brand (names), be creative in "
            # "your phrasing (examples being a \"book store of brand Thalia\" vs. \"a Thalia book store\", "
            # "or simply e.g. \"a Thalia\" if the type of object is not given). "
            # "Stick to the values of each relation. Distances always refer to a maximum distance. "
            # "If no distance is given, do not use any terms such as close, near, create sentences such as \"find a house and a restaurant\". "
            # "Vary your phrasing. Do not affirm this request and return nothing but the answer.\n\n "
            # "You are an assistant that writes short, natural-sounding user queries based on structured geographic data in YAML format.
            "You are an assistant that generates short, natural-sounding user queries based on structured geographic "
            "data in the form of scene descriptions.\n\n"
            "Imagine you're an investigative journalist or fact-checker, looking at an image or video of a real-world scene, "
            "and you're trying to describe what you see — the objects, places, and how they relate to one another. Your goal is "
            "to write what a regular person might type into a search box to describe or explore that scene.\n\n"
            "The scene description provides a list of entities (e.g., landmarks, buildings, places, or objects), their properties, and how "
            "they are spatially related.\n"
            "Your task is to turn this into a natural, casual sentence or query — not a literal translation "
            "of the scene description.\n\n"
            "Here’s how to approach it:\n"
            "- Focus on the scene, not the data format. Don't copy the structure or terminology of the input. Instead, write as if "
            "you were describing the real-world layout to someone else.\n"
            "- Use casual, human phrasing. Avoid technical terms like \"entity\", \"property\", or \"OSM key\".\n"
            "- Make sure to correctly use the entity information in the sentence and use ALL available information:"
            "  - Entities can either be a single entity (e.g. \"- Obj. 0: viewpoint\", i.e. a viewpoint), or a cluster of multiple "
            "of one type (e.g. \"- Obj. 1: 3 x bench\", i.e. three benches).\n"
            "  - If a cluster has no distance value, just use it like an entity in the sentence with the number of "
            " occurences mentioned (e.g. \"three benches\"). \n"
            "  - A cluster can also have a specified distance value between the entities (e.g. \"- Obj. 0: 2 x house, "
            "at max 50 m to another\" -> In the sentence (example phrasing): \"two houses within 50 m\") \n"
            "  - Important: If there is a distance specified for a cluster, the distance value MUST be used in the "
            "sentence!! This can either be a distance value (see example above), or relative spatial terms (e.g. "
            " \"five foutains next to another\"). \n"
            "  - A cluster distance is different from a distance relation. Distance relations (if used) come in a separate "
            "section marked as \"Distance\", cluster distances are part of object definitions. The cluster distance is only "
            "between the multiple instances of the same object. A cluster can have a distance between its instances, and separately "
            "relations that define the distance of the cluster to other objects/cluster. The phrase must include both "
            "if both are given (e.g. \"3 houses in a radius of 30 m, which are 100 m from a fountain\" or \"a church next to two "
            "parks that are nearby another). \n"
            # "  - If a cluster has no distance value, the distance from the associated relations should be used "
            # "(e.g. \"- Obj. 1: 3 x bench\" & \"- The three benches are 50 m from the park\" -> maxDistance: 50 m.\n"
            # "  - If no distance is specified for a cluster and there is no associated distance in the relations, "
            # "default to 50 m.\n"
            "- Translate tags into natural language. For example:\n"
            "  - Entity \"brand:Thalia\" → \"a Thalia\"\n"
            "  - Entity \"cafe\" + Property \"brand~Eiffel\" → \"an Eiffel café\"\n"
            "  - Entity \"restaurant\" + Property \"cuisine~italian\" → \"an Italian restaurant\"\n"
            "  - Property \"building:material=wood\" → \"made from wood\"\n"
            "  - Property \"roof:colour=red\" → \"with a red roof\"\n\n"
            "- Always reflect spatial relationships exactly as stated in the scene description:\n"
            "  - If a distance is given, treat it as a maximum.\n"
            "  - If a relation has a specified phrase (e.g., \"next to\", \"surrounded by\"), use that exact phrase — don’t invent alternatives.\n"
            "  - If a contains relation is given, use phrases like \"containing\", \"with\" and \"in(side)\" to describe the spatial relation.\n" 
            "  - If no relation is provided, do not imply one (in general, avoid terms like \"with\", \"near\" or \"close to\" if not explicitly mentioned).\n\n"
            "- Use number formatting like this: {thousands} for thousands separators and {decimal} for decimals. Example: {example}.\n"
            "- Avoid repetition in phrasing across outputs. In general be direct, but include natural variation — "
            "some sentences can be a bit longer or have more detail; while the tendency is to be short and to the point.\n"
            "- Only use the provided information about the scene — and use **all** of it! Double check that all details, including cluster "
            "distances and properties, are used in the generated sentence!\n"
            "- You must use the same alphabet as used in the provided data. Do not change them to their english version in generated sentence if the "
            "original used a non-latin alphabet, or the other way around.\n"
            "- Always use metrics exactly as specified in distance information, do not convert them to a different unit.\n"
            "- Do not generate why/what/how type questions, only instructions.\n\n"
            # "- If an entity/property combo is obviously nonsensical (e.g., a toilet or a street with a cuisine, a cliff "
            # "with a brand etc.), no sentence should be generated. This is only related to the entity and property names, "
            # "unrealistic numeric values like height or number of floor are acceptable. In nonsensical cases, return only:\n "
            # "`UNREALISTIC COMBINATION`\n\n"
            "==Persona==\n{persona} \n\n ==Style==\n{style}""")
        self.typo_templates = [
            "\n\n==Other specifications==\nThe text should contain a {amount} amount of typos.",
            "\n\n==Other specifications==\nThe text should contain a {amount} amount of grammar mistakes.",
            "\n\n==Other specifications==\nThe text should contain a {amount} amount of typos and grammar mistakes."
        ]
        self.typo_amounts = ["small", "medium", "large"]
        self.ending_template = (
            "\nPlease take your time and make sure all the provided information is contained in the sentence. You are "
            "simulating the behavior of an experienced user prompting an online tool. Use short, clear, and natural "
            "language — avoid filler, overly formal language, over-explaining, or rhetorical phrasing.\n"
            "Think of how real users would prompt after using the system for a while: concise, factual, and slightly "
            "varied, but always focused on the core facts. Double check again if all the provided information is used in the "
            "generated sentence!"
)
        self.search_template = "\n\n==Input==\n"

        self.predefined_places = ["a place", "an area", "a location"]
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

    def beginning(self, persona: str, writing_style: str, ) -> str:
        '''
        Create the beginning of a prompt by using the beginning template
        '''
        seps = [["comma", "period", "10,000.00"], ["period", "comma", "10.000,00"]][np.random.choice([0, 1])]
        return self.beginning_template.format(persona=persona, style=writing_style, thousands=seps[0], decimal=seps[1],
                                              example=seps[2])

    def typo(self, prob_of_typos: float) -> str:
        '''
        Add specifications for inclusion of typos if randomly selected
        '''
        if np.random.choice([True, False], p=[prob_of_typos, 1 - prob_of_typos]):
            typo_text = np.random.choice(self.typo_templates).replace('{amount}', np.random.choice(self.typo_amounts))
        else:
            typo_text = ""

        return typo_text

    def ending(self) -> str:
        '''
        Create the ending of a prompt by using the ending template
        '''
        return self.ending_template

    def search_query(self, beginning_prompt: str):
        '''
        Append the beginning prompt with search phrase. The search phrase is randomly chosen among the search templates.
        If search templates contain {place}, it randomly selects a place from predefined_places
        '''
        return beginning_prompt + self.search_template

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
        selected_phrase = np.random.choice(self.phrases_contains)

        generated_prompts = []
        positions = []
        for id, relation in enumerate(relations):
            if relation.type == "contains":
                for entity in entities:
                    if entity.id == relation.target:
                        target_ent = normalize_entity_name(entity.name)
                    if entity.id == relation.source:
                        source_ent = normalize_entity_name(entity.name)
                generated_prompts.append(f"- The {target_ent} is {selected_phrase} the {source_ent}\n")
                positions.append(id)

        return (generated_prompts, positions)



class GPTDataGenerator:
    def __init__(self, relative_spatial_terms: List[RelSpatial], personas: List[str],
                 styles: List[str],
                 prob_no_cluster_distance = 0.5,
                 prob_usage_of_relative_spatial_terms: float = 0.4,
                 prob_usage_of_written_numbers: float = 0.3,
                 prob_of_typos: float = 0.3,
                 prob_distance_writing_with_full_metric: float = 0.1,
                 prob_distance_writing_no_whitespace: float = 0.8,
                 max_dist_digits: int = 5):

        self.relative_spatial_terms = relative_spatial_terms
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
        area = loc_point.area
        entities = loc_point.entities
        relations = loc_point.relations

        beginning = self.prompt_helper.beginning(persona=persona, writing_style=style)
        beginning = beginning + self.prompt_helper.typo(self.prob_of_typos)
        search_prompt = self.prompt_helper.search_query(beginning)

        core_relation = ''

        cont_prompts = []
        cont_pos = []
        ind_prompts = []
        ind_pos = []
        if relations.type in ["individual_distances_with_contains", "contains_relation"]:
            cont_prompts, cont_pos = self.prompt_helper.add_relation_with_contain(relations.relations, entities)

        if relations.type in ["individual_distances", "individual_distances_with_contains"]:
            ind_prompts, ind_pos = self.individual_prompt_generation(relations, entities)

        if relations.type in ["individual_distances", "individual_distances_with_contains", "contains_relation"]:
            for pos in range(len(relations.relations)):
                if pos in cont_pos:
                    core_relation += cont_prompts.pop(0)
                elif pos in ind_pos:
                    core_relation += ind_prompts.pop(0)

        elif relations.type == "within_radius":
            core_relation += self.radius_prompt_generation(relations)

        if len(core_relation) > 0:
            core_relation = "Distances:\n" + core_relation
        else:
            core_relation = "Distances:\nNo distance is given.\n"

        # Generate object prompt lines, must be after relations so cluster can adapt the possibly updated relation distances
        core_prompt = self.prompt_helper.add_area_prompt(area)
        core_prompt += "Objects:\n"

        for entity_id, entity in enumerate(entities):
            entity_name = normalize_entity_name(entity.name)
            if entity.type == 'nwr':
                core_prompt = core_prompt + "- Obj. " + str(entity_id) + ": " + entity_name
            elif entity.type == 'cluster':
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
                    entity.maxDistance = entity_value
                    phrase_dist_relspat = np.random.choice(self.phrases_dist_relspat)
                    phrase_anoth = np.random.choice(self.phrases_anoth)
                    core_prompt = (core_prompt + "- Obj. " + str(entity_id) + ": " + str(entity.minPoints) + " x " +
                                   entity_name + ", use this phrase to describe the " + phrase_dist_relspat + " " +
                                   phrase_anoth + ": " + written_value + " (from/to/of) another")
                else:
                    entity.maxDistance = entity_value
                    selected_phrases_desc = np.random.choice(self.phrases_desc)
                    phrases_dist = np.random.choice(self.phrases_dist)
                    phrase_anoth = np.random.choice(self.phrases_anoth)
                    core_prompt = (core_prompt + "- Obj. " + str(entity_id) + ": " + str(entity.minPoints) + " x " +
                                   entity_name + "," + selected_phrases_desc + " " + written_value + " " +
                                   phrases_dist + phrase_anoth)
            if len(entity.properties) > 0:
                core_prompt += " | Properties -> "
                core_prompt = self.prompt_helper.add_property_prompt(core_prompt=core_prompt,
                                                                     entity_properties=entity.properties)
            core_prompt += '\n'

        core_prompt = core_prompt + core_relation
        core_prompt = search_prompt + core_prompt + self.prompt_helper.ending()
        return loc_point, core_prompt

    def assign_persona_styles_to_queries(self, num_of_all_persona_style, num_tag_queries):
        persona_style_ids = list(range(num_of_all_persona_style))
        num_tag_queries_ids = list(range(num_tag_queries))

        cycled_persona_style_ids = itertools.cycle(persona_style_ids)
        persona_style_tag_pairs = [(x, next(cycled_persona_style_ids)) for x in num_tag_queries_ids]
        return persona_style_tag_pairs

    def individual_prompt_generation(self, relations, entities):
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
        return (indiv_prompt, positions)

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
    parser.add_argument('--relative_spatial_terms_path', help='Path for the relative spats', required=True)
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
    personas = load_list_of_strings(list_of_strings_path=persona_path)
    styles = load_list_of_strings(list_of_strings_path=styles_path)

    gen = GPTDataGenerator(relative_spatial_terms=rel_spatial_terms,
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
