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

"""
Generate natural-language search queries (and optional sentences via an LLM)
from structured scene descriptions (areas, entities, relations, properties).

Pipeline overview:
- Build prompt text from IMR objects (`LocPoint`, `Entity`, `Relations`, etc.).
- Add optional style/persona/typos and distance-writing variations.
- (Optionally) call an LLM with retry/backoff to produce final sentences.
- Save prompts and/or generated sentences.

CLI flags allow generating prompts only, sentences only, or both.
"""

# define a retry decorator
def retry_with_exponential_backoff(
        func,
        initial_delay: float = 1,
        exponential_base: float = 2,
        jitter: bool = True,
        max_retries: int = 10,
        errors: tuple = (openai.RateLimitError,),
):
    """Wrapper around `CLIENT.chat.completions.create` with backoff (via decorator).

    Args:
        **kwargs: Keyword args forwarded to the OpenAI Chat Completions API.

    Returns:
        API response object.
    """
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
    """Wrapper around `CLIENT.chat.completions.create` with backoff (via decorator).

    Args:
        **kwargs: Keyword args forwarded to the OpenAI Chat Completions API.

    Returns:
        API response object.
    """
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

def request_openai(prompt: str) -> str:
    """Send a single-turn chat completion request and return the text content.

    Args:
        prompt: User message content.

    Returns:
        First choice message text.
    """
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


def is_number(s: str) -> bool:
    """Return True if string `s` parses as a float, else False."""
    if not s:
        return False
    try:
        float(s)  # Try converting the string to a float
        return True
    except ValueError:
        return False


def remove_surrounding_double_quotes(text: str) -> str:
    """Strip matching leading/trailing double quotes from `text`, if present."""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1]
    return text


def post_processing(text: str) -> str:
    """Basic cleanup for model output lines.

    Replaces CRs, trims, removes literal 'User:' prefix, and strips wrapping quotes.
    """
    text = text.replace('\r', '').strip()
    text = text.replace("User:", "")
    text = remove_surrounding_double_quotes(text)
    return text


def load_rel_spatial_terms(relative_spatial_terms_path: str) -> List[RelSpatial]:
    """Load relative spatial terms CSV → `RelSpatial` list.

    CSV columns: `Dist` (e.g., '3 m'), `Vals` (comma-separated descriptors).

    Args:
        relative_spatial_terms_path: Path to CSV.

    Returns:
        List of `RelSpatial` with parsed distance and term values.
    """
    relative_spatial_terms = pd.read_csv(relative_spatial_terms_path, sep=',').to_dict(orient='records')
    processed_rel_spatial_terms = []
    for relative_spatial_term in relative_spatial_terms:
        values = list(map(lambda x: x.rstrip().strip(), relative_spatial_term['Vals'].split(',')))
        relative_spatial_term_dist = relative_spatial_term['Dist'].split()
        processed_rel_spatial_terms.append(RelSpatial(distance=Distance(magnitude=relative_spatial_term_dist[0], metric=relative_spatial_term_dist[1]), values=values))
    return processed_rel_spatial_terms

def load_contains_terms(contains_terms_path: str) -> List[str]:
    """Load a single-row CSV of comma-separated 'contains' phrases.

    Args:
        contains_terms_path: Path to CSV.

    Returns:
        List of phrases (strings).
    """
    df = pd.read_csv(contains_terms_path, sep=',')  # assuming tab-separated
    contains_terms = df.iloc[0, 1]  # row 0 (after header), column 1
    return contains_terms.split(', ')

def load_list_of_strings(list_of_strings_path: str) -> List[str]:
    """Load a text file as a list of stripped lines.

    Useful for personas and styles.

    Args:
        list_of_strings_path: Path to `.txt` file.

    Returns:
        List of strings (one per line).
    """
    with open(list_of_strings_path, 'r') as f:
        list_of_strings = f.readlines()
        list_of_strings = list(map(lambda x: x.rstrip().strip(), list_of_strings))
    return list_of_strings


def normalize_entity_name(entity_name):
    """Normalize entity names for rendering in prompts (placeholder for tweaks)."""
    # if 'brand:' in entity_name:
    #     entity_name = entity_name.replace('brand:', '')

    return entity_name


class PromptHelper:
    """Helper utilities to compose prompt text from IMR data.

    Responsibilities:
      - Build instruction blocks and core rules.
      - Render areas, entities, properties, relations.
      - Randomize phrasing (typos, numeric vs written distances, etc.).

    Most methods return short text snippets used by the main generator.
    """
    def __init__(self, relative_spatial_terms, contains_terms, prob_usage_of_relative_spatial_terms,
                 prob_usage_of_written_numbers, prob_distance_writing_with_full_metric,
                 prob_distance_writing_no_whitespace):
        """Initialize with reference lists and probabilities."""
        self.relative_spatial_terms = relative_spatial_terms
        self.contains_terms = contains_terms
        self.prob_usage_of_relative_spatial_terms = prob_usage_of_relative_spatial_terms
        self.prob_usage_of_written_numbers = prob_usage_of_written_numbers
        self.prob_distance_writing_no_whitespace = prob_distance_writing_no_whitespace
        self.prob_distance_writing_with_full_metric = prob_distance_writing_with_full_metric


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
        """Render the instruction block used as the model prompt.

        Args:
            persona: Persona label.
            writing_style: Style label.
            rules: Bullet list of rules to include (stringified).
            input: The 'scene' block (area/objects/distances).
            typos: Optional 'typos' directive snippet.

        Returns:
            Full instruction text for the LLM.
        """
        finished_instructions = self.instruction_template.format(persona=persona, style=writing_style, rules=rules, typos=typos, input=input)
        seps = [["comma", "period", "10,000.00"], ["period", "comma", "10.000,00"]][np.random.choice([0, 1])]
        finished_instructions = finished_instructions.format(thousands=seps[0], decimal=seps[1], example=seps[2])
        return finished_instructions

    def typo(self, prob_of_typos: float) -> str:
        """Optionally include a typo directive based on probability."""
        if np.random.choice([True, False], p=[prob_of_typos, 1 - prob_of_typos]):
            typo_text = np.random.choice(self.typo_templates)
        else:
            typo_text = ""

        return typo_text

    def add_area_prompt(self, area: Area) -> str:
        """Format the 'Search area' part of the scene, if applicable."""
        area_prompt = ""
        if area.type not in ["bbox", "polygon"]:
            area_prompt = "Search area:\n- " + area.value + "\n"
        return area_prompt

    def add_numerical_prompt(self, entity_property: Property) -> str:
        """Format numeric-like properties (height, distances, etc.)."""
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
        """Format name-like properties with optional substring constraint."""
        selected_name_regex = np.random.choice(self.name_regex_templates)

        if len(entity_property.value) > 1 and len(selected_name_regex) > 0:
            len_substring = np.random.choice(np.arange(1, len(entity_property.value)))
            idx = random.randrange(0, len(entity_property.value) - len_substring + 1)
            entity_property.value = entity_property.value[idx: (idx + len_substring)].strip()

        return f": {selected_name_regex} \"{entity_property.value}\""

    def add_other_non_numerical_prompt(self, entity_property: Property) -> str:
        """Format other non-numeric properties (e.g., cuisine, material)."""
        return f": {entity_property.value}" if entity_property.value else ""

    def add_property_prompt(self, core_prompt: str, entity_properties: List[Property]) -> str:
        """Append a formatted property list to `core_prompt`."""
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
        """Create a 'use this term to describe the spatial relation' instruction."""
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
        """Helper that pairs a selected term with relation source/target names."""
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
        """Randomly generate a large rounded integer and return numeric and written variants.

        The magnitude is a multiple of 100 with a digit length up to `max_digits`.

        Args:
            distance: Original distance whose metric is preserved.
            max_digits: Max total digits (before the trailing two zeros).

        Returns:
            (numeric, written) as `Distance` objects.
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
        """Format a line like '- The A is ~ 3 m away from the B'."""
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
        """Return a string representation of a distance, with optional full metric and whitespace tweaks."""
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
        """Create a distance line using 'away from' phrasing and optional descriptors."""
        selected_phrases_desc = np.random.choice(self.phrases_desc)
        selected_phrases_away = np.random.choice(self.phrases_away)

        generated_prompt = self.add_desc_away_prompt_helper(relation, selected_phrases_desc, selected_phrases_away,
                                                            entities)
        return generated_prompt

    def add_prompt_for_within_radius_relation(self, distance: Distance) -> str:
        """Create a line describing a shared radius constraint among all objects."""
        distance = self.rewrite_distance(distance)
        selected_phrase = np.random.choice(self.phrases_radius)
        selected_phrase = selected_phrase.replace('DIST', distance)
        generated_prompt = f"- All objects are {selected_phrase}.\n"
        return generated_prompt

    def add_relation_with_contain(self, relations: List[Relation], entities: List[Entity]) -> Tuple[str, Relations]:
        """Extract and format 'contains' relations as instruction lines.

        Args:
            relations: Relations for the scene.
            entities: Entities referenced by the relations.

        Returns:
            (generated_prompts, positions) where positions are indices in `relations`.
        """
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
    """High-level generator for prompts and LLM-produced sentences.

    Orchestrates:
      - Converting IMR data to prompt text (via `PromptHelper`)
      - Sampling stylistic variations (personas, styles, typos)
      - Optionally converting relation/cluster distances to written words or rel-spatial phrases
      - Calling the LLM to obtain sentences

    Args control probabilities for optional behaviors.
    """
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
        """Initialize generator with resources and probability knobs."""

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
        """Mutate a `Relations` object by replacing one relation's distance value."""
        updated_relations = []
        for relation in relations.relations:
            if relation == relation_to_be_updated:
                relation.value = distance
                updated_relations.append(relation)
            else:
                updated_relations.append(relation)
        return relations.update(relations=updated_relations)

    def edit_cluster_distance(self, entity):
        """Possibly assign a cluster distance: none, relative-spatial, written number, or numeric.

        Returns:
            (entity_value, written_value, type) where type ∈ {'none','relspat','written','number'}.
        """
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
        """Convert a single IMR query (`LocPoint`) into an instruction prompt.

        The prompt includes:
          - Optional search area
          - Objects with properties (clusters rendered with count and optional distance hints)
          - Distances section (individual distances, within-radius, contains), with optional rel-spatial phrasing
          - Rules + persona/style + typo directives

        Args:
            loc_point: IMR scene input.
            persona: Persona to apply.
            style: Writing style to apply.

        Returns:
            (loc_point, prompt_text) where prompt_text is sent to the LLM.
        """
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
        """Cycle persona/style pairs across queries to distribute variety."""
        persona_style_ids = list(range(num_of_all_persona_style))
        num_tag_queries_ids = list(range(num_tag_queries))

        cycled_persona_style_ids = itertools.cycle(persona_style_ids)
        persona_style_tag_pairs = [(x, next(cycled_persona_style_ids)) for x in num_tag_queries_ids]
        return persona_style_tag_pairs

    def individual_prompt_generation(self, relations, entities):
        """Render lines for each 'distance' relation, possibly converting to rel-spatial or written numbers.

        Returns:
            (lines, positions, used_relspat: bool)
        """
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
        """Render the shared-radius variant; optionally convert the numeric to written words."""
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
        """Build instruction prompts from IMR queries, with persona/style assignments.

        Args:
            tag_queries: List of `LocPoint` IMR scenes.

        Returns:
            List of `GeneratedPrompt` items (IMR + prompt + persona/style).
        """
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
        """Call the LLM for each prompt and stream interim results to a temp file.

        Args:
            generated_prompts: Iterable of dicts or `GeneratedPrompt`-like items.
            output_gpt_generations_temp: Path for temp JSONL output.

        Returns:
            List of generated sentence dicts (query/prompt/style/persona/sentence).
        """
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
    """CLI: configure inputs and generation options, then run the pipeline."""
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
