import copy
import itertools
import json
import numpy as np
import openai
import os
import pandas as pd
import yaml
from argparse import ArgumentParser
from dotenv import load_dotenv
from num2words import num2words
from openai import OpenAI
from pathlib import Path
from random import randint
from tqdm import tqdm
from typing import List, Tuple

from datageneration.data_model import RelSpatial, LocPoint, Area, Property, Relation, Relations, GeneratedPrompt, \
    GeneratedIMRSentence
from datageneration.utils import (add_yaml_to_filename, write_output, write_dict_output, write_output_csv,
                                  translate_queries_to_yaml, clean_up_query)

load_dotenv()

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
MODEL = os.getenv('MODEL', 'gpt-4o')
TEMPERATURE = float(os.getenv('TEMPERATURE', 0.7))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', 4096))

CLIENT = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"], organization=os.environ["OPENAI_ORG"]
)


def request_openai(system_prompt: str, user_prompt: str):
    response = chatcompletions_with_backoff(
        model=MODEL,  # "gpt-4",
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    text = response.choices[0].message.content
    return text


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
    relative_spatial_terms = pd.read_csv(relative_spatial_terms_path).to_dict(orient='records')
    processed_rel_spatial_terms = []
    for relative_spatial_term in relative_spatial_terms:
        values = list(map(lambda x: x.rstrip().strip(), relative_spatial_term['Vals'].split(',')))
        processed_rel_spatial_terms.append(RelSpatial(distance=relative_spatial_term['Dist'], values=values))
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


class PromptHelper:
    '''
    It is a helper class for prompt generation. It has templates and functions for paraphrasing prompts.
    '''

    def __init__(self, relative_spatial_terms):
        self.relative_spatial_terms = relative_spatial_terms
        self.beginning_template = """Act as a {persona}: Return a sentence simulating a user using a natural language interface to search for specific geographic locations. Do not affirm this request and return nothing but the answers.\nWrite the search request {style}."""
        self.search_templates = [
            "\nThe sentence must use all of the following search criteria:\n",
            "\nThe user is searching for {place} that fulfills the following search criteria:\n",
        ]
        self.predefined_places = ["a place", "an area", "a location"]
        self.name_regex_templates = ["=", "=", "=", "contains the letters", "begins with the letters",
                                     "ends with the letters"]
        self.phrases_for_numerical_comparison = {
            "<": ["<", "<", "<", "<", "less than", "smaller than", "lower than", "beneath", "under"],
            ">": [">", ">", ">", ">", "greater than", "more than", "larger than", "above", "over", "at least"]
        }

        self.phrases_desc = ["", "", "", "", "", "", "", "more or less", "approximately", "less than",
                             "no more than", "no less than", "around", "at max", "about", "at least"]

        self.phrases_away = ["away", "away from", "from"]
        self.phrases_radius = ["within DIST", "in a radius of DIST", "no more than DIST from another",
                               "DIST from each other"]
        self.phrases_contains = ["within", "in", "inside", "contained in"]

        self.dist_lookup = {"cm": "centimeters", "m": "meters", "km": "kilometers", "in": "inches", "ft": "feet",
                            "yd": "yards", "mi": "miles"}

    def beginning(self, persona, writing_style):
        '''
        Create a beginning of a prompt by using beginning template
        '''
        return self.beginning_template.format(persona=persona, style=writing_style)

    def search_query(self, beginning_prompt: str):
        '''
        Append the beginning prompt with search phrase. The search phrase is randomly chosen among the search templates. If search templates contain {place}, it randomly selects a place from predefined_places
        '''
        search_template = np.random.choice(self.search_templates)

        if '{place}' in search_template:
            np.random.shuffle(self.predefined_places)
            selected_place = self.predefined_places[0]
            beginning_prompt += search_template.replace('{place}', selected_place)
        else:
            beginning_prompt += search_template

        return beginning_prompt

    def add_area_prompt(self, area: Area) -> str:
        '''
        Helper to generate area prompt that is appended to search_prompt
        '''
        area_prompt = ""
        if area.type not in ["bbox", "polygon"]:
            area_prompt = "Search area: " + area.value + "\n"
        return area_prompt

    def add_numerical_prompt(self, entity_property: Property) -> str:
        '''
        This helper generates a numerical prompt for numerical properties and properties such as height
        '''
        if not is_number(entity_property.value) and np.random.choice([True, False]):
            metric = self.dist_lookup[entity_property.value.rsplit(" ", 1)[-1]]
            value = entity_property.value.rsplit(" ", 1)[0] + " " + metric
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
            core_prompt = core_prompt + ", "
            core_prompt = core_prompt + entity_property.name

            if entity_property.name == 'height' or is_number(entity_property.value):
                core_prompt = core_prompt + self.add_numerical_prompt(entity_property=entity_property)
            elif entity_property.operator == '~':
                core_prompt = core_prompt + self.add_name_regex_prompt(entity_property=entity_property)
            else:
                core_prompt = core_prompt + self.add_other_non_numerical_prompt(entity_property=entity_property)
        return core_prompt

    def add_relative_spatial_terms(self, relation: Relation) -> tuple:
        '''
        Randomly selects relative spatial term
        '''
        selected_relative_spatial = np.random.choice(self.relative_spatial_terms)

        # select randomly descriptor of relative special
        descriptors_of_relative_spatial_terms = selected_relative_spatial.values
        np.random.shuffle(descriptors_of_relative_spatial_terms)
        selected_relative_spatial_term = descriptors_of_relative_spatial_terms[0]
        return (selected_relative_spatial_term, selected_relative_spatial.distance)

    # def add_relative_spatial_term_helper(self, selected_relative_spatial_term: str, relation: Relation,
    #                                      selected_relative_spatial: RelSpatial):
    #     generated_prompt = f"Use this term to describe the spatial relation between Obj. {relation.source} and {relation.target} similar to (similar to \"X is _ Y\"): {selected_relative_spatial_term}\n"
    #     overwritten_distance = selected_relative_spatial.distance
    #     return generated_prompt, overwritten_distance

    def generate_written_word_distance(self, metric: str, max_digits: int) -> tuple:
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

        numeric = randint(low, high) * 100
        written = num2words(numeric) + " " + metric
        numeric = str(numeric) + " " + metric

        return numeric, written

    def add_desc_away_prompt_helper(self, relation: Relation, selected_phrases_desc: str, selected_phrases_away: str):
        '''Helper function for generating desc away prompts'''
        if np.random.choice([True, False]):
            metric = self.dist_lookup[relation.value.rsplit(" ", 1)[-1]]
            distance = relation.value.rsplit(" ", 1)[0] + " " + metric
        else:
            distance = relation.value

        generated_prompt = f"Obj. {relation.source} is{selected_phrases_desc} {distance} {selected_phrases_away} Obj. {relation.target}\n"
        return generated_prompt

    def add_desc_away_prompt(self, relation: Relation) -> str:
        selected_phrases_desc = np.random.choice(self.phrases_desc)
        selected_phrases_away = np.random.choice(self.phrases_away)

        generated_prompt = self.add_desc_away_prompt_helper(relation, selected_phrases_desc, selected_phrases_away)
        return generated_prompt

    def add_prompt_for_within_radius_relation(self, distance: str) -> str:
        if np.random.choice([True, False]):
            metric = self.dist_lookup[distance.rsplit(" ", 1)[-1]]
            distance = distance.rsplit(" ", 1)[0] + " " + metric

        selected_phrase = np.random.choice(self.phrases_radius)
        selected_phrase = selected_phrase.replace('DIST', distance)
        generated_prompt = f"All objects are {selected_phrase}"
        return generated_prompt

    def add_relation_with_contain(self, relations: List[Relation]) -> [Relations]:
        '''
        This function identifies the objects having containing relationship, collect the remaining ones which have individual rels with the other ones.
        :param relations:
        :return: generated_prompt, List[Relation]: list of individual relations
        '''
        individual_rels = []
        for relation in relations:
            if relation.type != "contains":
                individual_rels.append(relation)

        # todo: question here we change it to individual distances but we don't compare it with individual distance
        return Relations(type='individual_distance', relations=individual_rels)

    def add_optional_prases(self, data):
        updated_data = copy.deepcopy(data)
        for entity in updated_data["entities"]:
            for property in entity["properties"]:
                if property["operator"] == ">":
                    property["operator"] = np.random.choice(self.phrases_for_numerical_comparison[">"]) + ""
                elif property["operator"] == "<":
                    property["operator"] = np.random.choice(self.phrases_for_numerical_comparison["<"]) + ""
                elif property["operator"] == "~":
                    property["operator"] = np.random.choice(self.name_regex_templates) + ""

        if updated_data["relations"]["relations"]:
            for relation in updated_data["relations"]["relations"]:
                if relation["value"]:
                    relation["value"] = np.random.choice(self.phrases_desc) + " " + relation["value"]

        return updated_data


class GPTDataGenerator:
    def __init__(self, system_prompt: str, relative_spatial_terms: List[RelSpatial], personas: List[str],
                 styles: List[str], prob_usage_of_relative_spatial_terms: float = 0.4,
                 prob_usage_of_written_numbers: float = 0.3, prob_of_typos: float=0.3, max_dist_digits: int = 5):

        self.relative_spatial_terms = relative_spatial_terms
        self.prob_usage_of_relative_spatial_terms = prob_usage_of_relative_spatial_terms
        self.prob_usage_of_written_numbers = prob_usage_of_written_numbers
        self.prob_of_typos = prob_of_typos
        self.max_dist_digits = max_dist_digits
        self.system_prompt = system_prompt
        self.personas = personas
        self.styles = styles
        self.prompt_helper = PromptHelper(relative_spatial_terms=relative_spatial_terms)

    def update_relation_distance(self, relations: Relations, relation_to_be_updated: Relation, distance: str):
        updated_relations = []
        for relation in relations.relations:
            if relation == relation_to_be_updated:
                relation.value = distance
                updated_relations.append(relation)
            else:
                updated_relations.append(relation)
        return relations.update(relations=updated_relations)

    def generate_prompt(self, loc_point: LocPoint, persona: str, style: str) -> str:
        '''
        A method that takes the intermediate query representation, and uses it to generate a natural language prompt for
        the GPT API. Different sentence structures are required for the different tasks, for the special tag "count",
        as well as for the different substring searches (beginning, ending, containing, equals).

        :param dict loc_point: The dictionary containing all relevant information for the query
        '''
        relations = loc_point.relations

        overwritten_relations = None
        overwritten_relations_for_prompt = None
        if relations.type in ["individual_distances_with_contains", "contains_relation"]:
            individual_rels = self.prompt_helper.add_relation_with_contain(relations.relations)
        else:
            individual_rels = relations

        # todo: we don't compare it with individual_rels but relations, why?
        if relations.type in ["individual_distances", "individual_distances_with_contains"]:
            overwritten_relations, overwritten_relations_for_prompt = self.individual_prompt_generation(individual_rels)
        elif relations.type == "within_radius":
            overwritten_relations, overwritten_relations_for_prompt = self.radius_prompt_generation(individual_rels)

        if overwritten_relations_for_prompt:
            if overwritten_relations:
                # update the relations
                loc_point_for_yaml = copy.deepcopy(loc_point)
                loc_point_for_yaml.update_relations(overwritten_relations_for_prompt)
                loc_point.update_relations(overwritten_relations)

                data = loc_point_for_yaml.dict()
        else:
            data = loc_point.dict()

        data = self.prompt_helper.add_optional_prases(data)

        data = clean_up_query(data)

        prompt_yaml_part = f"===Input===\n```yaml\n{yaml.dump(data)}```"
        persona_prompt = f"===Persona===\n{persona}"
        style_prompt = f"===Style===\n{style}"
        sentence_prompt = f"===Sentence===\n"

        core_prompt = f"{prompt_yaml_part}\n\n{persona_prompt}\n\n{style_prompt}\n\n{sentence_prompt}"

        return loc_point, core_prompt

    def assign_persona_styles_to_queries(self, num_of_all_persona_style, num_tag_queries):
        persona_style_ids = list(range(num_of_all_persona_style))
        num_tag_queries_ids = list(range(num_tag_queries))

        cycled_persona_style_ids = itertools.cycle(persona_style_ids)
        persona_style_tag_pairs = [(x, next(cycled_persona_style_ids)) for x in num_tag_queries_ids]
        return persona_style_tag_pairs

    def individual_prompt_generation(self, relations: Relations) -> Relations:
        updated_relations = copy.deepcopy(relations)
        for relation in relations.relations:
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
                spatial_term, overwritten_distance = self.prompt_helper.add_relative_spatial_terms(relation)
                self.update_relation_distance(relations=updated_relations,
                                              relation_to_be_updated=relation,
                                              distance=overwritten_distance)

                self.update_relation_distance(relations=relations,
                                              relation_to_be_updated=relation,
                                              distance=spatial_term)
            elif use_written_distance:
                metric = relation.value.split()[-1]
                numeric_distance, written_distance = self.prompt_helper.generate_written_word_distance(
                    metric, self.max_dist_digits)
                self.update_relation_distance(relations=relations,
                                              relation_to_be_updated=relation,
                                              distance=written_distance)

        return updated_relations, relations

    def radius_prompt_generation(self, relations: Relations) -> Relations:
        updated_relations = copy.deepcopy(relations)
        metric = relations.relations[0].value.split()[-1]
        use_written_distance = np.random.choice([False, True], p=[
            1.0 - self.prob_usage_of_written_numbers, self.prob_usage_of_written_numbers])
        if use_written_distance:
            numeric_distance, written_distance = self.prompt_helper.generate_written_word_distance(
                metric, self.max_dist_digits)
            for relation in relations.relations:
                self.update_relation_distance(relations=relations,
                                              relation_to_be_updated=relation,
                                              distance=numeric_distance)
        return updated_relations, relations

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
        generated_sentence = request_openai(system_prompt=self.system_prompt, user_prompt=generated_prompt["prompt"])
        return generated_sentence


if __name__ == '__main__':
    '''
    Define paths and run all desired functions.
    '''
    parser = ArgumentParser()
    parser.add_argument('--relative_spatial_terms_path', help='Path for the relative spats', required=True)
    parser.add_argument('--system_prompt', help='Path for the system prompt', required=True)
    parser.add_argument('--tag_query_file', required=True)
    parser.add_argument('--output_gpt_generations', required=True)
    parser.add_argument('--output_prompt_generations', required=True)
    parser.add_argument('--persona_path', required=True)
    parser.add_argument('--styles_path', required=True)
    parser.add_argument('--prob_usage_of_relative_spatial_terms', type=float, default=0.4)
    parser.add_argument('--prob_usage_of_written_numbers', type=float, default=0.3)
    parser.add_argument('--prob_of_typos', type=float, default=0.3)
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
    system_prompt = args.system_prompt
    persona_path = args.persona_path
    styles_path = args.styles_path
    tag_query_file = args.tag_query_file
    prob_usage_of_relative_spatial_terms = args.prob_usage_of_relative_spatial_terms
    prob_usage_of_written_numbers = args.prob_usage_of_written_numbers
    max_dist_digits = args.max_dist_digits
    generate_sentences = args.generate_sentences
    generate_prompts = args.generate_prompts
    translate_to_yaml = args.translate_to_yaml
    save_yaml_csv = args.save_yaml_csv

    system_prompt = load_list_of_strings(list_of_strings_path=system_prompt)
    rel_spatial_terms = load_rel_spatial_terms(relative_spatial_terms_path=relative_spatial_terms_path)
    personas = load_list_of_strings(list_of_strings_path=persona_path)
    styles = load_list_of_strings(list_of_strings_path=styles_path)

    gen = GPTDataGenerator(
        system_prompt=system_prompt,
        relative_spatial_terms=rel_spatial_terms,
        personas=personas,
        styles=styles,
        prob_usage_of_relative_spatial_terms=prob_usage_of_relative_spatial_terms,
        prob_usage_of_written_numbers=prob_usage_of_written_numbers,
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
