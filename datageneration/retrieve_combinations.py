import pandas as pd
import requests
import taginfo.query as ti
import unicodedata
from argparse import ArgumentParser
from diskcache import Cache
from tqdm import tqdm
from typing import List, Dict, Any

from datageneration.data_model import Tag, TagProperty, TagCombination, TagPropertyExample, \
    remove_duplicate_tag_properties
from datageneration.utils import CompoundTagPropertyProcessor, SEPERATORS, write_output, split_descriptors

"""
Tag combination and property example retrieval for OSM-based data generation.

This module:
- Splits and normalizes compound OSM tags from a CSV/XLSX "Primary Key" table.
- Fetches commonly co-occurring tag properties via the Taginfo API.
- Optionally generates example values (e.g., cuisines, colours) for selected properties.
- Writes out structured `TagCombination` and `TagPropertyExample` objects for downstream use.

Key types (from datageneration.data_model):
- Tag, TagProperty, TagCombination, TagPropertyExample

CLI usage (examples):
    python combination_retriever.py \
        --source data/primary_keys.csv \
        --output_file out/tag_combinations.jsonl \
        --generate_tag_list_with_properties \
        --prop_limit 100 --min_together_count 5000

    python combination_retriever.py \
        --source data/primary_keys.csv \
        --output_file out/property_examples.jsonl \
        --generate_property_examples --prop_example_limit 100000
"""

cache = Cache("tmp")

TAG_INFO_API_ENDPOINT = "https://taginfo.openstreetmap.org/api/4/tag/combinations?key=TAG_KEY&value=TAG_VALUE&sortname=together_count&sortorder=desc"


@cache.memoize()
def request_tag_combinations(tag_key: str, tag_value: str) -> Dict[str, Any]:
    """
    Tag combination and property example retrieval for OSM-based data generation.

    This module:
    - Splits and normalizes compound OSM tags from a CSV/XLSX "Primary Key" table.
    - Fetches commonly co-occurring tag properties via the Taginfo API.
    - Optionally generates example values (e.g., cuisines, colours) for selected properties.
    - Writes out structured `TagCombination` and `TagPropertyExample` objects for downstream use.

    Key types (from datageneration.data_model):
    - Tag, TagProperty, TagCombination, TagPropertyExample

    CLI usage (examples):
        python combination_retriever.py \
            --source data/primary_keys.csv \
            --output_file out/tag_combinations.jsonl \
            --generate_tag_list_with_properties \
            --prop_limit 100 --min_together_count 5000

        python combination_retriever.py \
            --source data/primary_keys.csv \
            --output_file out/property_examples.jsonl \
            --generate_property_examples --prop_example_limit 100000
    """
    url = TAG_INFO_API_ENDPOINT.replace("TAG_KEY", tag_key).replace("TAG_VALUE", tag_value)

    response = requests.get(url)
    response.raise_for_status()

    if response.status_code == 200:
        return response.json()


def is_similar_to_english(char: str) -> bool:
    """
    Check whether a character is ASCII English or a close diacritic variant.

    Parameters
    ----------
    char : str
        Single character to check.

    Returns
    -------
    bool
        True if character is in the English alphabet or a normalized variant; False otherwise.

    Notes
    -----
    - Uses NFKD normalization to strip diacritics before comparison.
    """
    english_alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    similar_chars = english_alphabet + 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ'

    if char in similar_chars:
        return True

    normalized_char = unicodedata.normalize('NFKD', char)
    stripped_char = ''.join([c for c in normalized_char if not unicodedata.combining(c)])

    return stripped_char in similar_chars


def is_roman(s: str) -> bool:
    """
    Check whether a string contains only (extended) Roman/Latin letters.

    Parameters
    ----------
    s : str
        Input string.

    Returns
    -------
    bool
        True if all characters are ASCII or pass `is_similar_to_english`; False otherwise.
    """
    for char in s:
        if ord(char) > 127 and not is_similar_to_english(char):
            return False
    return True


comp_prop_processor = CompoundTagPropertyProcessor()

def split_tags(tags: str) -> List[str]:
    """
    Split and normalize a comma-separated list of OSM tag patterns.

    Handles:
    - 'AND' compounds (e.g., "highway=primary AND surface=asphalt")
    - bracketed compounds processed by `CompoundTagPropertyProcessor`
    - whitespace removal and lowercase normalization

    Parameters
    ----------
    tags : str
        Comma-separated tag expression(s).

    Returns
    -------
    List[str]
        A list of normalized tag patterns like "key=val", "key~***example***", etc.
    """
    processed_tags = set()
    for tag in tags.split(','):
        tag = tag.lstrip().strip()
        if 'AND' in tag:
            _tags = tag.split('AND')
            for _tag in _tags:
                _tag = _tag.lstrip().strip().replace(' ', '').lower()
                processed_tags.add(_tag)
        else:
            tag = tag.replace(' ', '').lower()
            if len(tag) == 0:
                continue

            if '[' in tag:
                compound_tag = comp_prop_processor.run(tag)
                for alt_tag in compound_tag:
                    processed_tags.add(alt_tag)
            else:
                processed_tags.add(tag)
    return list(processed_tags)


class CombinationRetriever(object):
    """
    Build `TagCombination` and `TagPropertyExample` datasets from a Primary Key table.

    Parameters
    ----------
    source : str
        Path to CSV/XLSX file containing the Primary Key table (columns include 'tags',
        'core/prop', 'descriptors', 'area/point', etc.).
    prop_limit : int
        Max number of related properties to fetch per (key, value) via Taginfo.
    min_together_count : int
        Minimum together_count threshold for Taginfo combinations to be considered.
    add_non_roman_examples : bool
        If True, property examples may include non-Roman strings; otherwise filtered.

    Attributes
    ----------
    tag_properties : List[TagProperty]
        Parsed TagProperty templates from the Primary Key table (where 'core/prop' != 'core').
    prop_limit : int
        Stored property fetch limit.
    min_together_count : int
        Stored Taginfo together_count threshold.
    tag_df : pd.DataFrame
        Loaded and de-duplicated Primary Key dataframe.
    all_osm_tags_and_properties : Dict[str, dict]
        Mapping of normalized tag pattern to metadata (key/operator/value/type/descriptors).
    all_tags_property_ids : iterable
        Keys of `all_osm_tags_and_properties`.
    numeric_tags_property_ids : List[str]
        A subset of keys (ending with '>0') used in numeric setups.
    tags_requiring_many_examples : List[str]
        Whitelist of patterns for which a large number of examples should be retrieved.
    add_non_roman_examples : bool
        Example filtering behavior.
    """
    def __init__(self, source: str, prop_limit: int, min_together_count: int, add_non_roman_examples: bool):
        if source.endswith('xlsx'):
            tag_df = pd.read_excel(source, engine='openpyxl')
        else:
            tag_df = pd.read_csv(source, index_col=False)

        tag_df.drop_duplicates(subset='descriptors', inplace=True)
        tag_df["index"] = [i for i in range(len(tag_df))]
        all_osm_tags_and_properties = self.process_tag_properties(tag_df)

        self.tag_properties = self.fetch_tag_properties(tag_df)
        self.prop_limit = prop_limit
        self.min_together_count = min_together_count
        self.tag_df = tag_df
        self.all_osm_tags_and_properties = all_osm_tags_and_properties

        self.all_tags_property_ids = self.all_osm_tags_and_properties.keys()
        self.numeric_tags_property_ids = [f.split(">")[0] for f in filter(lambda x: x.endswith(">0"),
                                                                          self.all_tags_property_ids)]
        self.tags_requiring_many_examples = ["name~***example***", "brand~***example***", "addr:street~***example***",
                                             "addr:housenumber=***example***"]


        self.add_non_roman_examples = add_non_roman_examples

    def fetch_tag_properties(self, tag_df: pd.DataFrame) -> List[TagProperty]:
        """
        Build `TagProperty` objects from the Primary Key dataframe.

        Parameters
        ----------
        tag_df : pd.DataFrame
            The Primary Key dataframe.

        Returns
        -------
        List[TagProperty]
            Parsed properties (rows where 'core/prop' != 'core').
        """
        tag_property_df = tag_df[tag_df['core/prop'] != 'core']
        tag_properties = []
        for tag_prop in tag_property_df.to_dict(orient='records'):
            if isinstance(tag_prop['descriptors'], float):
                continue
            descriptors = split_descriptors(tag_prop['descriptors'])
            splited_tags = split_tags(tag_prop['tags'])
            processed_tags = []

            for _tag in splited_tags:
                _tag_splits = None
                tag_operator = None

                for seperator in SEPERATORS:
                    if seperator in _tag:
                        _tag_splits = _tag.split(seperator)
                        tag_operator = seperator
                        continue
                processed_tags.append(Tag(key=_tag_splits[0], value=_tag_splits[1], operator=tag_operator))

            tag_properties.append(TagProperty(descriptors=descriptors, tags=processed_tags))
        return tag_properties

    def process_tag_properties(self, tag_df):
        """
        Parse all tags/properties from the Primary Key dataframe into a normalized mapping.

        Parameters
        ----------
        tag_df : pd.DataFrame
            Input dataframe with at least 'tags', 'core/prop', 'descriptors' columns.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Mapping from normalized tag pattern (e.g., 'amenity=restaurant') to metadata
            (key/operator/value, type, descriptors, original tags string).
        """
        # all tags and properties
        all_osm_tags_and_properties = {}
        for tags in tag_df.to_dict(orient='records'):
            tag_type = tags['core/prop']
            if isinstance(tag_type, float):
                print(f'{tags} has no type, might be an invalid')
                continue
            tags_list = tags['tags']
            descriptors = tags['descriptors']
            tag_type = tag_type.strip()
            splited_tags = split_tags(tags['tags'])
            for _tag in splited_tags:
                _tag_splits = None
                tag_operator = None
                for seperator in SEPERATORS:
                    if seperator in _tag:
                        _tag_splits = _tag.split(seperator)
                        tag_operator = seperator
                        continue

                if _tag in all_osm_tags_and_properties:
                    if all_osm_tags_and_properties[_tag]["core/prop"] != tag_type:
                        all_osm_tags_and_properties[_tag] = {'tags': tags_list, 'key': _tag_splits[0],
                                                             'operator': tag_operator,
                                                             'value': _tag_splits[1],
                                                             'core/prop': "core/prop", 'descriptors': descriptors}
                else:
                    all_osm_tags_and_properties[_tag] = {'tags': tags_list, 'key': _tag_splits[0],
                                                         'operator': tag_operator, 'value': _tag_splits[1],
                                                         'core/prop': tag_type, 'descriptors': descriptors}

        return all_osm_tags_and_properties

    def request_property_examples(self, property_key: str, num_examples: int, count_limit: int = -1) -> List[str]:
        """
        Retrieve example values for a non-numeric property (e.g., cuisine → 'italian', 'turkish').

        Fetches pages from Taginfo (key/value counts) until the requested number of examples is met
        or pages are exhausted. Splits multi-values by ';'. Applies roman/non-roman filtering.

        Parameters
        ----------
        property_key : str
            OSM key to fetch examples for (e.g., 'cuisine', 'brand', 'name', 'roof:colour').
        num_examples : int
            Maximum number of examples to collect.
        count_limit : int, optional
            Minimum Taginfo count required for an example to be included (default: -1 = no threshold).

        Returns
        -------
        List[str]
            Collected, de-duplicated example values.
        """
        def fetch_examples_recursively(curr_page, fetched_examples):
            examples = ti.get_page_of_key_values(property_key, curr_page)
            if len(examples) == 0:
                return fetched_examples
            for example in examples:
                example_value = example['value']
                if count_limit !=-1:
                    example_count = example['count']
                    if example_count < count_limit:
                        continue

                for _example in example_value.split(';'):
                    if len(fetched_examples) > num_examples - 1:
                        return fetched_examples

                    if not self.add_non_roman_examples:
                        if is_roman(_example):
                            fetched_examples.add(_example)
                    else:
                        fetched_examples.add(_example)
            # Fetch next page recursively
            return fetch_examples_recursively(curr_page + 1, fetched_examples)

        fetched_examples = set()
        fetched_examples = fetch_examples_recursively(1, fetched_examples)
        return list(fetched_examples)

    def generate_property_examples(self, num_examples: int = 100000) -> List[TagPropertyExample]:
        """
        Generate example values for properties that require instances (***example***).

        Parameters
        ----------
        num_examples : int, optional
            Upper bound for examples to gather for properties that require many examples; defaults to 100000.

        Returns
        -------
        List[TagPropertyExample]
            Objects containing the tag pattern key (e.g., "name~***example***") and its example values.

        Notes
        -----
        - For colour-related keys, a high `count_limit` is applied to prioritize frequent values.
        - Only properties with type != 'core' and patterns containing '***example***' are processed.
        """
        properties_and_their_examples = []
        for curr_tag, all_tags in self.all_osm_tags_and_properties.items():
            if curr_tag not in self.tags_requiring_many_examples:
                curr_num_examples = 100
            else:
                curr_num_examples = num_examples

            if all_tags['core/prop'] != 'core' and '***example***' in curr_tag:
                if all_tags['key'] in ['roof:colour', 'building:colour', 'colour']:
                    examples = self.request_property_examples(all_tags['key'], num_examples=curr_num_examples, count_limit=10000)

                else:
                    examples = self.request_property_examples(all_tags['key'], num_examples=curr_num_examples)
                properties_and_their_examples.append(
                    TagPropertyExample(key=curr_tag, examples=examples))
        return properties_and_their_examples

    def check_other_tag_in_properties(self, other_tag: str) -> tuple:
        """
        Check whether a candidate 'other_tag' exists in the parsed TagProperty list.

        Matches both 'key<op>value' and 'key<op>' forms (treating '***any***', '***example***',
        '***numeric***', and 'yes' as empty values).

        Parameters
        ----------
        other_tag : str
            Candidate tag pattern (e.g., 'name=', 'name~', 'surface=asphalt').

        Returns
        -------
        tuple[bool, list[int] | int]
            (True, [indices...]) if found; otherwise (False, -1).
        """
        exists = False
        results = []
        for tag_prop_idx, tag_prop in enumerate(self.tag_properties):
            for tag_prop_tag in tag_prop.tags:
                tag_prop_tag_value = tag_prop_tag.value
                if tag_prop_tag_value in ['***any***', '***example***', 'yes', '***numeric***']:
                    tag_prop_tag_value = ''
                if f'{tag_prop_tag.key}{tag_prop_tag.operator}{tag_prop_tag_value}' == other_tag:
                    exists = True
                    results.append(tag_prop_idx)
                    # return (exists, tag_prop_idx)
                elif f'{tag_prop_tag.key}{tag_prop_tag.operator}' == other_tag:
                    exists = True
                    results.append(tag_prop_idx)
                    # return (exists, tag_prop_idx)
        if exists:
            return (exists, results)
        else:
            return (exists, -1)

    def request_related_tag_properties(self, tag_key: str, tag_value: str, limit: int = 100) -> List[TagProperty]:
        """
        Query Taginfo for (key,value) co-occurrence and return matching TagProperty templates.

        Parameters
        ----------
        tag_key : str
            The focal OSM key (e.g., 'amenity').
        tag_value : str
            The focal OSM value (e.g., 'clinic').
        limit : int, optional
            Maximum number of TagProperty items to return, by descending together_count.

        Returns
        -------
        List[TagProperty]
            TagProperty objects present in `self.tag_properties` whose (other_key, other_value)
            appear with the given (tag_key, tag_value) and pass the together_count threshold.
        """
        combinations = request_tag_combinations(tag_key=tag_key, tag_value=tag_value)['data']
        selected_properties = []
        for combination in combinations:
            if len(selected_properties) == limit or combination["together_count"] < self.min_together_count:
                return list(selected_properties)

            for seperator in SEPERATORS:
                exist_property, prop_indices = self.check_other_tag_in_properties(
                    other_tag=combination['other_key'] + seperator + combination['other_value'])
                if exist_property:
                    break

            if exist_property:
                for prop_index in prop_indices:
                    fetched_tag_prop = self.tag_properties[prop_index]
                    selected_properties.append(fetched_tag_prop)
            # else:
            #     print(f'{combination} does not exist')
            #     if (combination['other_key'] in self.numeric_tags_properties_ids and
            #             combination['other_value'].isnumeric()):
            #         if int(combination['other_value']) > 0:
            #             rewritten_tag = combination['other_key'] + ">0"
            #
            #             print("rewritten tag")
            #             print(rewritten_tag)
        return selected_properties

    def generate_tag_list_with_properties(self) -> List[TagCombination]:
        """
        Generate `TagCombination` objects (per row of Primary Key table) with related properties.

        Workflow
        --------
        - Parse row into cluster_id, descriptors, comb_type, tags.
        - For each (key, value) tag, fetch related TagProperty templates (unless comb_type == 'prop').
        - Deduplicate properties and pack into a TagCombination.

        Returns
        -------
        List[TagCombination]
            One `TagCombination` per input row with associated properties.
        """
        tag_combinations = []

        for row in tqdm(self.tag_df.to_dict(orient='records'), total=len(self.tag_df)):
            cluster_id = row['index']
            is_area = True if row['area/point'] == 'area' else False
            descriptors = split_descriptors(row['descriptors'])
            print(descriptors)
            print(row)
            comb_type = row['core/prop'].strip()
            tags = split_tags(row['tags'])

            processed_tags = []
            processed_properties = []
            for tag in tags:
                for sep in SEPERATORS:
                    if sep in tag:
                        tag_key, tag_value = tag.split(sep)
                        processed_tags.append(Tag(key=tag_key, operator=sep, value=tag_value))

                        if comb_type != 'prop':
                            tag_properties = self.request_related_tag_properties(tag_key=tag_key,
                                                                                 tag_value=tag_value,
                                                                                 limit=self.prop_limit)
                            processed_properties.extend(tag_properties)

            processed_properties = remove_duplicate_tag_properties(processed_properties)
            tag_combinations.append(
                TagCombination(cluster_id=cluster_id, is_area=is_area, descriptors=descriptors, comb_type=comb_type,
                               tags=processed_tags, tag_properties=processed_properties))
        return tag_combinations


if __name__ == '__main__':
    """
    CLI entry point.

    Flags:
        --generate_tag_list_with_properties : Build TagCombination objects and write to output_file.
        --generate_property_examples        : Build TagPropertyExample objects and write to output_file.

    Notes:
        - You can run both flags in a single invocation; each writes to the same output path in sequence.
        - `--add_non_roman_examples` controls filtering of example values (default True).
    """
    parser = ArgumentParser()
    parser.add_argument('--source', help='domain-specific primary keys', required=True)
    parser.add_argument('--output_file', help='Path to save the tag list', required=True)
    parser.add_argument('--prop_limit', help='Enter the number of related tags to be fetched by taginfo', default=100)
    parser.add_argument('--min_together_count', help='The min together count for a combination to be considered',
                        default=5000, type=int)
    parser.add_argument('--prop_example_limit', help='Enter the number of example values of the properties',
                        default=100000, type=int)
    parser.add_argument('--generate_tag_list_with_properties', help='Generate tag list with properties',
                        action='store_true')
    parser.add_argument('--generate_property_examples', help='Generate property examples',
                        action='store_true')
    parser.add_argument('--add_non_roman_examples', action='store_true', default=True)

    args = parser.parse_args()

    source = args.source
    prop_limit = int(args.prop_limit)
    min_together_count = args.min_together_count
    prop_example_limit = args.prop_example_limit
    add_non_roman_examples = args.add_non_roman_examples
    output_file = args.output_file
    generate_tag_list_with_properties = args.generate_tag_list_with_properties
    generate_property_examples = args.generate_property_examples

    comb_retriever = CombinationRetriever(source=source, prop_limit=prop_limit, min_together_count=min_together_count, add_non_roman_examples=add_non_roman_examples)

    if generate_tag_list_with_properties:
        tag_combinations = comb_retriever.generate_tag_list_with_properties()
        write_output(generated_combs=tag_combinations, output_file=output_file)

    if generate_property_examples:
        prop_examples = comb_retriever.generate_property_examples(num_examples=prop_example_limit)
        write_output(generated_combs=prop_examples, output_file=output_file)
