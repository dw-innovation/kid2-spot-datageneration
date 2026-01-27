import os
import csv
import itertools
import json
import numpy as np
import yaml
from pathlib import Path
from random import randint
from typing import List, Set

from datageneration.data_model import Distance

"""
General utilities for data generation and serialization.

Includes:
- Random number helpers for distances/integers.
- File output helpers (JSONL, CSV) with optional YAML-ified filename variants.
- Query cleanup/normalization for YAML export.
- Descriptor/tag parsing helpers, including expansion of compound tag patterns.

Notes
-----
- `SEPERATORS` intentionally uses the project’s original spelling.
- `split_descriptors` returns a **set** of lowercase descriptors to deduplicate.
"""

SEPERATORS = ['=', '>', '~']

NON_ROMAN_LANGUAGES = ['ru', 'ba', 'be', 'bg', 'ce', 'cv', 'kk', 'ky', 'mdf', 'mhr', 'mk', 'mrj', 'myv', 'os', 'sr', 'tg', 'tt', 'udm', 'ar', 'arz', 'azb', 'ckb', 'fa', 'pnb', 'ps', 'skr', 'ur', 'he', 'ye', 'ka', 'xmf', 'zh', 'zh_yue', 'wu', 'el', 'ja', 'ta', 'th', 'ur']

NON_ROMAN_LANG_GROUPS = {
    'cyrillic': ['ru', 'be', 'bg', 'ce', 'cv', 'kk', 'ky', 'mdf', 'mhr', 'mk', 'mrj', 'myv', 'os', 'sr', 'tg', 'tt', 'udm'],
    'arabic': ['ar', 'arz', 'azb', 'ckb', 'fa', 'pnb', 'ps', 'skr', 'ur'],
    'hebrew': ['he', 'ye'],
    'georgian': ['ka', 'xmf'],
    'chinese': ['zh', 'zh_yue', 'wu'],
    'greek': ['el'],
    'japanese': ['ja'],
    'korean': ['ko'],
    'tamil': ['ta'],
    'thai': ['th']
}


def get_random_decimal_with_metric(max_digits: int) -> Distance:
    """
    Generate a random distance with magnitude and metric.

    The magnitude is an integer with up to `max_digits` digits; it may be converted to a
    simple decimal by dividing by 10 or 100. The metric is sampled from a small unit list.

    Parameters
    ----------
    max_digits : int
        Maximum number of digits for the integer magnitude before optional division.

    Returns
    -------
    Distance
        A `Distance` object with `magnitude` (as a string) and `metric` (unit).
    """
    digits = randint(1, max_digits)
    low = np.power(10, digits - 1)
    high = np.power(10, digits) - 1
    magnitude = randint(low, high)

    if np.random.choice([True, False], 1)[0]:
        magnitude = magnitude / np.random.choice([10, 100], 1)[0]

    return Distance(magnitude=str(magnitude), metric=np.random.choice(["cm", "m", "km", "in", "ft", "yd", "mi"], 1)[0])


def get_random_integer(max_digits: int) -> int:
    """
    Generate a random positive integer with up to `max_digits` digits.

    Parameters
    ----------
    max_digits : int
        Maximum number of digits the generated integer can have.

    Returns
    -------
    int
        Random integer in [10^(d-1), 10^d - 1] where d ∈ [1, max_digits].
    """
    digits = randint(1, max_digits)
    low = np.power(10, digits - 1)
    high = np.power(10, digits) - 1

    return randint(low, high)


def add_yaml_to_filename(output_file: str) -> Path:
    """
    Produce a sibling filename with `_yaml` inserted before the original extension.

    Example
    -------
    "out/results.jsonl" -> "out/results_yaml.jsonl"

    Parameters
    ----------
    output_file : str
        Original output file path.

    Returns
    -------
    pathlib.Path
        New path with `_yaml` suffix applied to the stem.
    """
    parent_dir = Path(output_file).parent
    filename_without_extension = Path(output_file).stem
    file_extension = Path(output_file).suffix
    yaml_output_file = parent_dir / (filename_without_extension + "_yaml" + file_extension)
    return yaml_output_file


def write_output(generated_combs, output_file: str) -> None:
    """
    Write an iterable of Pydantic-like objects to JSON Lines, one per line.

    Each item is serialized via `.model_dump(mode="json")`.

    Parameters
    ----------
    generated_combs : Iterable
        Items with `.model_dump(mode="json")` method.
    output_file : str
        Target JSONL file path. Parent directories are created if needed.
    """
    dir_path = os.path.dirname(output_file)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_file:
            for generated_comb in generated_combs:
                json.dump(generated_comb.model_dump(mode="json"), out_file, ensure_ascii=False)
                out_file.write('\n')


def write_dict_output(generated_combs, output_file: str, bool_add_yaml: bool = True) -> None:
    """
    Write an iterable of dictionaries to JSON Lines, with optional `_yaml` filename variant.

    Parameters
    ----------
    generated_combs : Iterable[dict]
        Plain dicts ready for JSON serialization.
    output_file : str
        Target JSONL file path (may be rewritten with `_yaml` if `bool_add_yaml=True`).
    bool_add_yaml : bool, optional
        If True, modifies the filename via `add_yaml_to_filename`. Default True.
    """
    if bool_add_yaml:
        output_file = add_yaml_to_filename(output_file)

    dir_path = os.path.dirname(output_file)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as out_file:
            for generated_comb in generated_combs:
                print(generated_comb)
                json.dump(generated_comb, out_file, ensure_ascii=False)
                out_file.write('\n')


def write_output_csv(generated_combs, output_file: str, bool_add_yaml: bool = True) -> None:
    """
    Write a list of homogeneous dicts to CSV, using keys from the first element.

    The filename is rewritten to `.csv`. If `bool_add_yaml` is True, `_yaml` is inserted
    before the extension first, then `.csv` is applied.

    Parameters
    ----------
    generated_combs : List[dict]
        Sequence of dictionaries with identical keys.
    output_file : str
        Base output path; final file will be `<parent>/<stem>.csv` (with optional `_yaml`).
    bool_add_yaml : bool, optional
        Whether to add `_yaml` before changing extension to `.csv`. Default True.
    """
    if bool_add_yaml:
        output_file = add_yaml_to_filename(output_file)

    parent_dir = Path(output_file).parent
    filename_without_extension = Path(output_file).stem
    new_output_file = parent_dir / (filename_without_extension + ".csv")

    keys = generated_combs[0].keys()
    with open(new_output_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(generated_combs)


def translate_queries_to_yaml(combs) -> dict:
    """
    Convert each combination's `query` object into a YAML string, in-place.

    Steps:
    - Clean/normalize the `query` structure (`clean_up_query`).
    - Dump to YAML (unicode-safe).
    - Replace the original `query` object with its YAML string.

    Parameters
    ----------
    combs : Iterable
        Iterable of objects with a `.dict()` method (e.g., Pydantic models).

    Returns
    -------
    list[dict]
        New list of dicts with `query` replaced by a YAML string.
    """
    new_combs = [c.dict() for c in combs]

    for comb in new_combs:
        query = comb["query"]

        query = clean_up_query(query)

        yaml_string = yaml.safe_dump(query, allow_unicode=True)

        comb["query"] = yaml_string

    return new_combs


def clean_up_query(query: dict) -> dict:
    """
    Normalize a query object for YAML export by pruning/translating fields.

    Transformations
    ---------------
    - If `area.type == 'bbox'`, remove `area.value`.
    - For each entity:
        - Drop `is_area`.
        - If `type == 'cluster'`, stringify `maxDistance` as "<magnitude> <metric>".
        - Else, remove `maxDistance` and `minPoints`.
        - Remove empty `properties`, else prune null operator/value pairs.
    - Flatten `relations` from an object with key 'relations' to the list itself.
      Remove if None; otherwise stringify relation `value` distances to "<magnitude> <metric>".

    Parameters
    ----------
    query : dict
        Query dict to normalize (mutated copy is returned).

    Returns
    -------
    dict
        Normalized query dictionary.
    """
    area = query['area']
    if area['type'] == 'bbox':
        area.pop('value', None)
    for entity in query["entities"]:
        entity.pop("is_area", None)
        if entity["type"] == 'cluster':
            emag = entity['maxDistance']['magnitude']
            emet = entity['maxDistance']['metric']
            entity['maxDistance'] = f'{emag} {emet}'
        else:
            entity.pop('maxDistance')
            entity.pop('minPoints')
        if len(entity["properties"]) == 0:
            entity.pop('properties', None)
        else:
            for property in entity["properties"]:
                if property["operator"] is None and property["value"] is None:
                    property.pop('operator', None)
                    property.pop('value', None)
    query["relations"] = query["relations"]["relations"]
    if query["relations"] is None:
        query.pop('relations', None)
    else:
        for relation in query["relations"]:
            if relation["value"] is None:
                relation.pop('value', None)
            else:
                rel_magnitude = relation["value"]['magnitude']
                rel_metric = relation["value"]['metric']
                relation['value'] = f'{rel_magnitude} {rel_metric}'
    return query


def split_descriptors(descriptors: str) -> Set[str]:
    """
    Split a pipe-delimited descriptor string into a **deduplicated set** of lowercase descriptors.

    Example
    -------
    "Coffee| Café |COFFEE " → {"coffee", "café"}

    Parameters
    ----------
    descriptors : str
        Pipe-separated descriptor string.

    Returns
    -------
    set[str]
        Deduplicated, lowercased descriptors.
    """
    processed_descriptors = set()

    for descriptor in descriptors.split('|'):
        descriptor = descriptor.lstrip().strip().lower()
        if len(descriptor) == 0:
            continue
        processed_descriptors.add(descriptor)

    return processed_descriptors


class CompoundTagPropertyProcessor:
    """
    Expand and normalize compound tag expressions (with lists and separators).

    Supports inputs like:
        [highway|railway]=[primary|secondary]
    and produces all normalized combinations using `SEPERATORS`.
    """
    def expand_list(self, tag_compounds: str) -> List[str]:
        """
        Expand a bracketed, pipe-delimited list into individual items.

        Example
        -------
        "[a|b|c]" → ["a", "b", "c"]

        Parameters
        ----------
        tag_compounds : str
            Raw list string (possibly containing quotes/brackets).

        Returns
        -------
        List[str]
            Cleaned items with quotes/brackets removed; empty entries dropped.
        """
        processed_tag_compounds = []
        tag_compounds = tag_compounds.split('|')
        for tag_compound in tag_compounds:
            tag_compound = tag_compound.replace('[', '').replace(']', '').replace('"', '')
            if len(tag_compound) != 0:
                processed_tag_compounds.append(tag_compound)
        return processed_tag_compounds

    def run(self, tag_compounds: str) -> List[str]:
        """
        Expand a compound tag pattern into all concrete key<op>value combinations.

        The operator is detected by scanning `SEPERATORS`. Keys/values may be lists (in brackets)
        or singletons. All results are lowercased and normalized.

        Examples
        --------
        - '[highway|railway]=[primary|secondary]'
        - 'amenity~[restaurant|cafe]'

        Parameters
        ----------
        tag_compounds : str
            Compound tag string with an operator present.

        Returns
        -------
        List[str]
            All combinations like 'key=val', 'key~val', etc.

        Raises
        ------
        AssertionError
            If no operator from `SEPERATORS` is found in the input.
        """
        selected_seperator = None

        for seperator in SEPERATORS:
            _tag_compounds = tag_compounds.split(seperator)

            if len(_tag_compounds) == 2:
                tag_compounds_keys = _tag_compounds[0]
                tag_compounds_values = _tag_compounds[1]
                selected_seperator = seperator
            else:
                continue

        assert selected_seperator

        if '[' in tag_compounds_keys:
            tag_compounds_keys = self.expand_list(tag_compounds_keys)

        if '[' in tag_compounds_values:
            tag_compounds_values = self.expand_list(tag_compounds_values)

        if isinstance(tag_compounds_values, str):
            tag_compounds_values = [tag_compounds_values]

        if isinstance(tag_compounds_keys, str):
            tag_compounds_keys = [tag_compounds_keys]

        processed_tag_compounds = []
        for tag_key, tag_value in itertools.product(tag_compounds_keys, tag_compounds_values):
            processed_tag_compounds.append(f'{tag_key.lower()}{selected_seperator}{tag_value.lower()}')

        return processed_tag_compounds
