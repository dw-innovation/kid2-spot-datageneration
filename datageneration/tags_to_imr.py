import pandas as pd
import json
from argparse import ArgumentParser
from itertools import chain, product
from tqdm import tqdm
from typing import List, Dict, Union

from datageneration.data_model import Tag
from datageneration.utils import write_output, SEPERATORS, split_descriptors, write_dict_output

"""
Transform OSM-style tag expressions into an IMR (Intermediate Model Representation).

This module converts tag bundles from a primary key table into a graph-friendly IMR format:
- Parses AND/OR logic across tags.
- Supports key/value unions like `key1|key2=value1|value2`.
- Emits Tag objects (from datageneration.data_model) or nested dicts: {"and": [...]}, {"or": [...]}.

CLI:
    python transform_tags_to_imr.py \
        --primary_key_table data/primary_keys.xlsx \
        --output_file out/imr.jsonl
"""

def generate_and_condition(conditions: List) -> Dict[str, List[Tag]]:
    """
    Build an AND condition from a list of condition lists.

    Parameters
    ----------
    conditions : List[Iterable[Tag]]
        Each element is a list/iterable of Tag objects (a clause). They are flattened and joined by AND.

    Returns
    -------
    Dict[str, List[Tag]]
        {"and": [Tag, Tag, ...]} — a single AND clause containing all tags from all input clauses.
    """
    res = {"and": list(chain.from_iterable(conditions))}
    return res


def generate_or_condition(conditions: List) -> Union[List[Tag], Dict[str, List[Tag]]]:
    """
    Build an AND condition from a list of condition lists.

    Parameters
    ----------
    conditions : List[Iterable[Tag]]
        Each element is a list/iterable of Tag objects (a clause). They are flattened and joined by AND.

    Returns
    -------
    Dict[str, List[Tag]]
        {"and": [Tag, Tag, ...]} — a single AND clause containing all tags from all input clauses.
    """
    first_condition = conditions[0]

    if isinstance(first_condition, Tag) or len(conditions) > 1:
        return {"or": conditions}

    if isinstance(first_condition, dict):
        return first_condition

    return conditions[0]


def transform_tags_to_imr(tags_str: str) -> List[Dict[str, List[Tag]]]:
    """
    Convert a tag expression string into IMR.

    Splits comma-separated groups and converts each into a disjunction/conjunction of Tag filters.
    Currently, the implementation returns a single-element list containing an OR block (or passthrough).

    Parameters
    ----------
    tags_str : str
        Tag expression, e.g. "amenity=restaurant|cafe, cuisine=italian AND diet:vegan=yes".

    Returns
    -------
    List[Union[Dict[str, Any], List[Tag], Tag]]
        A list with a single element representing the IMR for the input tag(s).
        Example: [{"or": [Tag(...), {"and": [Tag(...), Tag(...)]}, ...]}]
    """
    if "," in tags_str:
        tags = [t_.strip() for t_ in tags_str.split(',')]
    else:
        tags = [tags_str]

    result = []
    if tags:
        result.append(generate_or_condition(list(yield_tag_filters_for_imr(tags))))
    return result if isinstance(result[0], list) else result


def yield_tag_filters_for_imr(tags: Union[str, List[str]]) -> List[Tag]:
    """
    Yield Tag filters (and grouped AND blocks) to build IMR.

    Handles:
    - AND groups: "a=b AND c=d"
    - Key/value unions: "a|b=c|d" → all cartesian products (a=c, a=d, b=c, b=d)
    - Operators found in SEPERATORS; defaults to "=" if none found.

    Parameters
    ----------
    tags : Union[str, List[str]]
        A single tag string or a list of tag strings.

    Yields
    ------
    Union[Tag, Dict[str, List[Tag]]]
        Either Tag objects, or {"and": [Tag, ...]} blocks.
    """
    if isinstance(tags, str):
        tags = [tags]
    for tag in tags:
        if not tag:
            continue
        if "AND" in tag:
            and_list = [t_.strip() for t_ in tag.split('AND')]
            flt_list = [yield_tag_filters_for_imr(al) for al in and_list]
            yield generate_and_condition(flt_list)
        else:
            op = next((o for o in SEPERATORS if o in tag), "=")
            tag_key, tag_value = tag.split(op)
            tag_key = [k.strip(" \"[]") for k in tag_key.split("|")]
            tag_value = [v.strip(" \"[]") for v in tag_value.split("|")]

            for comb in product(tag_key, tag_value):
                yield Tag(key=comb[0], operator=op, value=comb[1])

def tag_serializer(tag):
    """
    Convert a Tag object to a serializable dictionary.

    Parameters
    ----------
    tag : Tag
        The Tag to serialize.

    Returns
    -------
    Dict[str, Any]
        tag.to_dict() result.
    """
    return tag.to_dict()

if __name__ == '__main__':
    """
    Load the Primary Key table, transform each row's `tags` into IMR, and write JSON lines.

    For each descriptor in a row, an entry is emitted:
        {"key": <descriptor>, "imr": <IMR for row tags>}
    """
    parser = ArgumentParser()
    parser.add_argument('--primary_key_table', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()

    output_file = args.output_file
    tag_list_path = args.primary_key_table

    if '.csv' in tag_list_path:
        primary_key_table = pd.read_csv(args.primary_key_table)
    else:
        primary_key_table = pd.read_excel(args.primary_key_table, engine='openpyxl')

    results = []
    for row in tqdm(primary_key_table.to_dict(orient='records'), total=len(primary_key_table)):
        descriptors_str = row['descriptors']
        tags_str = row['tags']
        if isinstance(descriptors_str, float):
            print(tags_str)
            continue
        descriptors = split_descriptors(descriptors_str)
        tags = json.loads(json.dumps(transform_tags_to_imr(tags_str), default=tag_serializer))

        for descriptor in descriptors:
            results.append(dict(cluster_id=row['index'], key=descriptor, imr=tags, descriptors=list(descriptors)))
    write_dict_output(results, output_file, bool_add_yaml=False)
