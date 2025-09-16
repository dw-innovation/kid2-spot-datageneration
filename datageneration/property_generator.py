import numpy as np
import pandas as pd
from typing import List, Dict

from datageneration.data_model import TagPropertyExample, TagProperty, Property, ColorBundle
from datageneration.utils import get_random_integer, get_random_decimal_with_metric

"""
Property generation utilities for dataset creation.

This module provides helpers to:
- Build color bundles from a CSV and map them to color-related tag keys.
- Generate `Property` objects (numeric, non-numeric, and color) based on
  `TagProperty` definitions and example pools.

Key Concepts:
- `TagProperty` represents a property template with descriptors and tag patterns.
- `Property` is an instantiated property with an optional operator and value.
- "Colour" keys are treated specially using color bundles.
"""

def fetch_color_bundle(property_examples: List[TagPropertyExample], bundle_path: str)->Dict[str,List[str]]:
    """
    Build a mapping from each colour-related tag key to a list of related colour descriptors.

    This reads a CSV containing colour bundles and, for each example under colour-related
    tag keys (e.g., keys that contain 'colour'), returns all colour descriptors that appear
    in the same bundle as any of the example descriptors.

    Parameters
    ----------
    property_examples : List[TagPropertyExample]
        A list of property example dictionaries. Each item is expected to have:
        - 'key': str — the tag key (e.g., 'colour')
        - 'examples': List[str] — example descriptors under that key
    bundle_path : str
        Path to a CSV file with at least a column named 'Colour Descriptors' that contains
        comma-separated descriptors for a bundle.

    Returns
    -------
    Dict[str, List[str]]
        A mapping from colour-related tag keys (e.g., 'colour') to a **deduplicated**
        list of related colour descriptors. Example:
        {
            "building:colour": ["red", "crimson", "maroon"],
            "roof:colour": ["beige", "sand", "tan"]
        }

    Notes
    -----
    - The CSV is expected to contain a 'Colour Descriptors' column with comma-separated values.
    - Only examples whose 'key' contains 'colour' are considered.
    """
    data = pd.read_csv(bundle_path)
    data = data.to_dict('records')
    color_examples = [item for item in property_examples if 'colour' in item['key']]

    color_bundles = []
    color_bundles_with_tags = {}

    for color_bundle in data:
        color_bundles.append(ColorBundle(descriptors = [x.strip() for x in color_bundle['Colour Descriptors'].split(',')],
                    color_values = [x.strip() for x in color_bundle['Colour Descriptors'].split(',')]))

    for color_example in color_examples:
        color_example_key = color_example['key']
        related_color_examples = color_example['examples']

        related_colors = []
        for related_color_example in related_color_examples:
            for color_bundle in color_bundles:
                if related_color_example in color_bundle.descriptors:
                    related_colors.extend(color_bundle.descriptors)

        related_colors = list(set(related_colors))
        color_bundles_with_tags[color_example_key] = related_colors

    return color_bundles_with_tags


class PropertyGenerator:
    """
    Generator for creating concrete `Property` instances from abstract `TagProperty` specs.

    Parameters
    ----------
    named_property_examples : List[TagPropertyExample]
        Example pools keyed by tag pattern (e.g., "name=***example***") used to fill
        non-numerical properties when a concrete value is needed.
    color_bundles : List[ColorBundle]
        Pre-built colour bundles used for colour properties.

    Attributes
    ----------
    named_property_examples : List[TagPropertyExample]
        Stored reference to example pools.
    color_bundles : List[ColorBundle]
        Stored reference to colour bundles.
    tasks : List
        Reserved for future scheduling/aggregation of generation tasks.
    """
    def __init__(self, named_property_examples: List[TagPropertyExample],
        color_bundles: List[ColorBundle]
        ):
        self.named_property_examples = named_property_examples
        self.color_bundles = color_bundles

        self.tasks = []

    def select_named_property_example(self, property_name: str) -> List[str]:
        """
        Retrieve the example list for a given named property tag pattern.

        Parameters
        ----------
        property_name : str
            The tag pattern key to search (e.g., "name=***example***" or "brand~***example***").

        Returns
        -------
        List[str]
            A list of example strings if found; otherwise `None`-like (falls through without explicit return).

        Notes
        -----
        Returns immediately when the first matching 'key' is found.
        """
        for item in self.named_property_examples:
            if item['key'] == property_name:
                return item['examples']

    def generate_non_numerical_property(self, tag_properties) -> Property:
        """
        Generate a non-numeric `Property` from a `TagProperty` template.

        The method:
        - Randomly picks a descriptor from `tag_properties.descriptors`.
        - If the first tag in `tag_properties.tags` has a concrete value (i.e., not "***example***"),
          returns a `Property` with only the descriptor (no value/operator).
        - Otherwise, chooses one of the tag patterns in `tag_properties.tags`, resolves the operator
          from the pattern, fetches an example from the example pool, and returns a fully populated `Property`.

        Parameters
        ----------
        tag_properties : TagProperty
            The non-numeric tag property template containing descriptors and tag patterns.

        Returns
        -------
        Property
            A concrete property with `name` always set; `operator`/`value` set when a matching example exists.

        Warnings
        --------
        Prints to stdout if the operator cannot be determined from the tag pattern.
        """
        # todo: ipek -- i noticed that we haven't assign operator is equal initially the solution should uncomment the below line
        # operator = '='
        descriptor = np.random.choice(tag_properties.descriptors, 1)[0]

        if tag_properties.tags[0].value != "***example***":
            return Property(name=descriptor)

        # In case of bundle "name + brand", randomly select one of them
        selected_property = np.random.choice(tag_properties.tags)
        tag = selected_property.key + selected_property.operator + selected_property.value
        property_examples = self.select_named_property_example(tag)
        if not property_examples:
            print("=> NO VALUE!! - ", tag)
            return Property(name=descriptor)
            # return Property(key=tag_property.key, operator=tag_property.operator,value=tag_property.value, name=tag_property.value)

        if "~***example***" in tag:
            operator = "~"
        elif "=***example***" in tag:
            operator = "="
        else:
            print("Something does not seem to be right. Please check operator of property ", tag, "!")

        selected_example = np.random.choice(property_examples)

        return Property(name=descriptor, operator=operator, value=selected_example)

    def generate_numerical_property(self, tag_property: TagProperty) -> Property:
        """
        Generate a numeric `Property` using either integer or decimal-with-metric values.

        Logic
        -----
        - Randomly selects a descriptor from `tag_property.descriptors`.
        - Randomly selects an operator from {">", "=", "<"}.
        - If the first (canonical) tag key is "height", generates a decimal with a metric
          (e.g., "3.2 m") via `get_random_decimal_with_metric`.
        - Otherwise, generates a random integer as a string via `get_random_integer`.

        Parameters
        ----------
        tag_property : TagProperty
            The numeric tag property template.

        Returns
        -------
        Property
            A concrete numeric property with `name`, `operator`, and string `value`.
        """
        # todo --> we might need specific numerical function if we need to define logical max/min values.
        descriptor = np.random.choice(tag_property.descriptors, 1)[0]
        # operator = "="
        operator = np.random.choice([">", "=", "<"])
        tag = tag_property.tags[0]
        if tag.key == "height":
            # todo rename this
            generated_numerical_value = get_random_decimal_with_metric(max_digits=5)
            generated_numerical_value = f'{generated_numerical_value.magnitude} {generated_numerical_value.metric}'
        else:
            # todo rename this
            generated_numerical_value = str(get_random_integer(max_digits=3))

        return Property(name=descriptor, operator=operator, value=generated_numerical_value)
        # return Property(key=tag_property.key, operator=tag_aproperty.operator, value=generated_numerical_value, name=tag_property.key)

    def generate_color_property(self, tag_attribute: TagProperty) -> Property:        """
        Generate a colour `Property` by sampling from precomputed colour bundles.

        For all tag patterns in `tag_attribute.tags`, uses their concatenated key form
        (e.g., "colour=***example***") to look up the list of related colours from
        `self.color_bundles`, aggregates them, and samples one colour.

        Parameters
        ----------
        tag_attribute : TagProperty
            A tag property whose tags relate to colour.

        Returns
        -------
        Property
            A property with a sampled colour value and '=' operator.
        """
        bundles_to_select = []
        for tag in tag_attribute.tags:
            tag_key = f'{tag.key}{tag.operator}{tag.value}'
            bundles_to_select.extend(self.color_bundles[tag_key])
        selected_color = np.random.choice(bundles_to_select, 1)[0]
        selected_descriptor = np.random.choice(tag_attribute.descriptors)
        return Property(name=selected_descriptor, operator='=', value=selected_color)

    def categorize_properties(self, tag_properties: List[TagProperty]):
        """
        Categorize tag properties into buckets for downstream sampling logic.

        Buckets
        -------
        - 'numerical': any tag with value '***numeric***'
        - 'popular_non_numerical': tags for common proper nouns/addresses (brand, name, housenumber, street)
        - 'colour': tags whose key contains 'colour'
        - 'rare_non_numerical': tags whose key contains 'cuisine'
        - 'other_non_numerical': everything else

        Parameters
        ----------
        tag_properties : List[TagProperty]
            A list of tag property templates to categorize.

        Returns
        -------
        Dict[str, List[TagProperty]]
            Mapping from bucket name to the list of `TagProperty` objects in that bucket.
        """
        categories = {}
        for tag_property in tag_properties:
            tag_property_tags = tag_property.tags
            for tag_property_tag in tag_property_tags:
                if tag_property_tag.value == '***numeric***':
                    if 'numerical' not in categories:
                        categories['numerical'] = []
                    categories['numerical'].append(tag_property)
                elif ('brand' in tag_property_tag.key or 'name' in tag_property_tag.key or
                      'addr:housenumber' == tag_property_tag.key or 'addr:street' == tag_property_tag.key):
                    if 'popular_non_numerical' not in categories:
                        categories['popular_non_numerical'] = []
                    categories['popular_non_numerical'].append(tag_property)
                elif 'colour' in tag_property_tag.key:
                    if 'colour' not in categories:
                        categories['colour'] = []
                    categories['colour'].append(tag_property)
                elif 'cuisine' in tag_property_tag.key:
                    if 'rare_non_numerical' not in categories:
                        categories['rare_non_numerical'] = []
                    categories['rare_non_numerical'].append(tag_property)
                else:
                    if 'other_non_numerical' not in categories:
                        categories['other_non_numerical'] = []
                    categories['other_non_numerical'].append(tag_property)
        return categories

    def run(self, tag_property: TagProperty) -> Property:
        """
        Generate a concrete `Property` from a `TagProperty` by dispatching on type.

        Decision Flow
        -------------
        1. If any tag value equals '***numeric***' → numeric generator.
        2. Else if any tag key contains 'colour' → colour generator.
        3. Else → non-numeric named generator.

        Parameters
        ----------
        tag_property : TagProperty
            The input tag property template.

        Returns
        -------
        Property
            The generated property instance.
        """
        # if '***numeric***' in tag_property.value:
        if any(t.value == '***numeric***' for t in tag_property.tags):
            generated_property = self.generate_numerical_property(tag_property)
        else:
            if any('colour' in t.key for t in tag_property.tags):
                generated_property = self.generate_color_property(tag_property)
            else:
                generated_property = self.generate_non_numerical_property(tag_property)
        return generated_property
