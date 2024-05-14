import numpy as np
from typing import List

from datageneration.data_model import TagPropertyExample, TagProperty, Property
from datageneration.utils import get_random_integer, get_random_decimal_with_metric

class PropertyGenerator:
    def __init__(self, named_property_examples: List[TagPropertyExample]):
        self.named_property_examples = named_property_examples

    def select_named_property_example(self, property_name: str) -> List[str]:
        for item in self.named_property_examples:
            if item['key'] == property_name:
                return item['examples']

    def generate_non_numerical_property(self, tag_properties) -> Property:
        descriptor = np.random.choice(tag_properties.descriptors, 1)[0]

        if tag_properties.tags[0].value != "***example***":
            return Property(name=descriptor)

        # In case of bundle "name + brand", randomly select one of them
        selected_property = np.random.choice(tag_properties.tags)
        tag = selected_property.key + selected_property.operator + selected_property.value
        property_examples = self.select_named_property_example(tag)
        if not property_examples:
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
        # todo --> we might need specific numerical function if we need to define logical max/min values.
        descriptor = np.random.choice(tag_property.descriptors, 1)[0]
        # operator = "="
        operator = np.random.choice([">", "=", "<"])
        tag = tag_property.tags[0]
        if tag.key == "height":
            # todo rename this
            generated_numerical_value = get_random_decimal_with_metric(max_digits=5)
        else:
            # todo rename this
            generated_numerical_value = str(get_random_integer(max_digits=3))

        return Property(name=descriptor, operator=operator, value=generated_numerical_value)
        # return Property(key=tag_property.key, operator=tag_aproperty.operator, value=generated_numerical_value, name=tag_property.key)

    def generate_color_property(self, tag_attribute: TagProperty) -> Property:
        raise NotImplemented

    def run(self, tag_property: TagProperty) -> Property:
        '''
        Generate a property based on a tag property.

        Parameters:
            tag_property (TagProperty): The tag property object containing information about the property.

        Returns:
            Property: The generated property.

        This method checks the type of the tag property and generates a property accordingly.
        If the property is numeric, it generates a numerical property.
        If the property key contains 'name', it generates a proper noun property.
        If the property key contains 'color', it generates a color property.
        Otherwise, it generates a named property.
        '''
        # if '***numeric***' in tag_property.value:
        if any(t.value == '***numeric***' for t in tag_property.tags):
            generated_property = self.generate_numerical_property(tag_property)
        else:
            if any('colour' in t.key for t in tag_property.tags):
                generated_property = self.generate_color_property(tag_property)
            else:
                generated_property = self.generate_non_numerical_property(tag_property)
        return generated_property
