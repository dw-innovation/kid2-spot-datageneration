import numpy as np
from random import randint
from typing import List

from datageneration.data_model import TagPropertyExample, TagProperty, Property


# numerical value generator
def get_random_decimal_with_metric(max_digits: int) -> str:
    '''
    TODO: this should be reworked -- threshold should be defined based on metric
    '''
    digits = randint(1, max_digits)
    low = np.power(10, digits - 1)
    high = np.power(10, digits) - 1
    num = randint(low, high)
    if np.random.choice([True, False], 1)[0]:
        num = num / np.random.choice([10, 100], 1)[0]

    dist = str(num) + " " + np.random.choice(["m", "km", "in", "ft", "yd", "mi", "le"], 1)[0]  # "cm",

    return dist

def get_random_integer(max_digits: int) -> int:
    digits = randint(1, max_digits)
    low = np.power(10, digits - 1)
    high = np.power(10, digits) - 1

    return randint(low, high)

class PropertyGenerator:
    def __init__(self, named_property_examples: List[TagPropertyExample]):
        self.named_property_examples = named_property_examples

    def select_named_property_example(self, property_name: str) -> List[str]:
        for item in self.named_property_examples:
            if item['key'] == property_name:
                return item['examples']

    def generate_named_property(self, tag_property: TagProperty) -> Property:
        """
        Generate a Property object based on the given TagProperty.

        This function selects a random example for the specified tag property
        combination, shuffles the examples to ensure randomness, and then
        constructs a Property object using the selected example.

        Parameters:
        - tag_property (TagProperty): The TagProperty object containing key, operator,
          and value information to generate the property.

        Returns:
        - Property: A Property object constructed from the selected example.

        Example:
        ```python
        tag_prop = TagProperty(key='cuisine', operator='=', value='italian')
        property_obj = generate_named_property(tag_prop)
        print(property_obj)
        ```
        """
        return self.generate_non_numerical_property(tag_property)

    def generate_non_numerical_property(self, tag_property) -> Property:
        descriptor = np.random.choice(tag_property.descriptors, 1)[0]
        tag = tag_property.tags[0].key + tag_property.tags[0].operator + tag_property.tags[0].value
        property_examples = self.select_named_property_example(tag)
        if not property_examples:
            return Property(name=descriptor)
            # return Property(key=tag_property.key, operator=tag_property.operator,value=tag_property.value, name=tag_property.value)

        if "~***any***" in tag:
            operator = "~"
        elif "=***any***" in tag:
            operator = "="
        else:
            print("Something does not seem to be right. Please check operator of property ", tag, "!")

        np.random.shuffle(property_examples)
        selected_example = property_examples[0]

        return Property(name=descriptor, operator=operator, value=selected_example)

    def generate_proper_noun_property(self, tag_property: TagProperty) -> Property:
        '''Proper nouns are names such as name=Laughen_restaurant'''
        return self.generate_non_numerical_property(tag_property)

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
            if any(t.key == 'name' for t in tag_property.tags):
                generated_property = self.generate_proper_noun_property(tag_property)
            elif any(t.key == 'color' for t in tag_property.tags):
                generated_property = self.generate_color_property(tag_property)
            else:
                generated_property = self.generate_named_property(tag_property)
        return generated_property
