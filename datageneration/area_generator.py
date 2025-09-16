import json
import numpy as np
import pandas as pd
from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Union

from datageneration.data_model import Area
from datageneration.enrich_geo_database import area_names_non_roman_vocab
from datageneration.utils import NON_ROMAN_LANG_GROUPS

"""
Utilities for generating area strings (city/state/country combinations) from a
geo database, with optional non‑Roman script variants.

This module:
- Loads a nested geo database (countries → states → cities).
- Samples areas with a configurable probability of one/two‑word names.
- Optionally translates sampled areas into non‑Roman scripts when available.
- Exposes generators for several output formats (city only, city+country, etc.).
"""


class NamedAreaData(BaseModel):
    """Container for a single area triple.

    Attributes:
        city: City name.
        state: Administrative region / state name.
        country: Country name.
    """
    city: str
    state: str
    country: str


class AREA_TASKS(Enum):
    """Generation task types for area strings."""
    NO_AREA = 'no_area'
    CITY = 'city'
    CITY_AND_COUNTRY = 'city_and_country'
    REGION = 'administrative_region'
    REGION_AND_COUNTRY = 'administrative_region'
    CITY_AND_REGION_AND_COUNTRY = 'city_and_region_and_country'


def load_named_area_data(geolocation_file: str) -> List[NamedAreaData]:
    """Load and partition geo data into single-word and two-word buckets.

    The input JSON is expected to have the structure:
    [{"name": <country>, "states": [{"name": <state>, "cities": [{"name": <city>}, ...]}, ...]}, ...].

    Returns a 4‑tuple of lists of ``NamedAreaData``, split by whether the CITY/STATE
    contains one word or two+ words:

        (cities_single_word, cities_two_words, states_single_word, states_two_words)

    Args:
        geolocation_file: Path to the geo database JSON file.

    Returns:
        A tuple:
            - locs_with_cities_single_word: List[NamedAreaData]
            - locs_with_cities_two_words: List[NamedAreaData]
            - locs_with_states_single_word: List[NamedAreaData]
            - locs_with_states_two_words: List[NamedAreaData]
    """
    geolocation_data = pd.read_json(geolocation_file)

    processed_geolocation_data = []
    locs_with_cities_single_word = []
    locs_with_cities_two_words = []
    locs_with_states_single_word = []
    locs_with_states_two_words = []
    for sample in geolocation_data.to_dict(orient='records'):
        for state in sample['states']:
            for city in state['cities']:
                named_area = NamedAreaData(city=city['name'], state=state['name'], country=sample['name'])
                if len(state['name'].split()) > 1:
                    locs_with_states_two_words.append(named_area)
                else:
                    locs_with_states_single_word.append(named_area)
                if len(city['name'].split()) > 1:
                    locs_with_cities_two_words.append(named_area)
                else:
                    locs_with_cities_single_word.append(named_area)
                processed_geolocation_data.append(named_area)

    return (locs_with_cities_single_word, locs_with_cities_two_words, locs_with_states_single_word,
            locs_with_states_two_words)


def load_non_roman_vocab(non_roman_vocab_file: str) -> Dict:
    """Load non‑Roman transliteration/translation vocabulary.

    The JSON must map each Latin (Roman script) place name to its available
    non‑Roman versions keyed by language code/group.

    Args:
        non_roman_vocab_file: Path to the JSON vocabulary file.

    Returns:
        A dictionary of the non‑Roman vocabulary.
    """
    with open(non_roman_vocab_file, 'r') as json_file:
        non_roman_vocab = json.load(json_file)
    return non_roman_vocab


class NonRomanDoesNotExist(Exception):
    """Raised when a requested non‑Roman version for an area does not exist."""
    def __init__(self, message):
        """Initialize the exception.

        Args:
            message: Explanation of the missing non‑Roman variant.
        """
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"NonRomanDoesNotExist: {self.message}"


class AreaGenerator:
    """Generates area strings in several formats, optionally using non‑Roman scripts.

    Sampling behavior is controlled by:
      - ``prob_of_two_word_areas``: probability to draw from two‑word buckets.
      - ``prob_of_non_roman_areas``: probability to attempt non‑Roman script output.

    Attributes:
        locs_with_cities_single_word: Pool of areas with single‑word city names.
        locs_with_cities_two_words: Pool of areas with multi‑word city names.
        locs_with_states_single_word: Pool of areas with single‑word state names.
        locs_with_states_two_words: Pool of areas with multi‑word state names.
        area_non_roman_vocab: Mapping of Latin names to non‑Roman versions.
        tasks: List of available task identifiers (strings).
        prob_of_two_word_areas: Probability of drawing a two‑word city/state.
        prob_of_non_roman_areas: Probability of outputting non‑Roman variants.
        non_roman_lang_group_keys: Language‑group keys available for translation.
    """
    def __init__(self, geolocation_file: str, non_roman_vocab_file: str, prob_of_two_word_areas: float,
                 prob_of_non_roman_areas: float):
        """Initialize the generator with data sources and sampling probabilities.

        Args:
            geolocation_file: Path to the geo database JSON file.
            non_roman_vocab_file: Path to the non‑Roman vocabulary JSON file.
            prob_of_two_word_areas: Probability in [0, 1] to draw a two‑word
                city/state when sampling.
            prob_of_non_roman_areas: Probability in [0, 1] to attempt translation
                into a non‑Roman script for the sampled area.

        Raises:
            ValueError: If probabilities are outside [0, 1].
        """
        (self.locs_with_cities_single_word, self.locs_with_cities_two_words, self.locs_with_states_single_word,
         self.locs_with_states_two_words) = load_named_area_data(geolocation_file)
        self.area_non_roman_vocab = load_non_roman_vocab(non_roman_vocab_file)
        self.tasks = [area_task.value for area_task in AREA_TASKS]
        self.two_word_city_selection = [True, False]
        self.two_word_state_selection = [True, False]
        self.prob_of_two_word_areas = prob_of_two_word_areas
        self.prob_of_non_roman_areas = prob_of_non_roman_areas
        self.non_roman_lang_group_keys = list(NON_ROMAN_LANG_GROUPS.keys())

    def translate_into_non_roman(self, area: NamedAreaData, target_lang_group: List) -> Union[NamedAreaData, None]:
        """Translate a Latin‑script area into a non‑Roman script, if available.

        Iterates over the languages in the provided language group and returns
        the first available full triple (city/state/country) translation.

        Note:
            ``target_lang_group`` is expected to be a key in
            ``NON_ROMAN_LANG_GROUPS`` (e.g., a language family or group name).

        Args:
            area: The area triple to translate.
            target_lang_group: Key identifying a language group in
                ``NON_ROMAN_LANG_GROUPS``.

        Returns:
            A new ``NamedAreaData`` with non‑Roman script names if a matching
            language is found; otherwise ``None``.

        Raises:
            NonRomanDoesNotExist: If any of city/state/country is missing from
                the non‑Roman vocabulary entirely.
        """
        target_langs = NON_ROMAN_LANG_GROUPS[target_lang_group]
        np.random.shuffle(target_langs)

        for target_lang in target_langs:
            if any((area.city not in self.area_non_roman_vocab,
                    area.state not in self.area_non_roman_vocab,
                    area.country not in self.area_non_roman_vocab)):
                raise NonRomanDoesNotExist(f"any of city, state, country in {area} does not have a non-roman version")
            if any((target_lang not in self.area_non_roman_vocab[area.city]['non_roman_versions'],
                    target_lang not in self.area_non_roman_vocab[area.state]['non_roman_versions'],
                    target_lang not in self.area_non_roman_vocab[area.country]['non_roman_versions'])):
                continue
            translated_city = self.area_non_roman_vocab[area.city]['non_roman_versions'][target_lang]
            translated_state = self.area_non_roman_vocab[area.state]['non_roman_versions'][target_lang]
            translated_country = self.area_non_roman_vocab[area.country]['non_roman_versions'][target_lang]
            return NamedAreaData(city=translated_city, state=translated_state, country=translated_country)

    def get_area(self, required_type=None) -> NamedAreaData:
        """Sample a random area triple, honoring one/two‑word and script settings.

        Depending on ``prob_of_two_word_areas``, draws from single‑word or
        two‑word buckets. If ``required_type`` is provided, that bucket choice
        is applied specifically to the city or state list. With
        ``prob_of_non_roman_areas``, attempts to translate the sampled triple
        into a non‑Roman script, if possible.

        Args:
            required_type: If provided, restricts the one/two‑word selection to
                the specified field—either ``"city"`` or ``"state"``. If ``None``,
                the selection is random between city/state.

        Returns:
            A ``NamedAreaData`` triple (possibly in a non‑Roman script).
        """
        use_two_word_area = np.random.choice([True, False], p=[self.prob_of_two_word_areas,
                                                               1 - self.prob_of_two_word_areas])

        if use_two_word_area:
            if required_type == "city":
                draft_from = self.locs_with_cities_two_words
            elif required_type == "state":
                draft_from = self.locs_with_states_two_words
            else:
                idx = np.random.choice([0, 1])
                draft_from = [self.locs_with_cities_two_words, self.locs_with_states_two_words][idx]
        else:
            if required_type == "city":
                draft_from = self.locs_with_cities_single_word
            elif required_type == "state":
                draft_from = self.locs_with_states_single_word
            else:
                idx = np.random.choice([0, 1])
                draft_from = [self.locs_with_cities_single_word, self.locs_with_states_single_word][idx]

        selected_area = np.random.choice(draft_from)

        use_non_roman_alphabets = np.random.choice([True, False], p=[self.prob_of_non_roman_areas,
                                                                     1 - self.prob_of_non_roman_areas])

        if use_non_roman_alphabets:
            np.random.shuffle(self.non_roman_lang_group_keys)
            tmp_selected_area = None
            for selected_lang_group in self.non_roman_lang_group_keys:
                try:
                    tmp_selected_area = self.translate_into_non_roman(area=selected_area,
                                                                      target_lang_group=selected_lang_group)
                except NonRomanDoesNotExist:
                    selected_area = np.random.choice(draft_from)

                if not tmp_selected_area:
                    continue

                else:
                    selected_area = tmp_selected_area
                    break
        return selected_area


    def generate_no_area(self) -> Area:
        """Return an empty/bbox area.

        Returns:
            Area: An ``Area`` with ``type='bbox'`` and empty value.
        """
        return Area(type='bbox', value='')


    def generate_city_area(self) -> Area:
        """Generate a city‑only area string.

        Randomly samples a city (respecting the one/two‑word probability) and
        returns its name.

        Example:
            ``"Koblenz"``

        Returns:
            Area: An ``Area`` with ``type='area'`` and the city name as value.
        """
        area = self.get_area("city")

        return Area(type='area', value=area.city)


    def generate_city_and_country_area(self) -> Area:
        """Generate a ``city, country`` area string.

        Samples a city and appends its country.

        Example:
            ``"Koblenz, Germany"``

        Returns:
            Area: An ``Area`` with ``type='area'`` and ``"city, country"`` value.
        """
        area = self.get_area("city")

        return Area(type='area', value=f'{area.city}, {area.country}')


    def generate_region_area(self) -> Area:
        """Generate a state/administrative‑region area string.

        Samples a state/region (respecting the one/two‑word probability) and
        returns its name.

        Example:
            ``"Rheinland-Pfalz"``

        Returns:
            Area: An ``Area`` with ``type='area'`` and the state/region name.
        """
        area = self.get_area("state")

        return Area(type='area', value=area.state)


    def generate_region_and_country_area(self) -> Area:
        """Generate a ``state, country`` area string.

        Samples a state/region and appends its country.

        Example:
            ``"Rheinland-Pfalz, Germany"``

        Note:
            This may be problematic for countries without states/regions.

        Returns:
            Area: An ``Area`` with ``type='area'`` and ``"state, country"`` value.
        """
        # todo: this would be problematic when the country does not have states
        area = self.get_area("state")

        return Area(type='area', value=f'{area.state}, {area.country}')


    def generate_city_and_region_and_country_area(self) -> Area:
        """Generate a ``city, state, country`` area string.

        Samples a full triple and formats it as ``"city, state, country"``.

        Example:
            ``"Koblenz, Rheinland-Pfalz, Germany"``

        Note:
            This may be problematic for countries without states/regions.

        Returns:
            Area: An ``Area`` with ``type='area'`` and the full triple.
        """
        # todo: this would be problematic when the country does not have states
        area = self.get_area()

        return Area(type='area', value=f'{area.city}, {area.state}, {area.country}')


    def run(self) -> List[Area]:
        """Run one random generation task from ``AREA_TASKS``.

        Randomly selects a task and invokes the corresponding generator.

        Returns:
            Area: The generated ``Area`` object for the selected task.

        Note:
            The return type annotation says ``List[Area]`` but the method returns
            a single ``Area`` instance.
        """
        np.random.shuffle(self.tasks)
        selected_task = self.tasks[0]

        if selected_task == AREA_TASKS.NO_AREA.value:
            return self.generate_no_area()

        elif selected_task == AREA_TASKS.CITY.value:
            return self.generate_city_area()

        # elif selected_task == AREA_TASKS.DISTRICT.value:
        #     # todo: we probably need more comprehensive geolocation data, districs are not in the dataset
        #     return NotImplemented

        elif selected_task == AREA_TASKS.CITY_AND_COUNTRY.value:
            return self.generate_city_and_country_area()

        elif selected_task == AREA_TASKS.REGION.value:
            return self.generate_region_area()

        elif selected_task == AREA_TASKS.REGION_AND_COUNTRY.value:
            return self.generate_region_and_country_area()

        elif selected_task == AREA_TASKS.CITY_AND_REGION_AND_COUNTRY.value:
            return self.generate_city_and_region_and_country_area()
