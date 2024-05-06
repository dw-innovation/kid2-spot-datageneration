import numpy as np
import pandas as pd
from enum import Enum
from pydantic import BaseModel
from typing import List

from datageneration.data_model import Area


class NamedAreaData(BaseModel):
    city: str
    state: str
    country: str


class AREA_TASKS(Enum):
    NO_AREA = 'no_area'
    CITY = 'city'
    CITY_AND_COUNTRY = 'city_and_country'
    CITY_AND_REGION_AND_COUNTRY = 'city_and_region_and_country'
    ADMINISTRATIVE_REGION = 'administrative_region'


def load_named_area_data(geolocation_file: str) -> List[NamedAreaData]:
    geolocation_data = pd.read_json(geolocation_file)

    processed_geolocation_data = []
    for sample in geolocation_data.to_dict(orient='records'):
        for state in sample['states']:
            for city in state['cities']:
                processed_geolocation_data.append(
                    NamedAreaData(city=city['name'], state=state['name'], country=sample['name']))
    return processed_geolocation_data


class AreaGenerator:
    def __init__(self, geolocation_data: List[NamedAreaData]):
        self.geolocation_data = geolocation_data
        self.locs_with_cities_single_word, self.locs_with_cities_two_words = self.categorize_cities_with_two_words(
            geolocation_data)
        self.locs_with_states_single_word, self.locs_with_states_two_words = self.categorize_states_with_two_words(
            geolocation_data)
        self.tasks = [area_task.value for area_task in AREA_TASKS]
        self.two_word_city_selection = [True, False]
        self.two_word_state_selection = [True, False]

    def generate_no_area(self) -> Area:
        '''
        It returns no area, bbox
        '''
        return Area(type='bbox', value='')

    def generate_city_area(self) -> Area:
        '''
        Randomly select if select the city from cities with single word or two word. After corresponding category is selected, we will suffle the corresponding list and then select the city name
        e.g. Koblenz
        '''
        np.random.shuffle(self.two_word_city_selection)
        selection_city_with_two_words = self.two_word_city_selection[0]
        if selection_city_with_two_words:
            np.random.shuffle(self.locs_with_cities_two_words)
            selected_area = self.locs_with_states_two_words[0]
        else:
            np.random.shuffle(self.locs_with_cities_single_word)
            selected_area = self.locs_with_cities_single_word[0]

        return Area(type='area', value=selected_area.city)

    def generate_city_and_country_area(self) -> Area:
        '''
        Randomly shuffles the geolocation data point
        Selects the city name and return city_name, country_name where city is located.
        e.g Koblenz, Germany
        '''
        np.random.shuffle(self.geolocation_data)
        selected_area = self.geolocation_data[0]
        return Area(type='area', value=f'{selected_area.city}, {selected_area.country}')

    def generate_city_and_region_and_country(self) -> Area:
        '''
        Randomly shuffles the geolocation data point
        Selects the city name and return city_name, state_name and then country_name where city is located.
        e.g Koblenz, Rheinland-Palastine, Germany
        '''
        # todo: this would be problematic when the country does not have states
        np.random.shuffle(self.geolocation_data)
        selected_area = self.geolocation_data[0]
        return Area(type='area', value=f'{selected_area.city}, {selected_area.state}, {selected_area.country}')

    def generate_administrative_region(self) -> Area:
        '''
        It filters the unique states in geolocation data points
        Randomly shuffles it
        Selects the state
        e.g Rheinland-Palastine
        '''
        np.random.shuffle(self.two_word_state_selection)
        selection_state_with_two_words = self.two_word_state_selection[0]
        if selection_state_with_two_words:
            states = [area.state for area in self.locs_with_states_two_words]
            np.random.shuffle(states)
            selected_state = states[0]
        else:
            states = [area.state for area in self.locs_with_states_single_word]
            np.random.shuffle(states)
            selected_state = states[0]

        return Area(type='area', value=selected_state)

    def categorize_cities_with_two_words(self, geolocation_data) -> Tuple[List[NamedAreaData], List[NamedAreaData]]:
        '''

        :param geolocation_data:
        :return:
        '''
        locs_with_cities_single_word = []
        locs_with_cities_two_words = []
        for loc in geolocation_data:
            if len(loc.city.split()) > 1:
                locs_with_cities_two_words.append(loc)
            else:
                locs_with_cities_single_word.append(loc)
        return (locs_with_cities_single_word, locs_with_cities_two_words)

    def categorize_states_with_two_words(self, geolocation_data) -> Tuple[List[NamedAreaData], List[NamedAreaData]]:
        '''

        :param geolocation_data:
        :return:
        '''
        locs_with_states_single_word = []
        locs_with_states_two_words = []
        for loc in geolocation_data:
            if len(loc.state.split()) > 1:
                locs_with_states_two_words.append(loc)
            else:
                locs_with_states_single_word.append(loc)
        return (locs_with_states_single_word, locs_with_states_two_words)

    def run(self) -> List[Area]:
        '''
        This function a random generation pipeline. That randomly selects the task function which are defined in AREA_TASKS. Next, it calls the generator function that is corresponding to the selected task.
        '''
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

        elif selected_task == AREA_TASKS.CITY_AND_REGION_AND_COUNTRY.value:
            return self.generate_city_and_region_and_country()

        elif selected_task == AREA_TASKS.ADMINISTRATIVE_REGION.value:
            return self.generate_administrative_region()
