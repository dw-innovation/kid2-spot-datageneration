from enum import Enum
from typing import List

import numpy as np
from datageneration.data_model import Relation, Relations
from datageneration.property_generator import get_random_decimal_with_metric


class RELATION_TASKS(Enum):
    INDIVIDUAL_DISTANCES = 'individual_distances'
    WITHIN_RADIUS = 'within_radius'
    IN_AREA = 'in_area'


# def get_random_decimal_with_metric(range):
#     '''
#     TODO: this should be reworked -- threshold should be defined based on metric
#     '''
#     h_ = np.random.choice(np.arange(range), 1)[0]
#     if np.random.choice([True, False], 1)[0]:
#         h_ = h_ / np.random.choice([10, 100], 1)[0]
#
#     h_ = str(h_) + " " + np.random.choice(["m", "km", "in", "ft", "yd", "mi", "le"], 1)[0]  # "cm",
#     return h_


class RelationGenerator:
    def __init__(self, max_distance_digits: int):
        self.MAX_DISTANCE_DIGITS = max_distance_digits
        self.tasks = [relation_task.value for relation_task in RELATION_TASKS]

    def generate_individual_distances(self, num_entities: int) -> List[Relation]:
        relations = []
        for t_no in range(num_entities):
            if t_no != num_entities - 1:
                relations.append(
                    Relation(name='dist', source=t_no, target=t_no + 1,
                             value=get_random_decimal_with_metric(self.MAX_DISTANCE_DIGITS)))
        return relations

    def generate_within_radius(self, num_entities: int) -> List[Relation]:
        """
        Generate relations representing entities within a certain radius.
        Args:
            num_entities (int): The number of entities for which relations need to be generated.
        Returns:
            List[Relation]: A list of Relation objects representing entities within a radius.
        """
        relations = []
        distance = get_random_decimal_with_metric(self.MAX_DISTANCE_DIGITS)
        for t_no in range(num_entities):
            if t_no != num_entities - 1:
                relations.append(
                    Relation(name='dist', source=0, target=t_no + 1,
                             value=distance))
        return relations

    def generate_in_area(self, num_entities: int) -> None:
        '''
        It returns None, that indicates that the relation is not clear or one object exists
        '''
        return None

    def get_task(self, num_entities: int):
        """
        This method of selecting the task of the query using an exponential decay method. It first filters out
        tasks that are not viable due to the number of entities. It then selects the task based on a probability
        distribution that assigns higher probabilities to task at the beginning of the list. This allows for
        a drafting system that prioritises certain tasks over others, as e.g. "individual_distances" is a
        far more difficult task than "in_area" and therefore requires more training data.

        Example probability distribution with decay of 0.5:
            - 3 or more entities: [0.50648039 0.30719589 0.18632372]
            - 2 entities: [0.62245933 0.37754067]
            - 1 entity: [1.0]

        :param num_entities: The number of entities of the query
        :return: The selected task
        """
        viable_tasks = [RELATION_TASKS.INDIVIDUAL_DISTANCES.value, RELATION_TASKS.WITHIN_RADIUS.value,
                        RELATION_TASKS.IN_AREA.value]
        if num_entities < 3:
            viable_tasks.pop(0)
        if num_entities == 1:
            viable_tasks.pop(0)

        decay_rate = 0.5
        task_nums = np.arange(1, len(viable_tasks) + 1)
        probabilities = np.exp(-decay_rate * task_nums)
        probabilities /= np.sum(probabilities)
        selected_task = np.random.choice(viable_tasks, p=probabilities)

        return selected_task
      
    def run(self, num_entities: int) -> Relations:
        """
        This task runs the general pipeline for generating relations between entities.
        The specific task for relation generation is randomly selected.
        Once it is defined, it will execute the corresponding function.
        Args:
            num_entities (int): The number of entities involved in the task.

        Returns:
            List[Relation] or None: A list of Relation objects representing the task outcome.
        """
        # np.random.shuffle(self.tasks)
        # selected_task = self.tasks[0]
        selected_task = self.get_task(num_entities)

        if selected_task == RELATION_TASKS.INDIVIDUAL_DISTANCES.value:
            relations = Relations(type=selected_task,
                                  relations=self.generate_individual_distances(num_entities=num_entities))
        elif selected_task == RELATION_TASKS.WITHIN_RADIUS.value:  # Just search for all given objects in area, no distance required
            relations = Relations(type=selected_task,
                                  relations=self.generate_within_radius(num_entities=num_entities))
        elif selected_task == RELATION_TASKS.IN_AREA.value:
            relations = Relations(type=selected_task,
                                  relations=self.generate_in_area(num_entities=num_entities))

        return relations