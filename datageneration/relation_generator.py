import numpy as np
from enum import Enum
from typing import List

from datageneration.data_model import Relation, Relations, Entity
from datageneration.property_generator import get_random_decimal_with_metric


class RELATION_TASKS(Enum):
    INDIVIDUAL_DISTANCES = 'individual_distances'
    WITHIN_RADIUS = 'within_radius'
    IN_AREA = 'in_area'


class RelationGenerator:
    def __init__(self, max_distance_digits: int, prop_generating_contain_rel: float=0.7):
        self.MAX_DISTANCE_DIGITS = max_distance_digits
        self.prop_generating_contain_rel = [prop_generating_contain_rel, 1-prop_generating_contain_rel]
        self.tasks = [relation_task.value for relation_task in RELATION_TASKS]

    def generate_individual_distances(self, num_entities: int) -> List[Relation]:
        relations = []
        for t_no in range(num_entities):
            if t_no != num_entities - 1:
                relations.append(
                    Relation(type='dist', source=t_no, target=t_no + 1,
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
                    Relation(type='dist', source=0, target=t_no + 1,
                             value=distance))
        return relations

    def generate_in_area(self, num_entities: int) -> None:
        '''
        It returns None, that indicates that the relation is not clear or one object exists
        '''
        return None

    def generate_relation_with_contain(self, entities: List[Entity]) -> Relations:
        area_entities = []
        point_entities = []

        for entity in entities:
            if entity.is_area:
                area_entities.append(entity)
            else:
                point_entities.append(entity)

        # todo: it assumes we will have only one area entity but there might be more.
        area_entity = np.random.choice(area_entities, 1)[0]
        remaining_area_entities = list(filter(lambda x: x in area_entity, area_entities))

        if len(remaining_area_entities) > 0:
            point_entities.extend(remaining_area_entities)

        # randomly select entities they will be in a contain relation
        arr_num_of_point_entities_connecting_to_area_entity = np.arange(1, len(point_entities) + 1)
        num_of_point_entities_connecting_to_area_entity = \
            np.random.choice(arr_num_of_point_entities_connecting_to_area_entity, 1)[0]

        point_entities_connecting_to_area_entity = point_entities[:num_of_point_entities_connecting_to_area_entity]
        other_point_entities = point_entities[len(point_entities_connecting_to_area_entity):len(point_entities)]

        # these are must rule, helper for unittest
        assert len(point_entities_connecting_to_area_entity) == num_of_point_entities_connecting_to_area_entity
        assert len(other_point_entities) == len(point_entities) - num_of_point_entities_connecting_to_area_entity
        assert point_entities_connecting_to_area_entity != other_point_entities

        relations = self.generate_relation_with_contain_helper(area_entity, other_point_entities,
                                                               point_entities_connecting_to_area_entity)

        return Relations(type='relation_with_contain', relations=relations)

    def generate_relation_with_contain_helper(self, area_entity: Entity, other_point_entities: List[Entity],
                                              point_entities_connecting_to_area_entity: List[Entity]) -> List[Relation]:
        relations = []
        # contains relations
        for point_entity in point_entities_connecting_to_area_entity:
            relations.append(
                Relation(type='contains', source=area_entity.id, target=point_entity.id, value=None))
        # individual relations
        for point_entity in other_point_entities:
            value = get_random_decimal_with_metric(self.MAX_DISTANCE_DIGITS)
            relations.append(Relation(type='dist', source=area_entity.id, target=point_entity.id, value=value))
            for connected_point_entity in point_entities_connecting_to_area_entity:
                relations.append(
                    Relation(type='dist', source=connected_point_entity.id, target=point_entity.id, value=value))

        return relations

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

    def run(self, entities: List[Entity]) -> Relations:
        """
        This task runs the general pipeline for generating relations between entities.
        The specific task for relation generation is randomly selected.
        Once it is defined, it will execute the corresponding function.
        Args:
            entities (List): The entities involved in the task.

        Returns:
            List[Relation] or None: A list of Relation objects representing the task outcome.
        """
        num_entities = len(entities)
        contain_area_entity = any(entity.is_area for entity in entities)

        if not contain_area_entity or (num_entities == 1 and contain_area_entity):
            relations = self.standard_rel_tasks(num_entities)
        else:
            generating_contain_rel = np.random.choice([True, False], p=self.prop_generating_contain_rel)
            if generating_contain_rel:
                relations = self.generate_relation_with_contain(entities)
            else:
                relations = self.standard_rel_tasks(num_entities)
        return relations

    def standard_rel_tasks(self, num_entities):
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
