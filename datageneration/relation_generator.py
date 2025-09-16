import numpy as np
from enum import Enum
from typing import List
import copy
import random

from datageneration.data_model import Relation, Relations, Entity
from datageneration.utils import get_random_decimal_with_metric

"""
Relation generation utilities for synthetic spatial tasks.

This module generates different kinds of relations between entities for tasks such as:
- individual pairwise distances between entities,
- "within radius" relations anchored at a source entity,
- "in area" containment-style relations,
- compound relations mixing "contains" with individual distances.

Key types:
- `Entity`: basic object with an `id` and a boolean `is_area`.
- `Relation`: atomic relation with `type`, `source`, `target`, and optional `value`.
- `Relations`: a container with a global `type` and a list of `Relation` items.

Notes
-----
- Distances are produced by `get_random_decimal_with_metric(max_digits)` and returned as a
  decimal-plus-unit data structure (passed directly into `Relation.value` unless stringified
  elsewhere in downstream code).
- The "contains" relation uses `type='contains'` and does not carry a numeric value.
"""

RELATION_TYPE='distance'

class RELATION_TASKS(Enum):
    """
    Enum for the supported relation-generation tasks.
    """
    INDIVIDUAL_DISTANCES = 'individual_distances'
    WITHIN_RADIUS = 'within_radius'
    IN_AREA = 'in_area'

class RelationGenerator:
    """
    Generate relation structures (`Relations`) for a set of entities.

    Parameters
    ----------
    max_distance_digits : int
        Maximum number of digits for generated distance magnitudes.
    prob_generating_contain_rel : float
        Probability in [0, 1] of generating a "contains"-based scenario when possible.

    Attributes
    ----------
    MAX_DISTANCE_DIGITS : int
        Stored upper bound for distance magnitude digits.
    prob_generating_contain_rel : float
        Stored probability for branching into "contains" generation.
    tasks : List[str]
        Available task names taken from RELATION_TASKS values.
    """
    def __init__(self, max_distance_digits: int, prob_generating_contain_rel: float):
        self.MAX_DISTANCE_DIGITS = max_distance_digits
        self.prob_generating_contain_rel = prob_generating_contain_rel
        self.tasks = [relation_task.value for relation_task in RELATION_TASKS]

    def generate_individual_distances(self, entity_ids: List[int]) -> List[Relation]:
        """
        Generate chained pairwise distance relations over a list of entity IDs.

        Produces distances (entity[i] -> entity[i+1]) for i in [0, n-2].

        Parameters
        ----------
        entity_ids : List[int]
            Ordered IDs to connect with distances.

        Returns
        -------
        List[Relation]
            Distance relations with `type='distance'` and `value` set to a decimal-with-metric.
        """
        # np.random.shuffle(entity_ids)
        relations = []
        for t_no in range(len(entity_ids)-1):
            relations.append(
                Relation(type=RELATION_TYPE, source=entity_ids[t_no], target=entity_ids[t_no+1],
                         value=get_random_decimal_with_metric(self.MAX_DISTANCE_DIGITS)))
        return relations

    def generate_within_radius(self, num_entities: int) -> List[Relation]:
        """
        Generate a star-shaped set of 'within radius' distance relations.

        Uses entity 0 as the anchor (source) and connects it to all others with a shared radius value.

        Parameters
        ----------
        num_entities : int
            Number of entities involved (IDs are assumed 0..num_entities-1).

        Returns
        -------
        List[Relation]
            Relations of `type='distance'` from source 0 to each target, all sharing the same `value`.
        """
        relations = []
        distance = get_random_decimal_with_metric(self.MAX_DISTANCE_DIGITS)
        for t_no in range(num_entities):
            if t_no != num_entities - 1:
                relations.append(
                    Relation(type=RELATION_TYPE, source=0, target=t_no + 1,
                             value=distance))
        return relations

    def generate_in_area(self, num_entities: int) -> None:
        """
        Generate an 'in area' relation placeholder.

        Returns
        -------
        None
            Indicates that no explicit relation set is produced (e.g., ambiguous or singleton case).
        """
        return None

    def generate_relation_with_contain(self, area_entities: List[Entity], point_entities: List[Entity],
                                        max_within_combs: int) -> Relations:
        """
        Generate relations including at least one 'contains' group.

        Workflow
        --------
        1) Randomly choose a number of (area → contained points) groups.
        2) For each group, pick one area and 1..k points it contains.
        3) Compute any additional "other entities" to allow individual distances between
           a representative of one group and those others.
        4) Return either:
           - type='individual_distances_with_contains' (contains + extra distances), or
           - type='contains_relation' (only contains relations).

        Parameters
        ----------
        area_entities : List[Entity]
            Entities flagged as areas (i.e., `is_area=True`).
        point_entities : List[Entity]
            Entities flagged as points (`is_area=False`).
        max_within_combs : int
            Upper bound for the number of area→points containment groups.

        Returns
        -------
        Relations
            A `Relations` object with `type` and a list of `Relation` items.
        """
        num_within_combs = np.random.choice(np.arange(1,max_within_combs+1))
        remaining_area_entities = copy.deepcopy(area_entities)
        remaining_point_entities = copy.deepcopy(point_entities)
        remaining_num_possible_connections = len(point_entities)
        drawn_area_entities = []
        point_entities_connecting_to_area_entity = []

        for area_num in range(1,num_within_combs+1):
            # For each contains relation, draw one are entity and filter it from the "remaining_areas" list
            area_entity = np.random.choice(remaining_area_entities)
            remaining_area_entities = [e for e in remaining_area_entities if e != area_entity]
            drawn_area_entities.append(area_entity)

            # Randomly select one or multiple entities that will be contained in this area, leave enough behind for
            # all other "contains" areas in this query, filter drawn entities from "remaining_entities" list
            num_of_point_entities_connecting_to_area_entity = \
                np.random.choice(np.arange(1,remaining_num_possible_connections-num_within_combs+area_num+1))
            remaining_num_possible_connections -= num_of_point_entities_connecting_to_area_entity
            np.random.shuffle(remaining_point_entities)
            point_entities_connecting_to_area_entity.append(
                remaining_point_entities[:num_of_point_entities_connecting_to_area_entity])
            for point_entity_connecting_to_area_entity in point_entities_connecting_to_area_entity:
                remaining_point_entities = [e for e in remaining_point_entities if e not in
                                            point_entity_connecting_to_area_entity]

            # these are must rule, helper for unittest
            assert len(point_entities_connecting_to_area_entity[-1]) == num_of_point_entities_connecting_to_area_entity
            # assert len(remaining_point_entities) == len(point_entities) - remaining_num_possible_connections
            assert point_entities_connecting_to_area_entity[-1] != remaining_point_entities

        # "Other entities" are all not in "contains relations"
        other_entities = [*remaining_point_entities, *remaining_area_entities]
        # Extend other drawn entities to the first point entity of each "contains" group, as they are used to show
        # individual distances between this group and other entities
        n = len(drawn_area_entities)
        x = random.randint(0, n - 1)
        # Randomly decide whether to pick from the area or the point list
        if random.choice([True, False]):
            other_entities.append(drawn_area_entities[x])
        else:
            sub_list = point_entities_connecting_to_area_entity[x]
            other_entities.append(random.choice(sub_list))

        #other_entities.extend([e[0] for e in point_entities_connecting_to_area_entity])
        other_entity_ids = [e.id for e in other_entities]

        assert len(drawn_area_entities) == len(point_entities_connecting_to_area_entity)

        # Check if only one "contains" group is present, otherwise use "individual_distances_with_contains"
        if len(other_entity_ids) > 1:
            relations = self.generate_relation_with_contain_helper(drawn_area_entities,
                                               point_entities_connecting_to_area_entity)
            relations.extend(self.generate_individual_distances(other_entity_ids))
            relation_type = "individual_distances_with_contains"
        else:
            relations = self.generate_relation_with_contain_helper(drawn_area_entities,
                                               point_entities_connecting_to_area_entity)
            relation_type = "contains_relation"

        return Relations(type=relation_type, relations=relations)

    def generate_relation_with_contain_helper(self, drawn_area_entities: List[Entity],
             point_entities_connecting_to_area_entity: List[List[Entity]]) -> List[Relation]:
        """
        Build atomic 'contains' relations for each (area → list of points) group.

        Parameters
        ----------
        drawn_area_entities : List[Entity]
            The chosen area entities, one per group.
        point_entities_connecting_to_area_entity : List[List[Entity]]
            For each area, the corresponding list of contained point entities.

        Returns
        -------
        List[Relation]
            A flat list of `Relation(type='contains', source=area.id, target=point.id)`.
        """
        relations = []
        for aid, area in enumerate(drawn_area_entities):
            for point in point_entities_connecting_to_area_entity[aid]:
                    relations.append(
                        Relation(type='contains', source=area.id, target=point.id))

        return relations

    def get_task(self, num_entities: int) -> str:
        """
        Select a relation-generation task using an exponential-decay probability over viable tasks.

        Viability rules
        ---------------
        - < 3 entities: remove 'individual_distances'
        - == 1 entity: remove 'within_radius' (leaving 'in_area')

        Probability
        -----------
        p(i) ∝ exp(-decay_rate * i) over the ordered viable task list.

        Parameters
        ----------
        num_entities : int
            Number of entities in the current scenario.

        Returns
        -------
        str
            Selected task name from `RELATION_TASKS`.
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
        Execute the full pipeline: optionally create 'contains' relations, else a standard task.

        Parameters
        ----------
        entities : List[Entity]
            All entities in the scene (areas and points).

        Returns
        -------
        Relations
            A `Relations` object where:
            - `type` is one of {'individual_distances_with_contains', 'contains_relation',
              'individual_distances', 'within_radius', 'in_area'}
            - `relations` is a list of `Relation` or `None` (for 'in_area').
        """
        area_entities = []
        point_entities = []
        for id, entity in enumerate(entities):
            if entity.is_area:
                area_entities.append(entity)
            else:
                point_entities.append(entity)
        max_within_combs = min(len(area_entities), len(point_entities))

        generating_contain_rel = np.random.choice([True, False], p=[self.prob_generating_contain_rel,
                                                                    1 - self.prob_generating_contain_rel])
        if generating_contain_rel and max_within_combs>0:
            relations = self.generate_relation_with_contain(area_entities, point_entities, max_within_combs)
        else:
            relations = self.standard_rel_tasks(np.arange(len(entities)))
        return relations

    def standard_rel_tasks(self, entity_ids) -> Relations:
        """
        Dispatch to a standard task (no 'contains' logic).

        Parameters
        ----------
        entity_ids : iterable of int
            Ordered entity IDs to use for relation construction.

        Returns
        -------
        Relations
            Container with the selected task type and generated relations (or None for 'in_area').
        """
        num_entities = len(entity_ids)
        selected_task = self.get_task(num_entities)
        if selected_task == RELATION_TASKS.INDIVIDUAL_DISTANCES.value:
            relations = Relations(type=selected_task,
                                  relations=self.generate_individual_distances(entity_ids=entity_ids))
        elif selected_task == RELATION_TASKS.WITHIN_RADIUS.value:  # Just search for all given objects in area, no distance required
            relations = Relations(type=selected_task,
                                  relations=self.generate_within_radius(num_entities=num_entities))
        elif selected_task == RELATION_TASKS.IN_AREA.value:
            relations = Relations(type=selected_task,
                                  relations=self.generate_in_area(num_entities=num_entities))
        return relations