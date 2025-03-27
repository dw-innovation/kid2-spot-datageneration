import copy
import re
from typing import Dict
from benchmarking.utils import find_pairs_semantic,are_dicts_equal
from sklearn.metrics.pairwise import cosine_similarity

class EntityAndPropertyAnalyzer:
    def __init__(self):
        pass

    def pair_objects(self, predicted_objs, reference_objs):
        '''
        Pair entities/properties based on their names
        :param predicted_objs:
        :param reference_objs:
        :return: paired entities, unpaired entities
        '''
        # create a dictionary of reference entities and predicted entities, names will be the keys
        reference_entities_mapping = {}
        predicted_entities_mapping = {}

        for reference_entity in reference_objs:
            reference_entities_mapping[reference_entity['name']] = reference_entity

        if not predicted_objs:
            return None, None, {'prediction': [], 'reference': list(reference_entities_mapping.keys())}

        for predicted_entity in predicted_objs:
            predicted_entities_mapping[predicted_entity['name']] = predicted_entity

        paired_entities, unpaired_entities = find_pairs_semantic(reference_list=list(reference_entities_mapping.keys()), prediction_list=list(predicted_entities_mapping.keys()))

        full_paired_entities = [] # list of tuples
        for (ground_truth_name, predicted_entity_name) in paired_entities:
            full_paired_entities.append(
                (reference_entities_mapping[ground_truth_name],predicted_entities_mapping[predicted_entity_name])
            )

        return full_paired_entities, paired_entities, unpaired_entities

    def compose_height_value(self, height):
        height_value = re.findall(r'\d+', height)[0]
        height_metric = height.replace(height_value,'')
        return height_value, height_metric

    def compare_entities(self, reference_entities, predicted_entities) -> Dict:
        """
        Check if two lists of entities are identical. The lists are first sorted via their names, to make sure the order
        does not affect the results.

        :param reference_entities: The first entity list to compare (ref_data).
        :param predicted_entities: The second entity list to compare (generated data).
        :return: Boolean whether the two entity lists are the same.
        """
        full_paired_entities, paired_entities, unpaired_entities = self.pair_objects(predicted_objs=predicted_entities, reference_objs=reference_entities)
        total_ref_entities = len(reference_entities)
        # total_predicted_entities = len(predicted_entities)
        num_entity_match_perfect = 0
        num_correct_entity_type = 0 # entity type check nrw/cluster
        total_properties = 0
        total_height_property = 0
        num_correct_cluster_distance = 0
        num_correct_cluster_points = 0
        num_correct_properties_perfect = 0
        num_correct_properties_weak = 0
        num_hallucinated_properties = 0
        num_missing_properties = 0
        num_correct_height_metric = 0
        num_correct_height_distance = 0

        num_entity_weak_match= len(paired_entities) if paired_entities else 0
        # hallucination check
        num_hallucinated_entity = len(unpaired_entities['prediction'])
        # missing entity check
        num_missing_entity = len(unpaired_entities['reference'])

        total_clusters = 0
        for ref_entity in reference_entities:
            if ref_entity['type'] == 'cluster':
                total_clusters+=1

            if 'properties' in ref_entity:
                ref_properties = ref_entity.get('properties')
                total_properties += len(ref_properties)

                for ref_property in ref_properties:
                    if 'height' == ref_property['name']:
                        total_height_property += 1

        if full_paired_entities:
            for (ref_ent, predicted_ent) in full_paired_entities:
                if are_dicts_equal(ref_ent, predicted_ent):
                    num_entity_match_perfect+=1
                if ref_ent['type'] == 'cluster':
                    ref_min_points = ref_ent.get('minPoints')
                    predicted_min_points = predicted_ent.get('minpoints')

                    if ref_min_points == predicted_min_points:
                        num_correct_cluster_points+=1

                    ref_max_distance = ref_ent.get('maxDistance')
                    predicted_max_distance = predicted_ent.get('maxdistance')

                    if ref_max_distance == predicted_max_distance:
                        num_correct_cluster_distance+=1

                if 'properties' in ref_ent:
                    ref_properties = ref_ent.get('properties')
                    ent_properties = predicted_ent.get('properties', None)

                    full_paired_props, paired_props, unpaired_props = self.pair_objects(
                        predicted_objs=ent_properties, reference_objs=ref_properties)

                    if not full_paired_props:
                        num_missing_properties += len(unpaired_props['reference'])
                    else:
                        for (ref_prop, ent_prop) in full_paired_props:
                            if are_dicts_equal(ref_prop, ent_prop):
                                num_correct_properties_perfect += 1

                            if 'height' == ref_prop['name']:
                                ref_height_value, ref_height_metric = self.compose_height_value(ref_prop['value'])
                                pred_height_value, pred_height_metric = self.compose_height_value(ent_prop['value'])

                                if ref_height_value == pred_height_value:
                                    num_correct_height_distance+=1
                                if ref_height_metric == pred_height_metric:
                                    num_correct_height_metric+=1


                    # hallucinated prop
                    num_hallucinated_properties += len(unpaired_props['prediction'])
                    # missing prop
                    num_missing_properties += len(unpaired_props['reference'])

                    if paired_props:
                        num_correct_properties_weak+=len(paired_props)

                if ref_ent['type'] == predicted_ent['type']:
                    num_correct_entity_type+=1

        else:
            num_missing_properties = total_properties
            num_missing_entity += total_ref_entities

        return dict(
            total_clusters=total_clusters,
            total_properties=total_properties,
            total_ref_entities = total_ref_entities,
            num_entity_match_perfect = num_entity_match_perfect,
            num_entity_match_weak = num_entity_weak_match,
            num_correct_entity_type = num_correct_entity_type,
            num_correct_cluster_distance = num_correct_cluster_distance,
            num_correct_cluster_points = num_correct_cluster_points,
            num_correct_properties_perfect = num_correct_properties_perfect,
            num_correct_properties_weak=num_correct_properties_weak,
            total_height_property = total_height_property,
            num_hallucinated_properties = num_hallucinated_properties,
            num_hallucinated_entity = num_hallucinated_entity,
            num_missing_properties=num_missing_properties,
            num_missing_entity=num_missing_entity,
            num_correct_height_metric = num_correct_height_metric,
            num_correct_height_distance = num_correct_height_distance
        )

    def sort_entities(self, entities1, entities2):
        entities1_sorted = sorted(entities1, key=lambda x: x['name'].lower())
        entities2_sorted = sorted(entities2, key=lambda x: x['name'].lower())
        return entities1_sorted, entities2_sorted