import copy
from typing import Dict
from benchmarking.utils import find_pairs_semantic
from sklearn.metrics.pairwise import cosine_similarity

class PropertyAnalyzer:
    def __init__(self):
        pass

    def convert_values_to_string(self, data):
        for item in data:
            item["name"] = item["name"].lower()
            if 'value' not in item:
                continue
            if isinstance(item['value'], (int, float)):
                item['value'] = str(item['value'])
            else:
                item['value'] = item['value'].lower()

        return data

    def compare_properties(self, props1, props2) -> int:
        """
        Check if two lists of properties are identical. The lists are first sorted via their names, to make sure the order
        does not affect the results.

        :param props1: The first property list to compare.
        :param props2: The second property list to compare.
        :return: Boolean whether the two property lists are the same.
        """
        matches = 0
        props1 = self.convert_values_to_string(props1)
        props2_copy = self.convert_values_to_string(copy.deepcopy(props2))
        for p1 in props1:
            for id, p2 in enumerate(props2_copy):
                if p1 == p2:
                    props2_copy.pop(id)
                    matches += 1
                    break

        return matches

    def percentage_properties_same(self, ref_entities, prop_entities) -> float:
        total_props = 0
        correctly_identified_properties = 0
        for ent, props in ref_entities.items():

            if ent not in prop_entities:
                total_props += len(props)
                continue
            else:
                total_props += max(len(props), len(prop_entities[ent]))

            correctly_identified_properties += self.compare_properties(props1=props, props2=prop_entities[ent])

        if total_props > 0:
            return correctly_identified_properties / total_props
        else:
            return -1.0


class EntityAnalyzer:
    def __init__(self, property_analyzer: PropertyAnalyzer):
        self.property_analyzer = property_analyzer

    def pair_entities(self, predicted_entities, reference_entities):
        '''
        Pair entities based on their names
        :param predicted_entities:
        :param reference_entities:
        :return: paired entities, unpaired entities
        '''
        # create a dictionary of reference entities and predicted entities, names will be the keys
        reference_entities_mapping = {}
        predicted_entities_mapping = {}

        for reference_entity in reference_entities:
            reference_entities_mapping[reference_entity['name']] = reference_entity

        for predicted_entity in predicted_entities:
            predicted_entities_mapping[predicted_entity['name']] = predicted_entity

        paired_entities, unpaired_entities = find_pairs_semantic(reference_list=list(reference_entities_mapping.keys()), prediction_list=list(predicted_entities_mapping.keys()))

        full_paired_entities = [] # list of tuples
        for (ground_truth_name, predicted_entity_name) in paired_entities:
            full_paired_entities.append(
                (reference_entities_mapping[ground_truth_name],predicted_entities_mapping[predicted_entity_name])
            )

        return full_paired_entities, paired_entities, unpaired_entities

    def compare_entities(self, reference_entities, predicted_entities, compare_props=True) -> Dict:
        """
        Check if two lists of entities are identical. The lists are first sorted via their names, to make sure the order
        does not affect the results.

        :param reference_entities: The first entity list to compare (ref_data).
        :param predicted_entities: The second entity list to compare (generated data).
        :return: Boolean whether the two entity lists are the same.
        """
        num_matched_entity_wo_property = 0
        num_hallucinated_entity = 0
        num_missing_entity = 0
        num_correct_entity_type = 0

        full_paired_entities, paired_entities, unpaired_entities = self.pair_entities(predicted_entities=predicted_entities, reference_entities=reference_entities)
        num_reference_entities = len(reference_entities)
        percentage_num_matched_entity_wo_property = len(paired_entities) / num_reference_entities

        # todo: implement property check
        percentage_num_perfect_matched_entity = 0


        # entity type check nrw/cluster
        percentage_correct_entity_type 

        # hallucination check
        num_hallucinated_entity = len(unpaired_entities['prediction'])

        # missing entity check
        num_missing_entity = len(unpaired_entities['reference'])



        # percentage_num_perfect_matched_entity = percentage_num_perfect_matched_entity,
        # percentage_num_matched_wo_property = percentage_num_matched_entity_wo_property,
        # num_hallucinated_entity = num_hallucinated_entity,
        # num_missing_entity = num_missing_entity,
        # num_correct_entity_type = num_correct_entity_type


        # total_ents = max(len(reference_entities), len(predicted_entities))
        # matches = 0
        # predicted_entities_copy = copy.deepcopy(predicted_entities)
        # for reference_entity in reference_entities:
        #     print('reference entity')
        #     print(reference_entity)
        #     reference_entity_type = reference_entity.get('type')
        #     for id, predicted_entity in enumerate(predicted_entities_copy):
        #         entity_type = predicted_entity.get('type', None)
        #         print('predicted entity')
        #         print(predicted_entity)
        #
        #         if entity_type == reference_entity_type:
        #             num_correct_entity_type +=1
        #
        #             print(num_correct_entity_type)
        #
        #         if 'name' not in predicted_entity:
        #             break
        #         if isinstance(predicted_entity['name'], list):
        #             predicted_entity['name'] = predicted_entity['name'][0]
        #         if reference_entity['name'].lower() == predicted_entity['name'].lower() and reference_entity['type'] == entity_type:
        #             if compare_props and 'properties' in reference_entity:
        #                 prop_matches = self.property_analyzer.compare_properties(reference_entity.get('properties', []),
        #                                                                                   predicted_entity.get('properties', []))
        #                 percentage_properties_same = prop_matches / len(reference_entity.get('properties', []))
        #                 if percentage_properties_same in [1.0, -1.0]:
        #                     predicted_entities_copy.pop(id)
        #                     matches += 1
        #                     break
        #             else:
        #                 predicted_entities_copy.pop(id)
        #                 matches += 1
        #                 break
        # print('total ents')
        # print(total_ents)
        #
        # print('num correct entity type')
        # print(num_correct_entity_type)
        #
        # print('percentage of correct ents')
        # print(num_correct_entity_type / total_ents)

        return dict(
                percentage_num_perfect_matched_entity = percentage_num_perfect_matched_entity,
                percentage_num_matched_wo_property = percentage_num_matched_entity_wo_property,
                num_hallucinated_entity = num_hallucinated_entity,
                num_missing_entity = num_missing_entity,
                num_correct_entity_type = num_correct_entity_type
        )

    def sort_entities(self, entities1, entities2):
        entities1_sorted = sorted(entities1, key=lambda x: x['name'].lower())
        entities2_sorted = sorted(entities2, key=lambda x: x['name'].lower())
        return entities1_sorted, entities2_sorted