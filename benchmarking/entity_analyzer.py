import copy
import re
from thefuzz import fuzz
from autocorrect import Speller
from typing import Dict
from benchmarking.utils import find_pairs_semantic, are_dicts_equal, DIST_LOOKUP, normalize, compose_metric
from sklearn.metrics.pairwise import cosine_similarity

spell = Speller()

class EntityAndPropertyAnalyzer:
    def __init__(self, descriptors):
        self.descriptors = descriptors
        self.color_descriptors = self.descriptors['color']
        self.color_descriptors.sort()

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

    def check_equivalent_entities(self, descriptors, ref, gen):
        """
        DEPRECEATED -- integrated into the pair_objects
        In case the reference and the generated entities + properties have descriptors that differ, but come from the
        same bundle, this script replaces the generated entity/property descriptor with that of the reference to ensure it
        will be treated as equal for the rest of the script.

        :param descriptors: A map of descriptors where each descriptor maps to the corresponding bundle descriptor list.
        :param ref: The reference entity from the ground truth data.
        :param gen: The generated entity to be evaluated.
        :return: gen_copy - The copy with the corrected entity values.
        """
        gen_copy = copy.deepcopy(gen)
        for r in ref:
            for id, g in enumerate(gen_copy):
                if 'name' not in g:
                    break
                if isinstance(g['name'], list):
                    g['name'] = g['name'][0]

                if r['name'] in descriptors:
                    equivalent_descriptors = descriptors.get(r['name'])
                else:
                    continue
                if g['name']:
                    if g['name'] in equivalent_descriptors:
                        g['name'] = r['name']

                if 'properties' in r and 'properties' in g:
                    props_r = self.convert_values_to_string(r.get('properties', []))
                    props_g = self.convert_values_to_string(g.get('properties', []))
                    for pr in props_r:
                        if pr['name'] not in descriptors:
                            continue
                        equivalent_properties = descriptors.get(pr['name'])
                        for id, pg in enumerate(props_g):
                            if pg['name'] in equivalent_properties:
                                props_g[id]['name'] = pr['name']

        return gen_copy

    def normalization_with_descriptors(self, obj_name):
        obj_name = obj_name.lower()
        if obj_name not in self.descriptors:
            normalized_name = self.fuzzy_search(obj_name)
            if obj_name in normalized_name:
                normalized_name = obj_name
            else:
                corrected_name = spell(obj_name)
                normalized_name = self.fuzzy_search(corrected_name)
        else:
            normalized_name = self.descriptors[obj_name]
        return normalized_name

    def fuzzy_search(self, corrected_name):
        best_match = None
        highest_score = 0
        for descriptor in self.descriptors:
            score = fuzz.token_set_ratio(descriptor.lower(), corrected_name)
            if score > highest_score and score > 80:
                highest_score = score
                # we do this, because station is under multiple descriptors, and lead wrong pairing!!!
                if ' ' in corrected_name and descriptor=='station':
                    continue
                best_match = descriptor


        if best_match:
            normalized_name = self.descriptors[best_match][0]
        else:
            normalized_name = corrected_name
        return normalized_name

    def pair_objects(self, predicted_objs, reference_objs):
        '''
        Pair entities/properties based on their names
        :param predicted_objs:
        :param reference_objs:
        :return: paired entities, unpaired entities
        '''
        # create a dictionary of reference entities and predicted entities, names will be the keys
        print('===predicted objs===')
        print(predicted_objs)

        print('===reference objs===')
        print(reference_objs)

        reference_obj_mapping = {}
        predicted_obj_mapping = {}
        duplicate_references = {}
        for reference_obj in reference_objs:
            reference_obj['name'] = reference_obj['name']
            normalized_name = reference_obj['name']
            if 'color' in normalized_name or 'colour' in normalized_name:
                normalized_name = 'color'
            if 'brand' in normalized_name:
                normalized_name = normalized_name.replace('brand:', '')
            else:
                normalized_name = self.normalization_with_descriptors(normalized_name)
            if isinstance(normalized_name, list):
                normalized_name = normalized_name[0]
            if 'value' in reference_obj:
                if isinstance(reference_obj['value'], str):
                    reference_obj['value'] = reference_obj['value'].lower()
            if normalized_name in reference_obj_mapping:
                print('==normalized name==')
                print(normalized_name)
                if normalized_name not in duplicate_references:
                    old_ref_obj = reference_obj_mapping[normalized_name]
                    if 'value' in old_ref_obj:
                        # this is for props
                        duplicate_references[normalized_name] = [old_ref_obj['value']]
                        duplicate_references[normalized_name].append(reference_obj['value'])
                    else:
                        duplicate_references[normalized_name] = [old_ref_obj]
                        duplicate_references[normalized_name].append(reference_obj)
            else:
                reference_obj_mapping[normalized_name] = reference_obj
                reference_obj['normalized_name'] = normalized_name


        print('===duplicate references')
        print(duplicate_references)

        if not predicted_objs:
            return None, None, {'prediction': [], 'reference': list(reference_obj_mapping.keys())}

        duplicate_predictions = []
        for predicted_obj in predicted_objs:
            if isinstance(predicted_obj['name'], int):
                predicted_obj['name'] = str(predicted_obj['name'])
            normalized_name = predicted_obj['name']
            if 'color' in normalized_name or 'colour' in normalized_name:
                normalized_name = 'color'
            if 'brand' in normalized_name:
                normalized_name = normalized_name.replace('brand:', '')
            else:
                normalized_name = self.normalization_with_descriptors(normalized_name)
            if isinstance(normalized_name, list):
                normalized_name = normalized_name[0]
            if 'value' in predicted_obj:
                if isinstance(predicted_obj['value'], str):
                    predicted_obj['value'] = predicted_obj['value'].lower()

            if normalized_name in predicted_obj_mapping:
                # check if we have duplicate props
                if normalized_name in reference_obj_mapping and 'value' in reference_obj_mapping[normalized_name]:
                    if reference_obj_mapping[normalized_name]['value'] == predicted_obj['value']:
                       old_predicted_obj = predicted_obj_mapping[normalized_name]
                       duplicate_predictions.append(old_predicted_obj)
                predicted_obj_mapping[normalized_name] = predicted_obj
                predicted_obj['normalized_name'] = normalized_name
                duplicate_predictions.append(predicted_obj)
            else:
                predicted_obj_mapping[normalized_name] = predicted_obj
                predicted_obj['normalized_name'] = normalized_name

        paired_objs, unpaired_objs = find_pairs_semantic(reference_list=list(reference_obj_mapping.keys()), prediction_list=list(predicted_obj_mapping.keys()))
        print('==duplicate predictions==')
        print(duplicate_predictions)
        for duplicate_prediction in duplicate_predictions:
            print('==duplicate prediction==')
            print(duplicate_prediction)
            unpaired_objs['prediction'].append(duplicate_prediction['normalized_name'])

        print('===unpaired objs===')
        print(unpaired_objs)

        full_paired_entities = [] # list of tuples
        for (ground_truth_name, predicted_obj_name) in paired_objs:
            pred_obj = predicted_obj_mapping[predicted_obj_name]
            if ground_truth_name in duplicate_references:
                duplicate_ref_values = duplicate_references[ground_truth_name]
                for duplicate_ref_value in duplicate_ref_values:
                    if 'value' in pred_obj:
                        if duplicate_ref_value == pred_obj['value']:
                            reference_obj_mapping[ground_truth_name]['value'] = duplicate_ref_value
                        else:
                            unpaired_objs['reference'].append(ground_truth_name)
                    else:
                        unpaired_objs['reference'].append(ground_truth_name)
            full_paired_entities.append(
                (reference_obj_mapping[ground_truth_name],pred_obj)
            )

        full_unpaired_objs = {
            'reference': [],
            'prediction': []
        }

        for reference_obj_name in unpaired_objs['reference']:
            full_unpaired_objs['reference'].append(reference_obj_mapping[reference_obj_name])

        print('predicted obj mapping')
        print(predicted_obj_mapping)

        for pred_obj_name in unpaired_objs['prediction']:
            print(pred_obj_name)
            full_unpaired_objs['prediction'].append(predicted_obj_mapping[pred_obj_name])

        return full_paired_entities, paired_objs, full_unpaired_objs, unpaired_objs

    def compose_height_value(self, height):
        heigh_splits = height.split(' ')
        dist = heigh_splits[0]
        metric = heigh_splits[1]
        return dist, metric

    def compare_entities(self, reference_entities, predicted_entities) -> Dict:
        """
        Check if two lists of entities are identical. The lists are first sorted via their names, to make sure the order
        does not affect the results.

        :param reference_entities: The first entity list to compare (ref_data).
        :param predicted_entities: The second entity list to compare (generated data).
        :return: Boolean whether the two entity lists are the same.
        """
        full_paired_entities, paired_entities, full_unpaired_entities, unpaired_entities = self.pair_objects(predicted_objs=predicted_entities, reference_objs=reference_entities)
        total_ref_entities = len(reference_entities)
        num_entity_match_perfect = 0
        num_entity_match_weak = 0
        num_correct_entity_type = 0 # entity type check nrw/cluster
        total_properties = 0
        total_height_property = 0
        total_cuisine_property = 0
        total_color_property= 0
        num_correct_cluster_distance = 0
        num_correct_cluster_points = 0
        num_correct_properties_perfect = 0
        # num_correct_properties_weak = 0
        num_hallucinated_properties = 0
        num_missing_properties = 0
        num_correct_height_metric = 0
        num_correct_height_distance = 0
        num_correct_height = 0
        num_correct_cuisine_properties = 0
        num_correct_color = 0
        entity_perfect_result = False
        props_perfect_result = False

        # hallucination check
        num_hallucinated_entity = len(unpaired_entities['prediction'])

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

                    if 'cuisine' == ref_property['name']:
                        total_cuisine_property += 1

                    if 'color' in ref_property['name'] or 'colour' in ref_property['name'] or ref_property['name'] in self.color_descriptors:
                        total_color_property += 1

        for unpaired_ent in full_unpaired_entities['reference']:
            if 'properties' in unpaired_ent:
                num_missing_properties+=1

        for unpaired_ent in full_unpaired_entities['prediction']:
            if 'properties' in unpaired_ent:
                num_hallucinated_properties+=1

        if full_paired_entities:
            print('full paired entities')
            print(full_paired_entities)
            for (ref_ent, predicted_ent) in full_paired_entities:
                if are_dicts_equal(ref_ent, predicted_ent):
                    num_entity_match_perfect+=1
                if ref_ent['type'] == 'cluster':
                    num_entity_match_weak+=1
                    ref_min_points = str(ref_ent.get('minPoints'))
                    predicted_min_points = str(predicted_ent.get('minpoints'))

                    if ref_min_points == predicted_min_points:
                        num_correct_cluster_points+=1

                    ref_max_distance = str(ref_ent.get('maxDistance'))
                    predicted_max_distance = str(predicted_ent.get('maxdistance'))

                    if ref_max_distance == predicted_max_distance:
                        num_correct_cluster_distance+=1

                if ref_ent['type'] == 'nwr' or ref_ent['type']=='cluster':
                    if ref_ent['type'] == predicted_ent['type']:
                        num_correct_entity_type+=1
                        num_entity_match_weak += 1

                if 'properties' in ref_ent:
                    ref_properties = ref_ent.get('properties')
                    ent_properties = predicted_ent.get('properties', None)

                    if not ent_properties:
                        num_missing_properties += len(ref_properties)
                        continue

                    full_paired_props, paired_props, full_unpaired_props, unpaired_props = self.pair_objects(
                        predicted_objs=ent_properties, reference_objs=ref_properties)
                    if not full_paired_props:
                        num_missing_properties += len(unpaired_props['reference'])
                    else:
                        for (ref_prop, ent_prop) in full_paired_props:
                            if are_dicts_equal(ref_prop, ent_prop):
                                num_correct_properties_perfect += 1
                            if 'height' == ref_prop['name']:
                                # Ipek: I commented here, because they are handled in are_dicts_equal
                                # ref_prop['value'] = str(ref_prop['value'])
                                # ent_prop['value'] = ent_prop.get('value', None)
                                # if ent_prop['value']:
                                #     ent_prop['value'] = str(ent_prop['value'])
                                if ref_prop['value'] == ent_prop['value']:
                                    num_correct_height+=1
                                ref_height_value, ref_height_metric = self.compose_height_value(ref_prop['value'])
                                pred_height_value, pred_height_metric = self.compose_height_value(ent_prop['value'])
                                if ref_height_value == pred_height_value:
                                    num_correct_height_distance+=1
                                if ref_height_metric == pred_height_metric:
                                    num_correct_height_metric+=1
                                # Ipek: I commented here, because they are handled in are_dicts_equal
                                # else:
                                #     pred_height_metric = DIST_LOOKUP.get(pred_height_metric, None)
                                #     if ref_height_metric == pred_height_metric:
                                #         num_correct_height_metric += 1
                            if 'cuisine' == ref_prop['name']:
                                if 'value' not in ent_prop:
                                    print(f'Mismatch between the props: {ref_prop} and {ent_prop}')
                                else:
                                    if ref_prop['value'] == ent_prop['value']:
                                        num_correct_cuisine_properties += 1

                            if 'color' in ent_prop['name'] or 'colour' in ent_prop['name']:
                                ent_prop_value = ent_prop.get('value', None)
                                ref_prop_value = ent_prop.get('value', None)
                                if ref_prop_value and (ent_prop_value == ref_prop_value):
                                    num_correct_color+=1
                            else:
                                if ent_prop['name'] in self.color_descriptors:
                                    if ent_prop['value'] == ref_prop['value']:
                                        num_correct_color += 1


                    # hallucinated prop
                    num_hallucinated_properties += len(unpaired_props['prediction'])
                    # missing prop
                    num_missing_properties += len(unpaired_props['reference'])

                    # if paired_props:
                    #     num_correct_properties_weak+=len(paired_props)
                else:
                    if 'properties' in predicted_ent:
                        num_hallucinated_properties+=len(predicted_ent['properties'])
            num_missing_entity = total_ref_entities - len(full_paired_entities)
        else:
            num_missing_properties = total_properties
            num_missing_entity = len(unpaired_entities['reference'])



        if (total_clusters == num_correct_cluster_points) and \
                (total_clusters == num_correct_cluster_distance) and \
                num_hallucinated_entity == 0 and \
                total_ref_entities == num_entity_match_perfect:
            entity_perfect_result = True

        if (num_hallucinated_properties == 0 ) and (total_properties == num_correct_properties_perfect):
            props_perfect_result = True

        return dict(
            total_color_property = total_color_property,
            total_clusters=total_clusters,
            total_properties=total_properties,
            total_cuisine_property=total_cuisine_property,
            total_ref_entities = total_ref_entities,
            num_entity_match_perfect = num_entity_match_perfect,
            num_entity_match_weak = num_entity_match_weak,
            num_correct_entity_type = num_correct_entity_type,
            num_correct_cluster_distance = num_correct_cluster_distance,
            num_correct_cluster_points = num_correct_cluster_points,
            num_correct_properties_perfect = num_correct_properties_perfect,
            # num_correct_properties_weak=num_correct_properties_weak,
            total_height_property = total_height_property,
            num_hallucinated_properties = num_hallucinated_properties,
            num_hallucinated_entity = num_hallucinated_entity,
            num_missing_properties=num_missing_properties,
            num_missing_entity=num_missing_entity,
            num_correct_height=num_correct_height,
            num_correct_height_metric = num_correct_height_metric,
            num_correct_height_distance = num_correct_height_distance,
            num_correct_cuisine_properties = num_correct_cuisine_properties,
            num_correct_color = num_correct_color,
            entity_perfect_result = entity_perfect_result,
            props_perfect_result = props_perfect_result
        ), full_paired_entities

    def sort_entities(self, entities1, entities2):
        entities1_sorted = sorted(entities1, key=lambda x: x['name'].lower())
        entities2_sorted = sorted(entities2, key=lambda x: x['name'].lower())
        return entities1_sorted, entities2_sorted