import copy
import re
import pandas as pd
from benchmarking.utils import  DIST_LOOKUP
from collections import Counter

def load_rel_spatial_terms(relative_spatial_terms_path: str):
    relative_spatial_terms = pd.read_csv(relative_spatial_terms_path, sep=';').to_dict(orient='records')
    processed_rel_spatial_term_mapping = {}
    for relative_spatial_term in relative_spatial_terms:
        values = list(map(lambda x: x.rstrip().strip(), relative_spatial_term['Vals'].split(',')))
        relative_spatial_term_dist = relative_spatial_term['Dist']
        for value in values:
            processed_rel_spatial_term_mapping[value] = relative_spatial_term_dist
    return processed_rel_spatial_term_mapping

class RelationAnalyzer:
    def __init__(self, relative_spatial_terms: str='datageneration/data/relative_spatial_terms.csv'):
        self.rel_terms = load_rel_spatial_terms(relative_spatial_terms)
        print(self.rel_terms)
        # todo implement rel spatial

    def compose_dist_metric(self, height):
        dist = re.findall(r'\d+', height)
        if not dist:
            return None, None
        dist = dist[0]
        metric = height.replace(dist,'').replace(' ', '')
        return dist, metric

    def compare_relations(self, reference_data, generated_data, full_paired_entities):
        """
        Check if two lists of relations are identical. There are two different ways how the comparison is done, based on
        whether the order of source and target is relevant or not (only the case in "contains" relations).
        Contains relations (where the order matters) are compared as lists. Other relations (where the order of source
        and target does not matter) is compared as a list of frozensets.

        :param relations1: The first relations list to compare (ref_rel).
        :param relations2: The second relations list to compare (gen_rel).
        :return: Boolean whether the two relations lists are the same.
        """
        total_rels = 0
        total_dist_rels = 0
        total_contains_rels = 0
        num_correct_rel_type = 0
        num_correct_dist_edges = 0
        num_correct_dist_rels = 0
        num_correct_contains_rels = 0
        total_relative_spatial_terms = 0
        num_correct_relative_spatial_terms = 0
        num_correct_dist_metric = 0
        num_correct_dist_value = 0
        num_correct_dist = 0
        perfect_result = False

        ref_rels = reference_data.get('relations', None)
        gen_rels = generated_data.get('relations', None)

        if not ref_rels:
            return dict(total_dist_rels=total_dist_rels,
                        total_contains_rels=total_contains_rels,
                        num_correct_dist_rels=num_correct_dist_rels,
                        num_correct_dist_edges=num_correct_dist_edges,
                        num_correct_contains_rels=num_correct_contains_rels,
                        total_relative_spatial_terms=total_relative_spatial_terms,
                        num_correct_relative_spatial_terms=num_correct_relative_spatial_terms,
                        num_correct_dist_metric=num_correct_dist_metric,
                        num_correct_dist_value=num_correct_dist_value,
                        num_correct_dist=num_correct_dist,
                        num_correct_rel_type=num_correct_rel_type,
                        perfect_result=perfect_result)

        if ref_rels:
            total_rels = len(ref_rels)
            for ref_rel in ref_rels:
                ref_type = ref_rel.get('type')
                if 'dist' in ref_type:
                    total_dist_rels+=1
                elif 'contain' in ref_type:
                    total_contains_rels+=1
                else:
                    raise Exception(f'Invalid type:\n{ref_rel}')
                spatial_term = ref_rel.get('spatial_term', None)
                if spatial_term:
                    total_relative_spatial_terms+=1

        ref_id_to_text_map = {}
        gen_id_to_text_map = {}
        for ref_ent, gen_ent in full_paired_entities:
            ref_id_to_text_map[str(ref_ent['id'])] = ref_ent['normalized_name']
            gen_id_to_text_map[str(gen_ent['id'])] = gen_ent['normalized_name']
        if not gen_rels:
            return dict(total_dist_rels=total_dist_rels,
                        total_contains_rels=total_contains_rels,
                        num_correct_dist_rels=num_correct_dist_rels,
                        num_correct_dist_edges=num_correct_dist_edges,
                        num_correct_contains_rels=num_correct_contains_rels,
                        total_relative_spatial_terms=total_relative_spatial_terms,
                        num_correct_relative_spatial_terms=num_correct_relative_spatial_terms,
                        num_correct_dist_metric=num_correct_dist_metric,
                        num_correct_dist_value=num_correct_dist_value,
                        num_correct_dist=num_correct_dist,
                        num_correct_rel_type=num_correct_rel_type,
                        perfect_result=perfect_result)

        for ref_rel in ref_rels:
            ref_rel_src_id = str(ref_rel['source'])
            # i have to do this due to the non-matched entities (that happens when a large mistypo: gas station, brand:lokup a gast station)
            if ref_rel_src_id in ref_id_to_text_map:
                ref_rel['source'] = ref_id_to_text_map[ref_rel_src_id]
            ref_rel_trg_id = str(ref_rel['target'])
            if ref_rel_trg_id in ref_id_to_text_map:
                ref_rel['target'] = ref_id_to_text_map[ref_rel_trg_id]

        for gen_rel in gen_rels:
            gen_rel_src_id = str(gen_rel['source'])
            # i have to do this due to the non-matched entities (that happens when a large mistypo: gas station, brand:lokup a gast station)
            if gen_rel_src_id in gen_id_to_text_map:
                gen_rel['source'] = gen_id_to_text_map[gen_rel_src_id]
            gen_rel_trg_id = str(gen_rel['target'])
            if gen_rel_trg_id in gen_id_to_text_map:
                gen_rel['target'] = gen_id_to_text_map[gen_rel_trg_id]

        ref_distance_rels = set()
        gen_distance_rels = set()
        ref_contain_rels = list()
        gen_contain_rels = list()
        ref_rels_metadata = {}
        gen_rels_metadata = {}
        for idx in range(len(ref_rels)):
            if ref_rels[idx]['type'] == 'contains':
                ref_contain_rels.append([ref_rels[idx]["source"], ref_rels[idx]["target"]])
            elif ref_rels[idx]["type"] == "dist" or ref_rels[idx]["type"] == "distance":
                # source target value spatial term (None if it does not exist)
                spatial_term = ref_rels[idx].get('spatial_term', None)
                src_trg_pair = frozenset({ref_rels[idx]["source"], ref_rels[idx]["target"]})
                print(src_trg_pair)

                src_trg_pair_str = "-".join(sorted(map(str, src_trg_pair)))
                ref_rels_metadata[src_trg_pair_str] = {'distance': ref_rels[idx]['value'], 'spatial_term': spatial_term}
                ref_distance_rels.add(src_trg_pair)

        for idx in range(len(gen_rels)):
            if gen_rels[idx]["type"] == "contains":
                gen_contain_rels.append([gen_rels[idx]["source"], gen_rels[idx]["target"]])
            elif gen_rels[idx]["type"] == "dist" or gen_rels[idx]["type"] == "distance":
                src_trg_pair = frozenset({gen_rels[idx]["source"], gen_rels[idx]["target"]})
                print(src_trg_pair)
                src_trg_pair_str = "-".join(sorted(map(str, src_trg_pair)))
                gen_rels_metadata[src_trg_pair_str] = {'distance': gen_rels[idx]['value']}
                gen_distance_rels.add(src_trg_pair)

        # Distance Rels Comparison
        ref_distance_rels = Counter(ref_distance_rels)
        gen_distance_rels = Counter(gen_distance_rels)
        for ref_distance_rel in ref_distance_rels:
            if ref_distance_rel in gen_distance_rels:
                num_correct_dist_edges+=1
                print('===ref distance rel....')
                print(ref_distance_rel)
                src_trg_pair_str = "-".join(sorted(map(str, ref_distance_rel)))
                ref_metadata = ref_rels_metadata[src_trg_pair_str]
                gen_metadata = gen_rels_metadata[src_trg_pair_str]

                ref_dist_value, ref_dist_metric = self.compose_dist_metric(ref_metadata['distance'])
                gen_dist_value, gen_dist_metric = self.compose_dist_metric(gen_metadata['distance'])

                ref_spatial = ref_metadata['spatial_term']

                if ref_dist_value == gen_dist_value:
                    num_correct_dist_value += 1
                if ref_dist_metric == gen_dist_metric:
                    num_correct_dist_metric += 1
                else:
                    gen_dist_metric = DIST_LOOKUP.get(gen_dist_metric, None)
                    if ref_dist_metric == gen_dist_metric:
                        num_correct_dist_metric += 1

                ref_normalized_dist = f'{ref_dist_value} {ref_dist_metric}'
                gen_normalized_dist = f'{gen_dist_value} {gen_dist_metric}'

                if ref_normalized_dist == gen_normalized_dist:
                    num_correct_dist+=1
                    num_correct_dist_rels+=1

                    if ref_spatial:
                        num_correct_relative_spatial_terms+=1
        # Contains Rels Comparison
        gen_contain_rels_copy = copy.deepcopy(gen_contain_rels)
        for ref_contain_rel in ref_contain_rels:
            for idx, gen_contain_rel in enumerate(gen_contain_rels_copy):
                if ref_contain_rel == gen_contain_rel:
                    gen_contain_rel.pop(idx)
                    num_correct_contains_rels+=1

        num_correct_rel_type = num_correct_contains_rels+num_correct_dist_edges

        if (num_correct_rel_type == total_rels) and \
                (total_rels==num_correct_dist_rels+num_correct_contains_rels):
            perfect_result = True

        return dict(total_rels = total_rels,
                    total_dist_rels=total_dist_rels,
                    total_contains_rels=total_contains_rels,
                    num_correct_dist_rels=num_correct_dist_rels,
                    num_correct_dist_edges=num_correct_dist_edges,
                    num_correct_contains_rels=num_correct_contains_rels,
                    total_relative_spatial_terms=total_relative_spatial_terms,
                    num_correct_relative_spatial_terms=num_correct_relative_spatial_terms,
                    num_correct_dist_metric=num_correct_dist_metric,
                    num_correct_dist_value=num_correct_dist_value,
                    num_correct_dist=num_correct_dist,
                    num_correct_rel_type=num_correct_rel_type,
                    perfect_result=perfect_result)