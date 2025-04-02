import re
import pandas as pd
from benchmarking.utils import  DIST_LOOKUP

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

    def compare_relations(self, reference_data, generated_data):
        total_rels = 0
        num_correctly_predicted_ids = 0
        total_dist_rels = 0
        num_predicted_dist_rels = 0
        total_contains_rels = 0
        num_predicted_contains_rels = 0
        total_relative_spatial_terms = 0
        num_predicted_relative_spatial_terms = 0
        num_correct_dist_metric = 0
        num_correct_dist_value = 0
        num_missed_rels = 0
        num_hallucinated_rels = 0

        ref_rels = reference_data.get('relations', None)
        gen_rels = generated_data.get('relations', None)

        if not ref_rels:
            if gen_rels:
                num_hallucinated_rels = len(gen_rels)
            return dict(total_rels=total_rels,
                        total_dist_rels=total_dist_rels,
                        total_contains_rels=total_contains_rels,
                        total_relative_spatial_terms=total_relative_spatial_terms,
                        num_correctly_predicted_ids=num_correctly_predicted_ids,
                        num_predicted_dist_rels=num_predicted_dist_rels,
                        num_predicted_contains_rels=num_predicted_contains_rels,
                        num_correct_dist_metric=num_correct_dist_metric,
                        num_correct_dist_value=num_correct_dist_value,
                        num_predicted_relative_spatial_terms=num_predicted_relative_spatial_terms,
                        num_missed_rels=num_missed_rels,
                        num_hallucinated_rels=num_hallucinated_rels)

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

        # ref_id_to_text_map = {}
        # gen_id_to_text_map = {}
        #
        # for ref_ent, gen_ent in full_paired_entities:
        #     ref_id_to_text_map[ref_ent['id']] = ref_ent['name']
        #     # we mapped with the name of ref name, consider it as name normalization
        #     gen_id_to_text_map[gen_ent['id']] = ref_ent['name']

        # todo: maybe this is not so reliable

        if not gen_rels:
            num_missed_rels = total_rels
            return dict(total_rels=total_rels,
                        total_dist_rels=total_dist_rels,
                        total_contains_rels=total_contains_rels,
                        total_relative_spatial_terms=total_relative_spatial_terms,
                        num_correctly_predicted_ids=num_correctly_predicted_ids,
                        num_predicted_dist_rels=num_predicted_dist_rels,
                        num_predicted_contains_rels=num_predicted_contains_rels,
                        num_correct_dist_metric=num_correct_dist_metric,
                        num_correct_dist_value=num_correct_dist_value,
                        num_predicted_relative_spatial_terms=num_predicted_relative_spatial_terms,
                        num_missed_rels=num_missed_rels,
                        num_hallucinated_rels=num_hallucinated_rels)

        ref_ids = set()
        gen_ids = set()
        if gen_rels:
            for ref_rel, gen_rel in zip(ref_rels, gen_rels):
                ref_type = ref_rel.get('type')
                gen_type = gen_rel.get('type')

                ref_ids.add(ref_rel['source'])
                ref_ids.add(ref_rel['target'])
                gen_ids.add(int(gen_rel['source']))
                gen_ids.add(int(gen_rel['target']))

                if 'dist' in ref_type:
                    if 'dist' in gen_type:
                        num_predicted_dist_rels+=1

                    if (ref_rel['source'] == int(gen_rel['source'])) or (ref_rel['target'] == int(gen_rel['target'])):
                        num_correctly_predicted_ids += 1

                    ref_dist, ref_metric = self.compose_dist_metric(ref_rel['value'])

                    if 'value' in gen_rel:
                        gen_dist, gen_metric = self.compose_dist_metric(gen_rel['value'])

                        if ref_dist == gen_dist:
                            num_correct_dist_value+=1

                        if ref_metric== gen_metric:
                            num_correct_dist_metric += 1
                        else:
                            gen_metric = DIST_LOOKUP.get(gen_metric, None)
                            if ref_metric == gen_metric:
                                num_correct_dist_metric += 1

                    spatial_term = ref_rel.get('spatial_term', None)
                    if spatial_term:
                       spatial_term_value = ref_rel.get('value')
                       gen_term_value = gen_rel.get('value', None)
                       if spatial_term_value == gen_term_value:
                           num_predicted_relative_spatial_terms+=1


                if 'contain' in ref_type:
                    if 'contain' in gen_type:
                        num_predicted_contains_rels+=1

                        # ids are not important in "contains"
                        num_correctly_predicted_ids += 1

        num_missed_rels = total_rels - (num_predicted_dist_rels+num_predicted_contains_rels)

        if max(gen_ids) > max(ref_ids):
            num_hallucinated_rels = max(gen_ids) - max(ref_ids)

        return dict(total_rels=total_rels,
                    total_dist_rels=total_dist_rels,
                    total_contains_rels=total_contains_rels,
                    total_relative_spatial_terms=total_relative_spatial_terms,
                    num_correctly_predicted_ids=num_correctly_predicted_ids,
                    num_predicted_dist_rels=num_predicted_dist_rels,
                    num_predicted_contains_rels=num_predicted_contains_rels,
                    num_correct_dist_metric=num_correct_dist_metric,
                    num_correct_dist_value=num_correct_dist_value,
                    num_predicted_relative_spatial_terms=num_predicted_relative_spatial_terms,
                    num_missed_rels=num_missed_rels,
                    num_hallucinated_rels=num_hallucinated_rels)