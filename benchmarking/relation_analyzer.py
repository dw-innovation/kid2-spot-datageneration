import re
import pandas as pd

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
        total_rels = 0
        num_correctly_predicted_ids = 0
        total_dist_rels = 0
        num_predicted_dist_rels = 0
        total_contains_rels = 0
        num_predicted_contains_rels = 0
        total_relative_spatial_terms = 0
        num_predicted_relative_spatial_terms = 0
        num_correct_height_metric = 0
        num_correct_height_distance = 0
        hallucinated_rels = 0
        missed_rels = 0

        ref_rels = reference_data.get('relations', None)
        gen_rels = generated_data.get('relations', None)

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

        ref_id_to_text_map = {}
        gen_id_to_text_map = {}

        for ref_ent, gen_ent in full_paired_entities:
            ref_id_to_text_map[ref_ent['id']] = ref_ent['name']

            # we mapped with the name of ref name, consider it as name normalization
            gen_id_to_text_map[gen_ent['id']] = ref_ent['name']

        if gen_rels:
            for ref_rel, gen_rel in zip(ref_rels, gen_rels):
                if (ref_id_to_text_map[ref_rel['source']] == gen_id_to_text_map[gen_rel['source']]) or (ref_id_to_text_map[ref_rel['target']] == gen_id_to_text_map[gen_rel['target']]):
                    num_correctly_predicted_ids +=1

                ref_type = ref_rel.get('type')
                gen_type = gen_rel.get('type')

                if 'dist' in ref_type:
                    if 'dist' in gen_type:
                        num_predicted_dist_rels+=1

                    ref_dist, ref_metric = self.compose_dist_metric(ref_rel['value'])
                    gen_dist, gen_metric = self.compose_dist_metric(gen_rel['value'])

                    if ref_dist == gen_dist:
                        print(ref_dist)
                        print(gen_dist)
                        num_correct_height_distance+=1

                    if ref_metric== gen_metric:
                        print(ref_metric)
                        print(gen_metric)
                        num_correct_height_metric += 1

                if 'contain' in ref_type:
                    if 'contain' in gen_type:
                        num_predicted_contains_rels+=1




        return dict(total_rels=total_rels,
                    total_dist_rels=total_dist_rels,
                    num_correctly_predicted_ids=num_correctly_predicted_ids,
                    num_predicted_dist_rels=num_predicted_dist_rels,
                    num_predicted_contains_rels=num_predicted_contains_rels,
                    num_correct_height_distance=num_correct_height_distance,
                    num_correct_height_metric=num_correct_height_metric)