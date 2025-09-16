import copy
import re
import pandas as pd
from benchmarking.utils import  DIST_LOOKUP, compose_metric
from collections import Counter

"""
Relation analysis utilities.

This module provides:
- `load_rel_spatial_terms`: Load and normalize a CSV mapping of relative spatial
  terms to distance categories.
- `RelationAnalyzer`: Compare gold vs. generated relations, including distance-
  and containment-type relations, distance values/metrics, and (optionally)
  relative spatial terms. Returns detailed counters plus a 'perfect' flag.

Expected data format (high level):
- reference_data / generated_data: dicts with keys:
    - 'relations': list of relation dicts, where each relation has fields like:
        {
          "type": "distance" | "dist" | "contains",
          "source": <entity_id or name>,
          "target": <entity_id or name>,
          "value": "<number> <metric>",            # for distance relations
          "spatial_term": "<term>" (optional)      # e.g., "north of", etc.
        }
    - 'entities': list of entities with 'id' and 'name' (for mapping IDs->names)
- full_paired_entities: list of (ref_entity_dict, gen_entity_dict) pairs where
  each entity dict includes at least 'id' and 'normalized_name'.
"""

def load_rel_spatial_terms(relative_spatial_terms_path: str):
    """
    Load relative spatial terms from a CSV and build a term→distance mapping.

    The CSV is expected to have at least the columns:
      - 'Vals': comma-separated terms/synonyms (e.g., "north of, to the north")
      - 'Dist': associated distance/label for those terms

    Whitespace around individual terms is stripped.

    Args:
        relative_spatial_terms_path: Path to the CSV file.

    Returns:
        dict[str, str]: Mapping from individual term (lowercase preserved as read)
        to its associated distance label.
    """
    relative_spatial_terms = pd.read_csv(relative_spatial_terms_path, sep=',').to_dict(orient='records')
    processed_rel_spatial_term_mapping = {}
    for relative_spatial_term in relative_spatial_terms:
        values = list(map(lambda x: x.rstrip().strip(), relative_spatial_term['Vals'].split(',')))
        relative_spatial_term_dist = relative_spatial_term['Dist']
        for value in values:
            processed_rel_spatial_term_mapping[value] = relative_spatial_term_dist
    return processed_rel_spatial_term_mapping

class RelationAnalyzer:
    """
    Compare relations between reference and generated structures.

    Responsibilities:
      - Load relative spatial terms and keep them as a lookup.
      - Normalize distances via `compose_metric`.
      - Compare distance relations (source/target unordered) and 'contains'
        relations (order matters).
      - Track correctness of relation edges, distance values/metrics, and
        optional spatial terms; determine a 'perfect' result flag.

    Notes:
        - Entity IDs in relations are mapped to normalized names using
          `full_paired_entities` (list of (ref_entity, gen_entity) pairs).
        - For distance metric normalization, unknown metrics are mapped via
          `DIST_LOOKUP` as a fallback.
    """
    def __init__(self, relative_spatial_terms: str='datageneration/prompts/relative_spatial_terms.csv'):
        """
        Initialize the analyzer and load relative spatial term mappings.

        Args:
            relative_spatial_terms: Path to the CSV of relative spatial terms.
        """
        self.rel_terms = load_rel_spatial_terms(relative_spatial_terms)
        print(self.rel_terms)
        # todo implement rel spatial

    def compose_dist_metric(self, dist):
        """
        Normalize a distance string into (value, metric).

        Args:
            dist (str): Distance string, typically "<number> <metric>".

        Returns:
            tuple[str, str]: (value, metric) as produced by `compose_metric`.
        """
        return compose_metric(dist)

    def compare_relations(self, reference_data, generated_data, full_paired_entities):
        """
        Compare two relation lists (gold vs. generated) and compute metrics.

        Distance relations:
            - Order of (source, target) does NOT matter.
            - Edges are compared as frozensets of {source, target}.
            - Values and metrics are compared; metrics also checked via DIST_LOOKUP.

        Contains relations:
            - Order DOES matter; compared as ordered pairs [source, target].

        Also counts/validates (when present) relative spatial terms.

        Args:
            reference_data (dict): Parsed gold YAML dict with 'entities' and 'relations'.
            generated_data (dict): Parsed generated YAML dict with 'entities' and 'relations'.
            full_paired_entities (list[tuple[dict, dict]]): Pairs (ref_entity, gen_entity)
                used to map numeric IDs to normalized names before comparison.

        Returns:
            dict: Metrics summary containing (subset):
                - total_rels, total_dist_rels, total_contains_rels
                - num_correct_rel_type
                - num_correct_dist_edges, num_correct_dist_rels
                - num_correct_contains_rels
                - total_relative_spatial_terms, num_correct_relative_spatial_terms
                - num_correct_dist_metric, num_correct_dist_value, num_correct_dist
                - relation_perfect_result (bool)

        Raises:
            Exception: If a reference relation has an unexpected 'type'.
        """
        print('==reference data==')
        print(reference_data)
        print('==generated data==')
        print(generated_data)
        print('==full_paired entities')
        print(full_paired_entities)
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
            if not gen_rels:
                perfect_result = True
            return dict(
                        total_rels = total_rels,
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
                        relation_perfect_result=perfect_result)

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
                        relation_perfect_result=perfect_result)

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
        if len(gen_contain_rels) > 0:
            print('===gen contain rels===')
            print(gen_contain_rels)
            print('===rel contains rels===')
            print(ref_contain_rels)
            gen_contain_rels_copy = copy.deepcopy(gen_contain_rels)
            for ref_contain_rel in ref_contain_rels:
                for idx, gen_contain_rel in enumerate(gen_contain_rels_copy):
                    if ref_contain_rel == gen_contain_rel:
                        gen_contain_rel.pop(idx)
                        num_correct_contains_rels+=1

        num_correct_rel_type = num_correct_contains_rels+num_correct_dist_edges

        if (num_correct_rel_type == total_rels) and \
                (total_dist_rels==num_correct_dist_rels) and \
                (total_contains_rels==num_correct_contains_rels):
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
                    relation_perfect_result=perfect_result)