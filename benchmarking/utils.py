import copy
import json
import os
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datageneration.utils import split_descriptors

model = SentenceTransformer("all-MiniLM-L6-v2")

def check_equivalent_entities(descriptors, ref, gen):
    """
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
                props_r = property_analyzer.convert_values_to_string(r.get('properties', []))
                props_g = property_analyzer.convert_values_to_string(g.get('properties', []))
                for pr in props_r:
                    if pr['name'] not in descriptors:
                        continue
                    equivalent_properties = descriptors.get(pr['name'])
                    for id, pg in enumerate(props_g):
                        if pg['name'] in equivalent_properties:
                            props_g[id]['name'] = pr['name']

    return gen_copy

def write_output(generated_combs, output_file):
    """
    Writes the generated_combs to JSON with the given output_file path.

    :param generated_combs: The generated combinations.
    :param output_file: The path where the output file should be written.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as out_file:
        for generated_comb in generated_combs:
            json.dump(generated_comb.model_dump(mode="json"), out_file)
            out_file.write('\n')

def find_pairs_fuzzy(list1, list2, threshold=80):
    paired = []
    unpaired = {"list1": [], "list2": list2.copy()}

    for item in list1:
        match, score, idx = process.extractOne(item, list2, scorer=fuzz.ratio)
        if score >= threshold:
            paired.append((item, match))
            unpaired["list2"].remove(match)
        else:
            unpaired["list1"].append(item)

    return paired, unpaired


def find_pairs_semantic(reference_list, prediction_list, threshold=0.7):
    paired = []
    unpaired = {"reference": [], "prediction": prediction_list.copy()}

    # Compute embeddings
    embeddings1 = model.encode(reference_list, convert_to_numpy=True)
    embeddings2 = model.encode(prediction_list, convert_to_numpy=True)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)

    # Find best matches
    for i, row in enumerate(similarity_matrix):
        best_match_idx = np.argmax(row)  # Get index of highest similarity
        best_score = row[best_match_idx]  # Get the highest similarity score

        if best_score >= threshold:
            matched_item = prediction_list[best_match_idx]
            paired.append((reference_list[i], matched_item))
            unpaired["prediction"].remove(matched_item)  # Remove matched item from unpaired list
        else:
            unpaired["reference"].append(reference_list[i])

    return paired, unpaired
def load_key_table(path):
    """
    Loads the primary key table and transforms it into a map where each individual descriptor maps to a list of all
    descriptors in its bundle.

    :param path: The path to the primary key table file.
    :return: descriptors - Map of descriptors.
    """
    primary_key_table = pd.read_excel(path, engine='openpyxl')

    descriptors = {}
    for row in primary_key_table.to_dict(orient='records'):
        descriptors_str = row['descriptors']

        descriptors_lst = list(split_descriptors(descriptors_str))

        for desc in descriptors_lst:
            descriptors[desc] = descriptors_lst

    return descriptors


def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(v) for k, v in sorted(obj.items()) if k != "id"}  # Exclude 'id' key
    elif isinstance(obj, list):
        return sorted((normalize(item) for item in obj), key=lambda x: repr(x))
    return obj

def are_dicts_equal(dict1, dict2):
    return normalize(dict1) == normalize(dict2)
