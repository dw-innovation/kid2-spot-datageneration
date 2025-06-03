import copy
import json
import os
import re
import random
import torch
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datageneration.utils import split_descriptors

model = SentenceTransformer("cross-encoder/nli-deberta-v3-base")

DIST_LOOKUP = {
    "centimeter": "cm",
    "meters": "m",
    "kilometers": "km",
    "inches": "in",
    "feet": "ft",
    "yards": "yd",
    "miles": "mi"
}

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

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
def find_pairs_semantic(reference_list, prediction_list, threshold=0.7):
    paired = []
    unpaired = {"reference": reference_list.copy(), "prediction": prediction_list.copy()}

    # Compute embeddings
    embeddings1 = model.encode(reference_list, convert_to_numpy=True)
    embeddings2 = model.encode(prediction_list, convert_to_numpy=True)

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings1, embeddings2)
    matched_predictions = set()

    # Find best matches
    for i, row in enumerate(similarity_matrix):
        # Set similarity to -1 for already matched predictions
        for j in matched_predictions:
            row[j] = -1
        best_match_idx = np.argmax(row)  # Get index of highest similarity
        best_score = row[best_match_idx]  # Get the highest similarity score

        if best_score >= threshold:
            matched_item_pred = prediction_list[best_match_idx]
            matched_item_ref = reference_list[i]
            paired.append((reference_list[i], matched_item_pred))
            matched_predictions.add(best_match_idx)
            if matched_item_pred in unpaired['prediction']:
                unpaired["prediction"].remove(matched_item_pred)  # Remove matched item from unpaired list
            if matched_item_ref in unpaired['reference']:
                unpaired['reference'].remove(matched_item_ref)
        else:
            if reference_list[i] not in unpaired['reference']:
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
        if 'minPoints' in obj:
            obj['minpoints'] = obj.pop('minPoints')
            obj['maxdistance'] = obj.pop('maxDistance')

        if 'minpoints' in obj or 'maxdistance' in obj:
            obj['minpoints'] = str(obj['minpoints'])
            obj['maxdistance'] = str(obj['maxdistance'])
        if 'value' in obj:
            if isinstance(obj['value'], int):
                obj['value'] = str(obj['value'])
            obj['value']= obj['value'].lower()
        if obj['name'] == 'height':
            dist, metric = compose_metric(obj['value'])
            if dist:
                obj['value'] = f'{dist} {metric}'
        return {k: normalize(v) for k, v in sorted(obj.items()) if k != "id" and k!="name"}  # Exclude 'id' key
    elif isinstance(obj, list):
        return sorted((normalize(item) for item in obj), key=lambda x: repr(x))
    return obj

def are_dicts_equal(dict1, dict2):
    normalized_dict_1 = normalize(dict1)
    normalized_dict_2 = normalize(dict2)

    print('normalized dict 1')
    print(normalized_dict_1)

    print('normalized dict 2')
    print(normalized_dict_2)

    return normalized_dict_1 == normalized_dict_2

def compose_metric(height):
    dist = re.findall(r'\d+', height)
    if not dist:
        return None, None
    dist = dist[0]
    metric = height.replace(dist, '').replace(' ', '')
    metric = metric.replace('.', '').replace(',', '')
    metric = DIST_LOOKUP.get(metric, metric)
    return dist, metric
