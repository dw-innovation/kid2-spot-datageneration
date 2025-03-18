import json
import os
import numpy as np
from rapidfuzz import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def write_output(generated_combs, output_file):
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