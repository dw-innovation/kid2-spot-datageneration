import requests
import pandas as pd
import time
import jsonlines
from tqdm import tqdm

API_URL = "http://kid2spotapis11-live.dwelle.de:4000/transform-sentence-to-imr"

def query(payload, environment):
    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }

    body = {
        "sentence": payload["inputs"],
        "model": "llama",
        "username": "kid2",
        "environment": "kid2"
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=body,
        timeout=120
    )

    response.raise_for_status()
    return response


if __name__ == '__main__':
    start_time = time.perf_counter()

    output_file = 'benchmarking/results/spot_Mistral-Small-3_remote.jsonl'
    benchmarking_file = 'benchmarking/data/goldstandard_testing_dataset.xlsx'
    gold_sheet_name = 'descriptor_updates_02022026'
    environment = "development"

    gold_ds = pd.read_excel(benchmarking_file, sheet_name=gold_sheet_name)
    gold_ds = gold_ds.to_dict(orient='records')

    responses = []

    for item in tqdm(gold_ds, total=len(gold_ds)):
        sentence = item['sentence']
        request_start = time.perf_counter()

        response = query({
            "inputs": sentence.lower(),
        }, environment)

        request_end = time.perf_counter()
        latency = request_end - request_start

        raw_output = response.json()

        responses.append({
            'sentence': sentence,
            'model_result': raw_output,
            'latency_seconds': latency
        })

    with jsonlines.open(output_file, mode='w') as writer:
        writer.write_all(responses)

    end_time = time.perf_counter()

    total_runtime = end_time - start_time
    avg_latency = total_runtime / len(gold_ds)

    print(f"\nStart time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Average latency per request: {avg_latency:.3f} seconds")