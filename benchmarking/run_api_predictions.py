import argparse
import os
import time

import jsonlines
import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_URL = os.environ["API_URL"]


def query(payload, environment):
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    body = {
        "sentence": payload["inputs"],
        "model": "llmhub",
        "username": os.environ["API_USERNAME"],
        "environment": os.environ["API_ENVIRONMENT"],
    }

    response = requests.post(API_URL, headers=headers, json=body, timeout=120)
    response.raise_for_status()
    return response


if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = argparse.ArgumentParser(
        description="Run API predictions on the benchmarking dataset."
    )
    parser.add_argument(
        "--output_file", default="benchmarking/results/gpt_oss_remote.jsonl"
    )
    parser.add_argument(
        "--benchmarking_file",
        default="benchmarking/data/goldstandard_testing_dataset.xlsx",
    )
    parser.add_argument("--gold_sheet_name", default="descriptor_updates_13052026")
    parser.add_argument("--environment", default="development")
    args = parser.parse_args()

    output_file = args.output_file
    benchmarking_file = args.benchmarking_file
    gold_sheet_name = args.gold_sheet_name
    environment = args.environment

    gold_ds = pd.read_excel(benchmarking_file, sheet_name=gold_sheet_name)
    gold_ds = gold_ds.to_dict(orient="records")

    responses = []

    MAX_RETRIES = 10
    RETRY_DELAY = 5  # seconds

    for item in tqdm(gold_ds, total=len(gold_ds)):
        sentence = item["sentence"]

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                request_start = time.perf_counter()

                response = query(
                    {
                        "inputs": sentence.lower(),
                    },
                    environment,
                )

                request_end = time.perf_counter()
                latency = request_end - request_start
                raw_output = response.json()["rawOutput"]["content"]

                if not raw_output:
                    raise ValueError("None value got")

                responses.append(
                    {
                        "sentence": sentence,
                        "model_result": raw_output,
                        "latency_seconds": latency,
                    }
                )
                break  # success — move on to the next item

            except Exception as e:
                print(
                    f"\n[Attempt {attempt}/{MAX_RETRIES}] Error on sentence: {sentence!r}"
                )
                print(f"  {type(e).__name__}: {e}")
                if attempt < MAX_RETRIES:
                    print(f"  Retrying in {RETRY_DELAY}s...")
                    time.sleep(RETRY_DELAY)
                else:
                    print("  Max retries reached — skipping this sentence.")
                    responses.append(
                        {
                            "sentence": sentence,
                            "model_result": None,
                            "latency_seconds": None,
                            "error": str(e),
                        }
                    )

    with jsonlines.open(output_file, mode="w") as writer:
        writer.write_all(responses)

    end_time = time.perf_counter()

    total_runtime = end_time - start_time
    avg_latency = total_runtime / len(gold_ds)

    print(f"\nStart time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total runtime: {total_runtime:.2f} seconds")
    print(f"Average latency per request: {avg_latency:.3f} seconds")
