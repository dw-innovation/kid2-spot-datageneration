import pandas as pd
import requests
import os
import jsonlines
from dotenv import load_dotenv
from argparse import ArgumentParser
from tqdm import tqdm

load_dotenv()
HF_LLAMA_ENDPOINT_PROD = os.getenv("HF_LLAMA_ENDPOINT_PROD")
HF_LLAMA_ENDPOINT_DEV = os.getenv("HF_LLAMA_ENDPOINT_DEV")
HF_ACCESS_TOKEN = os.getenv("HF_ACCESS_TOKEN")
HF_MAX_NEW_TOKEN = os.getenv("HF_MAX_NEW_TOKEN", 1048)
HF_TOP_P = os.getenv("HF_TOP_P", 0.1)
HF_TEMPERATURE = os.getenv("HF_TEMPERATURE", 0.001)
PROMPT_FILE_DEV = os.getenv("PROMPT_FILE_DEV")

with open(PROMPT_FILE_DEV, 'r') as file:
    PROMPT_DEV = file.read()

# {"sentence": "", "model_result": ""}

headers = {
    "Accept": "application/json",
    "Authorization": f'Bearer {HF_ACCESS_TOKEN}',
    "Content-Type": "application/json"
}

def query(payload, environment):
    endpoint = HF_LLAMA_ENDPOINT_DEV if environment == "development" else HF_LLAMA_ENDPOINT_PROD
    print(f'Environment is {environment}')
    response = requests.post(endpoint, headers=headers, json=payload)
    # response = response.json()
    return response
    # return response[0]['generated_text']


if __name__ == '__main__':
    output_file = 'benchmarking/results/spot_Mistral-Small-24B-Base-2501-unsloth_ep10_training_ds_v16_3-17_1_2-18_3_param-4_prompt-v2-benchmarking050625_remote.jsonl'
    benchmarking_file = 'benchmarking/data/gold_annotations_14072025.xlsx'
    gold_sheet_name = 'gold_annotations_14072025'
    environment = "development"

    gold_ds = pd.read_excel(benchmarking_file, sheet_name=gold_sheet_name)
    gold_ds = gold_ds.to_dict(orient='records')

    responses = []
    for item in tqdm(gold_ds, total=len(gold_ds)):
        sentence = item['sentence']
        response = query({
            "inputs": sentence.lower(),
            "prompt": PROMPT_DEV if environment == "development" else PROMPT,
            "max_new_tokens": HF_MAX_NEW_TOKEN,
            "top_p": HF_TOP_P,
            "temperature": HF_TEMPERATURE
        }, environment)
        raw_output = response.json()[0]['generated_text']
        responses.append({
            'sentence': sentence,
            'model_result': raw_output
        })

    with jsonlines.open(output_file, mode='w') as writer:
        for sample in responses:
            writer.write(sample)
