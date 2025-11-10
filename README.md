<img width="1280" height="200" alt="Github-Banner_spot" src="https://github.com/user-attachments/assets/bec5a984-2f1f-44e7-b50d-cc6354d823cd" />

# 🧪 SPOT Datageneration

This repository contains the **synthetic data generation pipeline** for the SPOT system.  
It creates YAML ↔ natural language sentence pairs for training and fine-tuning large language models (LLMs) on structured geospatial query generation.

---

## 🚀 Quickstart

To execute the python files in this repository, please use the shell scripts in the folder "scripts" and adjust the relevant parmeters.

The main steps of the data generation pipeline are the following:
1) retrieve_combinations.sh: Find co-occurence pattern in tags and extraxt example values using [Taginfo](https://taginfo.openstreetmap.org/)
2) generate_combinations.sh: Generate a list of random combinations of areas, tags and distance values in YAML format
3) generate_samples_with_gpt.sh: Feed the YAML from 2) to GPT to create natural sentences for training
4) construct_train_test_split.sh: (Optional) Split the resulting dataset into train & test for model training

For benchmarking, please use the script "run_benchmarking.sh".

---

## ⚙️ Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for GPT-4 or GPT-4o-based sentence generation. |
| `OPENAI_ORG` | Your OpenAI organization ID. |
| `LLM_API_KEY` | Optional: API key for an alternative hosted LLM provider (might require additional configuration). |

> **Security Note:** Store API keys securely in `.env` and never commit real secrets to the repository.

---

## 🔑 Features

- Dynamically generates diverse sentence–YAML pairs from parameterized templates
- Supports multiple personas and writing styles
- Injects natural errors (typos, grammar variations) for robust training
- Supports generation using OpenAI or other LLM endpoints

---

## 🧩 Part of the SPOT System

This module is used to generate training data for:
- [`central-nlp-api`](https://github.com/dw-innovation/kid2-spot-central-nlp-api) — which relies on LLMs fine-tuned on this data
- [`unsloth-training`](https://github.com/dw-innovation/unsloth-training) — which uses this dataset for model adaptation

---

## 📁 Output Structure

The generated data includes:
- Structured YAML files describing geospatial scenes
- Matching synthetic sentences (with optional noise injection)
- Optional metadata about persona, language, and style

---

## 🔗 Related Docs

- [Main SPOT Repo](https://github.com/dw-innovation/kid2-spot)
- [Unsloth Training Script](https://github.com/dw-innovation/unsloth-training)
- [ACL Demo Paper (2025)](https://github.com/dw-innovation/kid2-spot/tree/main/docs)

---

## 🙌 Contributing

We welcome contributions of all kinds — from developers, journalists, mappers, and more!  
See [CONTRIBUTING.md](https://github.com/dw-innovation/kid2-spot/blob/main/CONTRIBUTING.md) for how to get started.
Also see our [Code of Conduct](https://github.com/dw-innovation/kid2-spot/blob/main/CODE_OF_CONDUCT.md).

---

## 📜 License

Licensed under [AGPLv3](../LICENSE).
