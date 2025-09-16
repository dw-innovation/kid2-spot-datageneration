import pandas as pd
import random
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

"""
Script for sentence augmentation using the OpenAI API.

This script loads TSV files containing original sentences and queries, 
then probabilistically modifies the sentences by:
1. Adding "find all"-style instruction variations.
2. Introducing typos or grammar mistakes at varying severity levels.

The modified sentence is sent as a prompt to a GPT model to generate the final altered sentence.
Results are saved to new TSV files with the same structure.

Requirements:
- OPENAI_API_KEY must be set in a .env file or environment variable.
"""

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

# Load the Excel file
input_filenames = ["results/v18_3/dev_v18_3.tsv", "results/v18_3/train_v18_3.tsv"]
for input_filename in input_filenames:
    df = pd.read_csv(input_filename, sep='\t')

    # Define helper lists
    all_variants = ["find all", "show me all", "list all", "give me all", "display all"]
    error_levels = ["low", "medium", "high"]
    error_types = ["typos", "grammar mistakes", "typos and grammar mistakes"]

    # Function to decide if a sentence should be altered
    def maybe_add_all_phrase(query):
        """
        Randomly adds an instruction to include a 'find all'-style phrase in the sentence.

        Parameters:
            query (str): The original query string from the dataset.

        Returns:
            str: Instructional string for the GPT prompt or an empty string.
        """
        if "cluster" not in str(query).lower() and random.random() < 0.20:
            phrase = random.choice(all_variants)
            return (f"The sentence should be updated to use a variation of the phrase \"{phrase}\" for one entity similar to "
                    f"\"show me all X that...\" (where X is the entity from the sentence).\n")
        return ""

    def maybe_add_typos_instruction():
        """
        Randomly adds an instruction to introduce typos or grammar mistakes.

        Returns:
            str: Instructional string for the GPT prompt or an empty string.
        """
        if random.random() < 0.40:
            level = random.choice(error_levels)
            mistake_type = random.choice(error_types)
            return f"Rewrite the sentence to contain a {level} amount of {mistake_type}.\n"
        return ""

    # Loop through each row and update the "sentence"
    for idx, row in df.iterrows():
        print("ID: ", idx, " (", input_filename, ")")
        original_sentence = row["sentence"]
        query = row["query"]

        # Possibly modify the input sentence
        findall_instruction = maybe_add_all_phrase(query)
        typo_instruction = maybe_add_typos_instruction()

        # Build the full prompt
        prompt = (
            f"You are an assistant tasked with modifying the following sentence:\n"
            f"'{original_sentence}'\n\n"
            f"{findall_instruction}"
            f"{typo_instruction}"
            f"Make sure to fulfill all instructions and return only the updated sentence."
        )

        # Call GPT API
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that rewrites user input. You follow the user "
                                             "instructions strictly and are willing to return sentences with typos and grammar "
                                             "mistakes if requested. Grammar mistakes can include things like wrong tense, "
                                             "sentence structure or pluralisation and might therefore require more extensive "
                                             "alterations to the original sentence. You don't change anything that is not explicitly "
                                             "requested, and preserve even details like capitalization."},
                    {"role": "user", "content": prompt}
                ]
            )
            updated_sentence = response.choices[0].message.content.strip()
            df.at[idx, "sentence"] = updated_sentence

        except Exception as e:
            print(f"Error at row {idx}: {e}")
            df.at[idx, "sentence"] = original_sentence  # Keep the original on failure

    # Save the modified DataFrame to a new Excel file
    output_filename = input_filename.replace(".tsv", "_updated.tsv")
    df.to_csv(output_filename, sep='\t', index=False)
    print(f"Updated file saved as: {output_filename}")
