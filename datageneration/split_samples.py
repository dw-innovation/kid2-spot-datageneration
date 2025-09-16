import os
import jsonlines

"""
Utility to split a large JSON Lines (.jsonl) file into multiple smaller chunks.

The main function, `split_jsonl`, reads an input .jsonl file and writes out
contiguous chunks (each up to `max_samples_per_chunk` records) into an output
directory. Files are named sequentially as: `samples_chunk_1.jsonl`, `samples_chunk_2.jsonl`, ...

Example
-------
>>> split_jsonl("data/samples.jsonl", "data/chunks", max_samples_per_chunk=500)

Requirements
------------
- jsonlines
"""

def split_jsonl(input_file, output_dir, max_samples_per_chunk=500):
    """
    Split a JSONL file into multiple chunk files with up to N records each.

    Parameters
    ----------
    input_file : str
        Path to the source JSON Lines file to split.
    output_dir : str
        Directory where chunk files will be written. Created if it does not exist.
    max_samples_per_chunk : int, optional
        Maximum number of JSON objects per output chunk file (default: 500).

    Returns
    -------
    None
        Writes chunk files to `output_dir`. Each chunk is named:
        `samples_chunk_<index>.jsonl` starting from 1.

    Notes
    -----
    - Preserves the order of records from the input file.
    - Uses streaming IO; memory footprint is bounded by `max_samples_per_chunk`.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with jsonlines.open(input_file, 'r') as reader:
        chunk_count = 1
        samples = []
        for sample in reader:
            samples.append(sample)
            if len(samples) == max_samples_per_chunk:
                output_file = os.path.join(output_dir, f'samples_chunk_{chunk_count}.jsonl')
                with jsonlines.open(output_file, 'w') as writer:
                    writer.write_all(samples)
                samples = []
                chunk_count += 1

        # Write remaining samples to the last chunk file
        if samples:
            output_file = os.path.join(output_dir, f'samples_chunk_{chunk_count}.jsonl')
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(samples)


# Usage
split_jsonl('datageneration/results/v12/samples.jsonl', 'datageneration/results/v12')
