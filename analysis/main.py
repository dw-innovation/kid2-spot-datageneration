import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

"""
Script to filter and extract unique input sentences from an Excel file based on a date filter.

This script:
- Reads an Excel file containing timestamped entries.
- Filters the data to only include rows with a specific date.
- Extracts and prints unique input sentences from the filtered data.

Usage:
    python script.py --input_file path/to/file.xlsx --date_filter YYYY-MM-DD
"""


def transform_timestamp_to_str(timestamp_obj):
    """
    Convert a timestamp object to a string in 'YYYY-MM-DD' format.

    Args:
        timestamp_obj (datetime or str): The timestamp object to convert.

    Returns:
        str: A string representation of the date (first 10 characters).
    """
    return str(timestamp_obj)[:10]

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--input_file')
    parser.add_argument('--date_filter')

    args = parser.parse_args()

    input_file = args.input_file
    date_filter = args.date_filter

    df = pd.read_excel(input_file)

    print(df.head())

    df.dropna(subset=['timestamp'], inplace=True)

    df['date'] = df['timestamp'].apply(lambda x: transform_timestamp_to_str(x))
    print(df['date'])
    filtered_df = df[df['date'] == date_filter]

    print(f'Number of filtered data {len(filtered_df)}')

    input_texts = list(filtered_df['inputSentence'].unique())
    print(f'Number of unique data {len(input_texts)}')

    for input_text in input_texts:
        print(input_text)
