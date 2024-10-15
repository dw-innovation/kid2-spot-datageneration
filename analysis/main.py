import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

def transform_timestamp_to_str(timestamp_obj):
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
