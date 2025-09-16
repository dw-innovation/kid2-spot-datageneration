import yaml
import pandas as pd
from argparse import ArgumentParser


def preprocessing(yaml_str: str) -> str:
    """
    Parse and clean the query YAML string.

    Specifically:
    - Removes the 'value' key from the 'area' dict if its type is 'bbox'.

    Args:
        yaml_str: A string containing a YAML-formatted query.

    Returns:
        A cleaned YAML string with the necessary modifications.
    """
    query = yaml.safe_load(yaml_str)
    area = query["area"]

    if area['type'] == 'bbox':
        area.pop('value', None)


    query_string = yaml.dump(query, allow_unicode=True)
    return query_string

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--output_file")
    args = parser.parse_args()

    input_file = args.input_file
    output_file = args.output_file

    data = pd.read_csv(input_file, sep='\t')
    data['query'] = data['query'].apply(lambda x: preprocessing(x))

    data.to_csv(output_file, sep='\t', index=False)
