from datageneration.utils import split_descriptors
import pandas as pd

old_table = pd.read_csv('SPOT_OSM-tag-bundles_old.csv')
new_table = pd.read_csv('SPOT_OSM-tag-bundles_new.csv')
output_file = 'tag_bundles_diff.csv'

def retrive_all_descriptors(df):
    tmp_list = []
    for row in df.to_dict(orient='records'):
        if isinstance(row['descriptors'], float):
            continue
        descriptors = split_descriptors(row['descriptors'])
        tmp_list.extend(descriptors)
    return tmp_list

old_tags = retrive_all_descriptors(old_table)
new_tags = retrive_all_descriptors(new_table)

new_tags = list(set(new_tags).difference(old_tags))
df_new_tags = pd.DataFrame({"descriptors": list(new_tags)})
df_new_tags.to_csv(output_file, index=False)