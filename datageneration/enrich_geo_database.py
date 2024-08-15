from tqdm import tqdm
import pandas as pd


WIKIDATA_API_WD_REQUEST_ENDPOINT='https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&titles=TITLE&format=json'

def request_wd_id(wd_title):
    endpoint = WIKIDATA_ENDPOINT.replace('ENTITY_ID', wd_ent)
    response = requests.get(endpoint)
    response.raise_for_status()
    if response.status_code == 200:
        return response.json()

area_names_non_roman_vocab = {}

if __name__ == '__main__':
    geolocation_file = 'datageneration/data/countries+states+cities.json'
    geolocation_data = pd.read_json(geolocation_file)

    for sample in tqdm(geolocation_data.to_dict(orient='records'), total=len(geolocation_data)):
        for state in sample['states']:
            for city in state['cities']:
                print(city)
                # named_area = NamedAreaData(city=city['name'], state=state['name'], country=sample['name'])
                #
                # print(named_area)
