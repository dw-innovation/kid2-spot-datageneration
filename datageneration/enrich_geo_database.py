import pandas as pd
import requests
import json
from argparse import ArgumentParser
from tqdm import tqdm
from typing import Dict
from diskcache import Cache

NON_ROMAN_LANGUAGES = [
    'am', 'ar', 'arz', 'azb', 'ba', 'be_x_old', 'be', 'bg', 'bn', 'ce', 'ckb', 'cv',
    'el', 'fa', 'he', 'hi', 'hy', 'hyw', 'ja', 'ka', 'kk', 'kn', 'ky', 'mdf', 'mhr', 'ml',
    'mrj', 'mr', 'myv', 'os', 'pa', 'pnb', 'ps', 'ru', 'skr', 'ta', 'tg', 'th', 'tt', 'udm',
    'ur', 'wuu', 'xmf', 'yi', 'zgh', 'zh_min_nan', 'zh_yue', 'zh'
]

WIKIDATA_API_WIKIPEDIA_SITE_LINKS = 'https://www.wikidata.org/w/api.php?action=wbgetentities&ids=WD_ID&props=sitelinks&format=json'
WIKIDATA_API_WD_REQUEST_ENDPOINT='https://en.wikipedia.org/w/api.php?action=query&prop=pageprops&ppprop=wikibase_item&redirects=1&titles=TITLE&format=json'
cache_wikidata = Cache("wikidata-cache")

@cache_wikidata.memoize()
def request_wd_id(wd_title):
    '''
    Given name of an area, it returns its wikidata id if it has.
    :param wd_title:
    :return:
    '''
    endpoint = WIKIDATA_API_WD_REQUEST_ENDPOINT.replace('TITLE', wd_title)
    response = requests.get(endpoint)
    response.raise_for_status()
    wd_id = None
    data = None
    if response.status_code == 200:
        data= response.json()
    else:
        return wd_id

    pages = data.get("query", {}).get("pages", {})

    for page_id, page_info in pages.items():
        page_props = page_info.get("pageprops", {})
        wd_id = page_props.get("wikibase_item")

        if wd_id:
            return wd_id
    return wd_id

@cache_wikidata.memoize()
def get_dict_of_non_roman_alternatives(wd_id: str) -> Dict:
    endpoint = WIKIDATA_API_WIKIPEDIA_SITE_LINKS.replace('WD_ID', wd_id)
    response = requests.get(endpoint)
    response.raise_for_status()
    data = None
    results = {}
    if response.status_code == 200:
        data= response.json()
    else:
        return results

    entities = data.get("entities", {})
    if wd_id in entities:
        sitelinks = entities[wd_id].get("sitelinks", {})
        for key, value in sitelinks.items():
            if key.endswith('wiki'):
                lang = key.replace('wiki', '')
                alt_name = value['title']

                if lang in NON_ROMAN_LANGUAGES:
                    results[lang] = alt_name

    return results

area_names_non_roman_vocab = {}


def extract_non_roman_alternatives(area_name: str):
    '''
    extract wikidata
    :param area_name:
    :return:
    '''
    non_roman_versions = None
    area_wd_id = request_wd_id(area_name)
    if area_wd_id:
        non_roman_versions = get_dict_of_non_roman_alternatives(area_wd_id)
        if len(non_roman_versions) == 0:
            return non_roman_versions
    return non_roman_versions


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--geolocation_file', type=str, default='datageneration/data/countries+states+cities.json')
    parser.add_argument('--output_file', type=str, default='datageneration/data/area_non_roman_vocab.json')

    args = parser.parse_args()

    geolocation_file = args.geolocation_file
    output_file = args.output_file

    geolocation_data = pd.read_json(geolocation_file)
    vocabulary = {}
    for sample in tqdm(geolocation_data.to_dict(orient='records'), total=len(geolocation_data)):
        country_name = sample['name']
        if country_name not in vocabulary:
            non_roman_versions = extract_non_roman_alternatives(country_name)
            if non_roman_versions:
                vocabulary[country_name] = {'non_roman_versions': non_roman_versions}

        for state in sample['states']:
            state_name = sample['name']

            if state_name not in vocabulary:
                non_roman_versions = extract_non_roman_alternatives(state_name)
                if non_roman_versions:
                    vocabulary[state_name] = {'non_roman_versions': non_roman_versions}

            for city in state['cities']:
                city_name = sample['name']
                if city_name not in vocabulary:
                    non_roman_versions = extract_non_roman_alternatives(city_name)
                    if non_roman_versions:
                        vocabulary[state_name] = {'non_roman_versions': non_roman_versions}

    with open(output_file, 'w') as json_file:
        json.dump(vocabulary, json_file, ensure_ascii=False)