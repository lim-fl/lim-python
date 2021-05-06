#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""Functions to generate a report on how many clean and malware apps we have in a set of stores from AndroZoo.

I only count the latest versions of the apps in each of the stores.

It uses the androzoo_list.csv file, which incorporates the VirusTotal scores for all the apps along with their provenance store.

Right now I transfer the csv file I downloaded some time ago.
"""

import pandas as pd
import play_scraper
import anzhi_scraper
import appchina_scraper
from mtranslate import translate
from tqdm import tqdm
from time import sleep
import pytest

big_androzoo_list = 'androzoo_list.csv'

stores = ['play.google.com', 'anzhi', 'appchina']
keys = ['clean', 'malware']
androzoo_attributes = ['sha256', 'sha1', 'md5', 'dex_date', 'apk_size',
                       'pkg_name', 'vercode', 'vt_detection', 'vt_scan_date',
                       'dex_size', 'markets']
one_gb = pow(10, 9)

androzoo_attributes_with_category = androzoo_attributes + ['category']
csv_filename = 'androzoo_list_deduplicated.csv'


def up_to_n_gb_csv(how_many_gigabytes):
    output_csv = '{}_gb.csv'.format(how_many_gigabytes)

    big_df = pd.read_csv(big_androzoo_list)
    random_state = 28373
    shuffled_df = big_df.sample(frac=1, random_state=random_state)
    shuffled = {'clean': shuffled_df[shuffled_df['vt_detection'] == 0],
                'malware': shuffled_df[shuffled_df['vt_detection'] > 0]}

    dataframes = capped_dataframes(shuffled, keys, how_many_gigabytes/2)
    combined_df = combine_dataframes(dataframes, keys)

    for int_key in ['vercode', 'vt_detection']:
        combined_df[int_key] = combined_df[int_key].fillna(0.0).astype(int)

    combined_df.to_csv(output_csv, index=False,
                       columns=androzoo_attributes_with_category)

    return output_csv


def test_up_to_n_gb_csv():
    how_many_gigabytes = 2
    expected_filename = '2_gb.csv'

    real_filename = up_to_n_gb_csv(how_many_gigabytes=how_many_gigabytes)
    assert expected_filename == real_filename

    df = pd.read_csv(real_filename)
    filter_empty_categories = df['category'].map(lambda x: len(x.strip()) > 0)
    assert len(df) == len(df[filter_empty_categories])

    assert sum(df['apk_size']) > how_many_gigabytes * one_gb

    half_gigabytes = how_many_gigabytes/2
    clean = df[df['vt_detection'] == 0]
    assert sum(clean['apk_size']) > half_gigabytes * one_gb

    malware = df[df['vt_detection'] > 0]
    assert sum(malware['apk_size']) > half_gigabytes * one_gb


def capped_dataframes(shuffled, keys, how_many_gigabytes):
    how_many_bytes = how_many_gigabytes * one_gb

    dataframes = {}
    for key in keys:
        f = 'with_category_{}_{}_gb.csv'.format(key, how_many_gigabytes)
        try:
            dataframes[key] = pd.read_csv(f)
            print('Loaded {} dataframe from file {}'.format(key, f))
        except Exception:
            dataframes[key] = with_category(df=shuffled[key],
                                            how_many_bytes=how_many_bytes)
            dataframes[key].to_csv(f, index=False,
                                   columns=androzoo_attributes_with_category)

    return dataframes


@pytest.fixture
def complete_df():
    return pd.read_csv(big_androzoo_list)


def test_capped_dataframes(complete_df):
    shuffled = {'clean': complete_df[complete_df['vt_detection'] == 0],
                'malware': complete_df[complete_df['vt_detection'] > 0]}
    dataframes = capped_dataframes(shuffled=shuffled,
                                   keys=keys,
                                   how_many_gigabytes=1)
    expected_dataframe_keys = androzoo_attributes_with_category
    for key in keys:
        real_keys = list(dataframes[key].keys())
        assert real_keys == expected_dataframe_keys


def combine_dataframes(dataframes, keys):
    combined_df = pd.DataFrame()
    for key in keys:
        if combined_df.empty:
            combined_df = dataframes[key]
        else:
            combined_df = combined_df.merge(dataframes[key],
                                            how='outer')

    return combined_df


def with_category(df, how_many_bytes):
    cumsum = 0
    pkg_names = []
    app_categories = []
    pbar = tqdm(total=how_many_bytes, unit='bytes', unit_scale=True)
    for _, app in df.iterrows():
        pkg_name = app['pkg_name']
        if cumsum < how_many_bytes:
            try:
                app_category = category(app)
                pkg_names.append(pkg_name)
                app_categories.append(app_category)
                apk_size = app['apk_size']
                cumsum += apk_size
                pbar.update(apk_size)
                if cumsum >= how_many_bytes:
                    print('Finished finding categories online')
                    pbar.close()
                    break
            except ValueError as e:
                pass
    categories = {'pkg_name': pkg_names, 'category': app_categories}
    categories_df = pd.DataFrame.from_dict(categories)

    return df.merge(categories_df, on='pkg_name', how='inner')


def category(app):
    package_name = app['pkg_name']
    store = app['markets']
    sleep(0.1)
    if 'play.google.com' in store:
        try:
            details = play_scraper.details(package_name)
            category = ' '.join((details['category']))
        except Exception as e:
            raise ValueError('Exception in play_scraper')
    elif 'anzhi' in store:
        try:
            details = anzhi_scraper.details(package_name)
            c = translate(details['category'])
            prefix = 'Category: '
            category = c[len(prefix):] if c.startswith(prefix) else c
        except Exception as e:
            raise ValueError('Exception scraping {} from anzhi: {}'.format(package_name, e))
    elif 'appchina' in store:
        try:
            details = appchina_scraper.details(package_name)
            category = translate(details['category'])
        except Exception as e:
            raise ValueError('Exception scraping {} from Appchina: {}'.format(package_name, e))
    else:
        raise ValueError('No data for the store {}'.format(store))

    if not category.strip():
        raise ValueError('Empty category')
    return category


def generate_csv_file():
    apps = clean_malware_per_store_apps()
    dfs = [apps[key][store]
           for store in stores
           for key in keys]
    combined_df = pd.DataFrame()
    for _df in dfs:
        combined_df = _df if combined_df.empty else combined_df.merge(_df,how='outer')

    for int_key in ['vercode', 'vt_detection']:
        combined_df[int_key] = combined_df[int_key].fillna(0.0).astype(int)

    total_apk_size_terabytes = sum(combined_df['apk_size'])/pow(10,12)

    print('The CSV file has {} apps, in total {} Terabytes'.format(len(combined_df),
                                                                   total_apk_size_terabytes))
    combined_df.to_csv(csv_filename,index=False,columns=androzoo_attributes)
    return csv_filename

def report():
    """Returns and prints the report
    :returns: a report on how many clean and malware apps we have in a set of stores from AndroZoo.
    """
    report = []
    numbers_table = clean_malware_per_store_table()
    caption, numbers_table_string = pretty_string_clean_malware_per_store_table(numbers_table)
    report.append(caption)
    report.append(numbers_table_string)

    print('\n'.join(report))
    return '\n'.join(report)


def clean_malware_per_store_table():
    """Generates a pandas dataframe with how many clean and malware apps we have per store in AndroZoo.
    :returns: table['clean'/'malware'][store] = len(apps[key][store])
    """
    apps = clean_malware_per_store_apps()
    lengths = { key: { store: len(apps[key][store])
                       for store in stores}
                for key in keys}

    for key in keys:
        total_length = 0
        for store in lengths[key]:
            total_length += lengths[key][store]
        lengths[key]['Total'] = total_length

    return pd.DataFrame.from_dict(lengths)


def pretty_string_clean_malware_per_store_table(table):
    """Takes the raw numbers from table and divides them by 1000, reporting e.g. 1K instead of 1000.
    :param table: the table generated with clean_malware_per_store_table()
    :returns: the caption of the table, and the formated contents
    """
    pretty_lengths = { key: { store: '{}K'.format(table[key][store]/pow(10,3))
                              for store in table[key].keys() }
                       for key in table.keys()}

    pretty_dataframe = pd.DataFrame.from_dict(pretty_lengths)

    caption = 'Number of clean and malware apps coming from each store. Filtered duplicates by keeping the newest.'
    contents = pretty_dataframe.to_string()
    return caption, contents
    


def clean_malware_per_store_apps():
    """
    :returns: dict['clean'/'malware'][store] = apps
    """
    dictionary = {'clean': {}, 'malware': {}}
    for store in stores:
        apps = apps_from(store)
        dictionary['clean'][store] = apps[apps['vt_detection'] == 0]
        dictionary['malware'][store] = apps[apps['vt_detection'] > 0]

    return dictionary

def apps_from(store):
    """
    :param store: store provenance
    :returns: a dataframe with the latest versions of the apps in the store
    """
    store_criterion = df['markets'] == store
    apps_from_store = df[store_criterion]
    return apps_from_store.sort_values(by='vercode').drop_duplicates(subset='pkg_name')


if __name__ == '__main__':
    up_to_n_gb_csv(how_many_gigabytes=100)
