#!/usr/bin/env python3
'''Fingerprint apps through permissions.

The goal is to train a classifier with app names and their permissions lists,
and predict from a list the corresponding app.
'''

import sqlite3
from extract_permissions import api_28_permissions
import pandas as pd
import matplotlib.pyplot as plt


def report():
    """
    Generates and prints a written report with relevant results.
    """
    df = pandas_from_database()
    report = []

    
    report.append('There are {} clean apps, and {} malware apps, after removing duplicates.'.format(
        total_clean_apps(df),
        total_malware_apps(df)))
    nonunique_percentages = nonunique_percentages_per_tuple_length(df)[0]
    for up_to_n_permissions in range(10,26,5):
        report.append('An average of {} of apps with up to {} permissions share all of them '.format(
            nonunique_percentages[0:up_to_n_permissions].mean(),
            up_to_n_permissions))
    # internet_only = 'There are {} apps with only the INTERNET permission'.format(proportion_of_internet_only_permissions())

    report.append('{} of apps have unique permissions'.format(
        unique_percentage(df)['Apps with unique permissions']))

    print('\n'.join(report))
    return report


def total_clean_apps(df):
    value_counts_per_class = df['CLASS'].value_counts()
    return value_counts_per_class['CLEAN']


def total_malware_apps(df):
    value_counts_per_class = df['CLASS'].value_counts()
    return value_counts_per_class['MALWARE']


def nonunique_percentages_per_tuple_length(df):
    """
    :param df: the dataframe created from the SQLite database, with PACKAGENAME, SORTED_PERMISSIONS, CLASS.
    :returns: a pandas dataframe with the percentage of apps that have exactly the same permission sorting, per each of the existing tuple lengths.
    """
    data = {length: percentage_of_apps_with_same_permissions(df, length)
            for length in range(maximum_permission_tuple_length(df))}
    piechart_df = pd.DataFrame.from_dict(data, orient='index')
    return piechart_df


def plot_nonunique_percentages_piechart(filename='piechart_repeated_permissions.svg'):
    """
    Plots the results of nonunique_percentages_per_tuple_length
    :param filename: where the plot will be saved. By default, piechart_repeated_permissions.svg.
    :returns: filename
    """
    df = pandas_from_database()
    nonunique_df = nonunique_percentages_per_tuple_length(df)
    nonunique_df.plot.pie(subplots=True, autopct='%.2f')
    plt.savefig(filename)

    return filename


def unique_percentage(df):
    """
    :param df: the dataframe created from the SQLite database, with PACKAGENAME, SORTED_PERMISSIONS, CLASS.
    :returns:{'Apps with unique permissions': X, 'Apps with repeated permissions': 1-X}
    """

    value_counts = df['SORTED_PERMISSIONS'].value_counts()
    apps_with_unique_permissions = sum(value_counts.map(lambda x: x == 1))
    apps_with_repeated_permissions = sum(value_counts[value_counts.map(lambda x: x != 1)])
    repeated_permission_patterns = sum(value_counts.map(lambda x: x != 1))
    total_number_of_apps = df.count()['PACKAGENAME']
    data = {'Apps with unique permissions': apps_with_unique_permissions/total_number_of_apps,
            'Apps with repeated permissions': apps_with_repeated_permissions/total_number_of_apps}
            # 'Number of repeated permission patterns': repeated_permission_patterns}

    return data


def plot_unique_percentage_piechart(filename='piechart_apps_with_unique_permissions.svg'):
    """
    Plots the results of unique_percentage.
    :param filename: where the plot will be saved. By default, piechart_apps_with_unique_permissions.svg.
    :returns: filename
    """
    
    df = pandas_from_database()
    data = unique_percentage(df)
    
    piechart_df = pd.DataFrame.from_dict(data, orient='index')
    piechart_df.plot.pie(subplots=True, autopct='%.2f')
    plt.savefig(filename)

    return filename


def percentage_of_apps_with_same_permissions(df, tuple_length):
    """
    :param df: the dataframe created from the SQLite database, with PACKAGENAME, SORTED_PERMISSIONS, CLASS.
    :param tuple_length: how many permissions the apps must have to be included in the percentage.
    :returns: the percentage of apps with the same permissions, within the apps that have the same number of permissions.
    """
    number_of_apps_with_same_permissions = sum(
        permissions_sorted_by_popularity(df, tuple_length).count()['PACKAGENAME'])
    total_number_of_apps = df.count()['PACKAGENAME']
    return float(number_of_apps_with_same_permissions)/total_number_of_apps
    

def pandas_from_database(db='permissions.db', table='permissions'):
    """Creates a pandas dataframe from a table in a SQLite database, removing duplicated PACKAGENAMEs. It imports all columns of the table.
    :param db: the database
    :param table: the table
    :returns: the dataframe without duplicated package names.
    """
    
    conn = sqlite3.connect(db)
    query = 'SELECT * FROM {};'.format(table)
    df = pd.read_sql_query(query,conn)

    return df.drop_duplicates(subset='PACKAGENAME')

    
def permissions_sorted_by_popularity(df, permission_tuples_length=1):
    """
    :param df: the dataframe created from the SQLite database, with PACKAGENAME, SORTED_PERMISSIONS, CLASS.
    :param permission_tuples_length: how many permissions the apps must have to be included in the percentage. By default, 1.
    :returns: a pandas dataframe containing only the apps with the specified number of permissions, grouped by SORTED_PERMISSIONS.
    """
    split_permissions = df['SORTED_PERMISSIONS'].map(
        lambda x: x.split())
    criterion = split_permissions.map(
        lambda x: len(x) == permission_tuples_length)
    apps = df[criterion]
    grouped_by_permissions = apps.groupby(['SORTED_PERMISSIONS'])
    return grouped_by_permissions


def maximum_permission_tuple_length(df):
    """
    :param df: the dataframe created from the SQLite database, with PACKAGENAME, SORTED_PERMISSIONS, CLASS.
    :returns: the maximum number of permissions in any app of the dataframe.
    """
    tuple_permissions_lengths = df['SORTED_PERMISSIONS'].map(
        lambda x: len(x.split()))
    return max(tuple_permissions_lengths)


def number_of_apps_with_permission(permission):
    """ 
    :param permission: the permission being studied.
    :returns: how many apps contain a given permission.
    """

    df = pandas_from_database()
    split_permissions = df['SORTED_PERMISSIONS'].map(
        lambda x: str(permission) in x.split())

    return sum(split_permissions)
