import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import math
from sklearn.model_selection import train_test_split


class ZivsProcessor:
    fill = {"median": lambda data: data.median(),
            "mode": lambda data: data.mode()[0]}

    def __init__(self):
        pass

    def categorize_features_by_values_amount(self, data, threshold=2):
        '''
        Can be used to convert a/b values to 0/1 values using threshold=2
        :param data: the data
        :param threshold: the MAX amount of different values we would like to split
        :return: the categorized data
        '''
        categorized_features = [col for col in data.columns if
                                len(data[col].value_counts()) <= threshold]
        return pd.get_dummies(data, columns=categorized_features, drop_first=True,
                              prefix_sep='$')

    def print_and_return_data_to_be_completed(self, data):
        '''
        Prints the columns that has null values in them (with the amount of values), and return the
        column names
        :return:
        '''
        null_data = data.isnull().sum()
        print('Data columns with null values:\n', null_data)
        return null_data[null_data != 0].index.tolist()

    def complete_missing_data(self, data, feature, filling="median"):
        '''
        Complete the missing values in the data by a certain method (median, mode, etc.)
        :param data: data
        :param feature: feature to be completed
        :param filling: way of completing -- possible options are listed in self.fill dictionary
        :return: None, data is changed globally
        '''
        data[feature].fillna(self.fill[filling](data[feature]), inplace=True)

    def convert_rare_values(self, data, feature, threshold=10, new_value='Misc',
                            print_result=False):
        '''
        Convert the rare values of the feature to a new_value, by a given threshold
        :param data: the data
        :param feature: feature to convert from
        :param threshold: values will less appearances than this will become new_value
        :param new_value: the value to put in rare values
        :param print_result: default is false
        :return: The changed data
        '''
        title_names = (data[feature].value_counts() < threshold)
        data[feature] = data[feature].apply(
            lambda x: new_value if title_names.loc[x] == True else x)
        if print_result:
            print(data[feature].value_counts())
            print("-" * 10)
        return data

    def convert_features_with_label_encoder(self, data, features):
        '''
        Adds the converted feature as a new col with 'feature_Code'
        :param data: the data
        :param features: features to be converted
        :return: changed data
        '''
        from sklearn.preprocessing import LabelEncoder
        label = LabelEncoder()
        for feature in features:
            data[feature + '_Code'] = label.fit_transform(data[feature])
        return data
