import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder

from visualize_data import ZivsVisualizer
from process_data import ZivsProcessor
from itertools import product
from sklearn import tree
from sklearn import feature_selection
from sklearn import model_selection
from sklearn import metrics

if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
    test_data = pd.read_csv('data/test.csv')

    # check which columns are null
    for dataset in [data, test_data]:
        # complete data
        ZivsProcessor.complete_missing_data(dataset, "Age", "median")
        ZivsProcessor.complete_missing_data(dataset, "Embarked", "mode")
        ZivsProcessor.complete_missing_data(dataset, "Fare", "median")

    # remove redundant features
    data = data.drop(columns=['PassengerId', 'Ticket', 'Cabin'])
    test_data = test_data.drop(columns=['PassengerId', 'Ticket', 'Cabin'])

    # create features
    for dataset in [data, test_data]:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

        dataset['IsAlone'] = 1  # initialize to yes/1 if alone
        dataset['IsAlone'].loc[
            dataset['FamilySize'] > 1] = 0  # now update to no/0 if family size is greater than 1

        # get title by name
        dataset['Title'] = dataset['Name'].str.split(
            ", ", expand=True)[1].str.split(".", expand=True)[0]

        dataset['FareBin'] = pd.qcut(dataset['Fare'], 4)
        dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int), 5)

        dataset = ZivsProcessor.convert_rare_values(data, 'Title')

    # convert features
    features = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']
    for dataset in [data, test_data]:
        ZivsProcessor.convert_features_with_label_encoder(dataset, features)

    # data
    labels = data['Survived']
    data = data.drop(columns=['Survived'])

    # convert to dummy variables
    dummy_features = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age',
                      'Fare', 'FamilySize', 'IsAlone']
    dummy_data = pd.get_dummies(data[dummy_features])

    # real model
    # this is an alternate to train_test_split
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6,
                                            random_state=0)
    dtree = tree.DecisionTreeClassifier(random_state=0)

    # feature selection by RFE (recursive feature elimination)
    dtree_rfe = feature_selection.RFECV(dtree, step=1, scoring='accuracy', cv=cv_split)
    dtree_rfe.fit(dummy_data, labels)

    # Get eliminated features
    X_rfe = dummy_data.columns.values[dtree_rfe.get_support()]
    # rfe_results = model_selection.cross_validate(dtree, dummy_data[X_rfe], labels, cv=cv_split)

    # train the model based on the new features, which are the original minus the eliminated
    param_grid = {'criterion': ['gini', 'entropy'],
                  'max_depth': [2, 4, 6, 8, 10, None],
                  'random_state': [0]}
    rfe_tune_model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(),
                                                  param_grid=param_grid, scoring='roc_auc',
                                                  cv=cv_split)
    rfe_tune_model.fit(dummy_data[X_rfe], labels)

    # print results
    print('AFTER DT RFE Parameters: ', rfe_tune_model.best_params_)
    print("AFTER DT RFE Training w/bin score mean: {:.2f}".format(
        rfe_tune_model.cv_results_['mean_train_score'][rfe_tune_model.best_index_] * 100))
    print("AFTER DT RFE Tuned Test w/bin score mean: {:.2f}".format(
        rfe_tune_model.cv_results_['mean_test_score'][rfe_tune_model.best_index_] * 100))
    print("AFTER DT RFE Tuned Test w/bin score 3*std: +/- {:.2f}".format(
        rfe_tune_model.cv_results_['std_test_score'][rfe_tune_model.best_index_] * 100 * 3))
