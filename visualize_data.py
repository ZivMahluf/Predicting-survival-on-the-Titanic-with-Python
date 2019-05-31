import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from process_data import ZivsProcessor


class ZivsVisualizer:

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def save_heatmap(self):
        # check for any correlations between variables
        corr = self.data.assign(labels=self.labels).corr()
        plt.figure(figsize=(5, 5))  # increase pic size
        sns.heatmap(corr, annot=True, annot_kws={"size": 7})
        plt.savefig("images/heatmap.png", type='png')

    def save_classification_graphs(self):
        '''
        Save scatter graphs for all features, where each point in the graph is marked by the label
        of the feature.
        :return: None
        '''
        fig, ax = plt.subplots(1, 1)
        pos = np.where(self.labels == 1)
        neg = np.where(self.labels == 0)
        cdict = {0: 'red', 1: 'green'}
        for feature in self.data.columns:
            ax.scatter(self.data[feature].iloc[neg], neg, c=cdict[0], s=20, label='dead')
            ax.scatter(self.data[feature].iloc[pos], pos, c=cdict[1], s=20, label='alive')
            ax.set_ylabel('labels')
            ax.set_xlabel(feature)
            ax.set_title(feature + ' by label')
            ax.legend()
            plt.savefig('images/' + feature + '.png', format='png')
            plt.cla()

    def percentage_of_ones_for_features(self, threshold=2):
        '''
        Save a bar graph, where the x_axis is the feature name, and the y graph is the percentage
        of y=1 for the value of the feature
        For each binary feature, we will have 2 bars, one for the feature=1 and one for the
        feature=0
        :param threshold: The MAX amount of different values we allow the feature to have
        :return: None
        '''

        def autolabel(rects, xpos='center'):
            """
            Attach a text label above each bar in *rects*, displaying its height.

            *xpos* indicates which side to place the text w.r.t. the center of
            the bar. It can be one of the following {'center', 'right', 'left'}.
            """

            ha = {'center': 'center', 'right': 'left', 'left': 'right'}
            offset = {'center': 0, 'right': 1, 'left': -1}

            for rect in rects:
                height = rect.get_height()
                plt.annotate('{}'.format(height),
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(offset[xpos] * 3, 3),  # use 3 points offset
                             textcoords="offset points",  # in both directions
                             ha=ha[xpos], va='bottom')

        plt.rcParams.update({'font.size': 8})
        fig, ax = plt.subplots(figsize=(20, 6))
        for feature in self.data.columns:
            diff_values = self.data[feature].value_counts()
            if len(diff_values) <= threshold:
                for value in diff_values.index:
                    # will contain 1 when data is 'value' AND label is 1
                    condition_met = np.where(self.data[feature] == value, self.labels, 0)
                    # plot the percentage of feature=value and label=1
                    rects = ax.bar(feature + '=' + str(value),
                                   round(condition_met.sum() / len(condition_met), 2),
                                   label=feature + '=' + str(value))
                    autolabel(rects)

        # rotate x axis test so it'll be seen
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)

        plt.ylabel('percentage of y=1 for ')
        plt.legend(loc='best', frameon=False)
        plt.savefig('images/percentage_of_features.png', format='png')
        plt.cla()



