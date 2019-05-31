import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from visualize_data import ZivsVisualizer
from process_data import ZivsProcessor
from itertools import product


def find_threshold(D, X, y, sign, j):
    """
    Finds the best threshold.
    D =  distribution
    S = (X, y) the data
    """
    # sort the data so that x1 <= x2 <= ... <= xm
    sort_idx = np.argsort(X[:, j])
    X, y, D = X[sort_idx], y[sort_idx], D[sort_idx]

    thetas = np.concatenate([[-np.inf], (X[1:, j] + X[:-1, j]) / 2, [np.inf]])
    minimal_theta_loss = np.sum(D[y == sign])  # loss of the smallest possible theta
    losses = np.append(minimal_theta_loss, minimal_theta_loss - np.cumsum(D * (y * sign)))
    min_loss_idx = np.argmin(losses)
    return losses[min_loss_idx], thetas[min_loss_idx]


class DecisionStump(object):
    """
    Decision stump classifier for 2D samples
    """

    def __init__(self, D, X, y):
        self.theta = 0
        self.j = 0
        self.sign = 0
        self.train(D, X, y)

    def train(self, D, X, y):
        """
        Train the classifier over the sample (X,y) w.r.t. the weights D over X
        Parameters
        ----------
        D : weights over the sample
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        """
        loss_star, theta_star = np.inf, np.inf
        for sign, j in product([-1, 1], range(X.shape[1])):
            loss, theta = find_threshold(D, X, y, sign, j)
            if loss < loss_star:
                self.sign, self.theta, self.j = sign, theta, j
                loss_star = loss

    def predict(self, X):
        """
        Parameters
        ----------
        X : shape=(num_samples, num_features)
        Returns
        -------
        y_hat : a prediction vector for X shape=(num_samples)
        """

        y_hat = self.sign * ((X[:, self.j] <= self.theta) * 2 - 1)
        return y_hat


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None] * T  # list of base learners
        self.w = np.zeros(T)  # weights
        self.coef = np.zeros(T)

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        self.w = np.full(shape=len(X), fill_value=(1 / len(X)))
        for t in range(self.T):
            self.h[t] = self.WL(self.w, X, y)
            eq = np.equal(self.h[t].predict(X), y)  # get True where they're equal, false otherwise
            error = np.sum(np.where(eq == 0, self.w, 0))  # error is sum of weights
            self.coef[t] = 0.5 * np.log((1 / error) - 1)
            # update weights for each i
            multiply_by_weight = np.where(eq == 0, np.e ** self.coef[t], np.e ** -self.coef[t])
            normalization_factor = np.sum(self.w * multiply_by_weight)
            self.w = (self.w * multiply_by_weight) / normalization_factor
        return self.w

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        sum = 0
        for i in range(max_t):
            sum += self.h[i].predict(X) * self.coef[i]
        return np.sign(sum)

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the correct predictions when predict only with max_t weak learners (float)
        """
        prediction = self.predict(X, max_t)
        return 1 - (np.count_nonzero(np.equal(prediction, y)) / len(y))


def change_plot(ax, title, xlabel, ylabel):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_train_test_error(train_err, test_err, T, noise):
    fig, ax = plt.subplots(1, 1)
    change_plot(ax, "Train and test error for T", "T in [1,500]", "Train and test error")
    ax.axis([0, T, 0, 1])
    ax.plot(range(1, T), train_err, label='Train error')
    ax.plot(range(1, T), test_err, label='Test error')
    ax.legend(loc='best', frameon=False)
    plt.savefig('images/q8_noise=' + str(noise) + '.png', format='png')
    plt.cla()


def q8(ada, X, y, X_test, y_test, T):
    train_err = list()
    test_err = list()
    for t in range(1, T):
        train_err.append(ada.error(X, y, t))
        test_err.append(ada.error(X_test, y_test, t))
    plot_train_test_error(train_err, test_err, T)


def q10(ada, X, y, X_test, y_test, T):
    min_err = 1
    best_t = 1
    for t in range(1, T):
        curr_err = ada.error(X_test, y_test, t)
        if curr_err < min_err:
            min_err = curr_err
            best_t = t

    print("The T with the lowest test error is: " + str(best_t))
    print("Its test error is: " + str(min_err))


if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')
    data = data.drop(columns=['Survived'])

    test_data = pd.read_csv('data/test.csv')
    processor = ZivsProcessor()

    # check which columns are null
    for dataset in [data, test_data]:
        null_cols = processor.print_and_return_data_to_be_completed(dataset)

        # complete data
        processor.complete_missing_data(dataset, "Age", "median")
        processor.complete_missing_data(dataset, "Embarked", "mode")
        processor.complete_missing_data(dataset, "Fare", "median")
        # # remove redundant features
        dataset = dataset.drop(columns=['PassengerId', 'Ticket', 'Cabin'])

        # validate that there's no null columns
        null_cols = processor.print_and_return_data_to_be_completed(dataset)

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

        dataset = processor.convert_rare_values(data, 'Title')

    # convert features
    features = ['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin']
    for dataset in [data, test_data]:
        processor.convert_features_with_label_encoder(dataset, features)

    labels = data['Survived']

    # convert to dummy variables
    dummy_features = ['Sex', 'Pclass', 'Embarked', 'Title', 'SibSp', 'Parch', 'Age',
                      'Fare', 'FamilySize', 'IsAlone']
    data = pd.get_dummies(data[dummy_features])


    # # categorize
    # data = processor.categorize_features_by_values_amount(data, 5)
    #

    #
    # # create adaboost and train it
    # T = 200
    # ada = AdaBoost(DecisionStump, T)
    # weights = ada.train(np.array(data), np.array(labels))
    #
    # # get test data
    # x_test = pd.read_csv('data/train.csv')
    #
    # # preprocess the test data
    # # categorize
    # x_test = processor.categorize_features_by_values_amount(x_test, 5)
    #
    # # remove redundant features
    # IDs = x_test['PassengerId']
    # x_test = x_test.drop(columns=['Name', 'Age', 'PassengerId', 'Ticket', 'Cabin']) # check ticket and cabin
    #
    # # get predictor
    # y_pred = ada.predict(np.array(x_test), T)
    #
    # print()
