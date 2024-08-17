import pandas as pd
import numpy as np
from collections import Counter

class my_NB:

    def __init__(self, alpha=1):
        # alpha: smoothing factor
        # P(xi = t | y = c) = (N(t,c) + alpha) / (N(c) + n(i)*alpha)
        # where n(i) is the number of available categories (values) of feature i
        # Setting alpha = 1 is called Laplace smoothing
        self.alpha = alpha

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, str
        # y: list, np.array, or pd.Series, dependent variables, int or str

        # Store the unique classes in self.classes_
        self.classes_ = list(set(list(y)))

        # Calculate the prior probability P(yj) for each class
        self.P_y = Counter(y)
        self.class_probabilities = {}
        for c in self.classes_:
            # Calculate P(yj)
            self.class_probabilities[c] = (y == c).sum() / len(y)
        # Initialize a dictionary to store conditional probabilities P(xi|yj)
        self.P = {}
        # Calculate P(xi|yj) for each feature i and class yj
        for c in self.classes_:
            self.P[c] = {}
            for feature in X.columns:
                feature_values = X[feature].unique()
                self.P[c][feature] = {}
                for value in feature_values:
                    # Calculate P(xi = t | y = c) using Laplace smoothing
                    # (N(t,c) + alpha) / (N(c) + n(i)*alpha)
                    n_t_c = ((X[feature] == value) & (y == c)).sum()
                    n_c = (y == c).sum()
                    n_i = len(feature_values)
                    probability = (n_t_c + self.alpha) / (n_c + n_i * self.alpha)
                    self.P[c][feature][value] = probability
        return self.P

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, str
        # prob is a dict of prediction probabilities belonging to each categories
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # P(yj|x) = P(x|yj)P(yj)/P(x)
        # P(x|yj) = P(x1|yj)P(x2|yj)...P(xk|yj) = self.P[yj][X1][x1]*self.P[yj][X2][x2]*...*self.P[yj][Xk][xk]
        probs = {}
        for label in self.classes_:
            p = self.P_y[label]
            for key in X:
                p *= X[key].apply(lambda value: self.P[label][key][value] if value in self.P[label][key] else 1)
            probs[label] = p
        probs = pd.DataFrame(probs, columns=self.classes_)
        sums = probs.sum(axis=1)
        probs = probs.apply(lambda v: v / sums)
        return probs

    def predict(self, X):
        # X: pd.DataFrame, independent variables, str
        # return predictions: list
        # Hint: predicted class is the class with highest prediction probability (from self.predict_proba)
        probs = self.predict_proba(X)
        predictions = []
        predictions = probs.idxmax(axis=1)
        return predictions





