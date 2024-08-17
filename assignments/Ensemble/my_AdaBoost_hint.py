import pandas as pd
import numpy as np
from copy import deepcopy
from pdb import set_trace

class my_AdaBoost:

    def __init__(self, base_estimator = None, n_estimators = 50):        
        # Multi-class Adaboost algorithm (SAMME)
        # base_estimator: the base classifier class, e.g. my_DT
        # n_estimators: # of base_estimator rounds
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str

        self.classes_ = list(set(list(y)))
        k = len(self.classes_)
        n = len(y)
        w = np.array([1.0 / n] * n)
        labels = np.array(y)
        self.alpha = []
        for i in range(self.n_estimators):
            # Sample with replacement from X, with probability w
            sample = np.random.choice(n, n, p=w)
            # Train base classifier with sampled training data
            sampled = X.iloc[sample]
            sampled.index = range(len(sample))
            self.estimators[i].fit(sampled, labels[sample])
            predictions = self.estimators[i].predict(X)
            diffs = np.array(predictions) != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w)
            while error >= (1 - 1.0 / k):
                w = np.array([1.0 / n] * n)
                sample = np.random.choice(n, n, p=w)
                # Train base classifier with sampled training data
                sampled = X.iloc[sample]
                sampled.index = range(len(sample))
                self.estimators[i].fit(sampled, labels[sample])
                predictions = self.estimators[i].predict(X)
                diffs = np.array(predictions) != y
                # Compute error rate and alpha for estimator i
                error = np.sum(diffs * w)
            # If one base estimator predicts perfectly,
            # Use that base estimator only
            if error == 0:
                self.alpha = [1]
                self.estimators = [self.estimators[i]]
                break
            # Compute alpha for estimator i (don't forget to use k for multi-class)
            alpha_i = 0.5 * np.log((1 - error) / error)  # Calculate alpha
            w *= np.exp(-alpha_i * diffs)  # Update weights
            w /= np.sum(w)  # Normalize weights
            self.alpha.append(alpha_i)

            # Update wi
            w = "write your own code"

        # Normalize alpha
        total_alpha = sum(self.alpha)
        self.alpha = [alpha / total_alpha for alpha in self.alpha]
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob: what percentage of the base estimators predict input as class C
        # prob(x)[C] = sum(alpha[j] * (base_model[j].predict(x) == C))
        # return probs = pd.DataFrame(list of prob, columns = self.classes_)
        # Note that len(self.estimators) can sometimes be different from self.n_estimators
        # write your code below
        probs = {}
        for label in self.classes_:
            # Calculate probs for each label
            "write your own code"
            base_predictions = [estimator.predict(X) for estimator in self.estimators]
            weighted_predictions = np.array(
                [alpha * (prediction == label) for alpha, prediction in zip(self.alpha, base_predictions)])
            prob_label = np.sum(weighted_predictions, axis=0)  # Sum over all estimators
            probs[label] = prob_label
        probs = pd.DataFrame(probs, columns=self.classes_)
        return probs





