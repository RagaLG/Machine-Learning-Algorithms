import pandas as pd
import numpy as np
from copy import deepcopy

class my_AdaBoost:

    def __init__(self, base_estimator=None, n_estimators=50):
        self.base_estimator = base_estimator
        self.n_estimators = int(n_estimators)
        self.estimators = [deepcopy(self.base_estimator) for i in range(self.n_estimators)]

    def fit(self, X, y):
        self.classes_ = list(set(y))
        k = len(self.classes_)
        n = len(y)
        w = np.array([1.0 / n] * n)
        labels = np.array(y)
        self.alpha = []

        for i in range(self.n_estimators):
            # Train base classifier with sampled training data
            self.estimators[i].fit(X, labels, sample_weight=w)
            predictions = self.estimators[i].predict(X)
            diffs = predictions != y
            # Compute error rate and alpha for estimator i
            error = np.sum(diffs * w) / np.sum(w)
            # If one base estimator predicts perfectly, use that base estimator only
            if error == 0:
                self.alpha.append(1)
                self.estimators = [self.estimators[i]]
                break
            # Compute alpha for estimator i (don't forget to use k for multi-class)
            alpha_i = 0.5 * np.log((1 - error) / error) + np.log(k - 1)
            self.alpha.append(alpha_i)
            # Update wi
            w *= np.exp(alpha_i * diffs)
            w /= np.sum(w)

        return self

    def predict(self, X):
        probs = self.predict_proba(X)
        predictions = [self.classes_[np.argmax(prob)] for prob in probs.to_numpy()]
        return predictions

    def predict_proba(self, X):
        probs = np.zeros((len(X), len(self.classes_)))

        for i, label in enumerate(self.classes_):
            # Calculate probs for each label
            prob = np.sum([alpha * (estimator.predict(X) == label) for alpha, estimator in zip(self.alpha, self.estimators)], axis=0)
            probs[:, i] = prob

        probs /= np.sum(probs, axis=1, keepdims=True)
        return pd.DataFrame(probs, columns=self.classes_)