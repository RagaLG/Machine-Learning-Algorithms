import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=1, metric="cosine", p=2):
        # metric = {"minkowski", "euclidean", "manhattan", "cosine"}
        # p value only matters when metric = "minkowski"
        # notice that for "cosine", 1 is closest and -1 is furthest
        # therefore usually cosine_dist = 1 - cosine(x,y)
        self.n_neighbors = int(n_neighbors)
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        # X: pd.DataFrame, independent variables, float
        # y: list, np.array or pd.Series, dependent variables, int or str
        self.classes_ = list(set(list(y)))
        self.X_train = X
        self.y_train = y
        return

    def predict(self, X):
        # X: pd.DataFrame, independent variables, float
        # return predictions: list
        predictions = []
        for index, row in X.iterrows():
            # Calculate distances
            distances = self.calculate_distances(row)
            # Find indices of k-nearest neighbors
            indices = np.argsort(distances)[:self.n_neighbors]
            # Get the labels of k-nearest neighbors
            knn_labels = [self.y_train[i] for i in indices]
            # Count the occurrences of each class among k-nearest neighbors
            counter = Counter(knn_labels)
            # Get the most common class label
            prediction = counter.most_common(1)[0][0]
            predictions.append(prediction)
        return predictions

    def predict_proba(self, X):
        # X: pd.DataFrame, independent variables, float
        # prob is a dict of prediction probabilities belonging to each category
        probs = []
        for index, row in X.iterrows():
            # Calculate distances
            distances = self.calculate_distances(row)
            # Find indices of k-nearest neighbors
            indices = np.argsort(distances)[:self.n_neighbors]
            # Get the labels of k-nearest neighbors
            knn_labels = [self.y_train[i] for i in indices]
            # Count the occurrences of each class among k-nearest neighbors
            counter = Counter(knn_labels)
            # Calculate probabilities for each class
            prob = {label: count / self.n_neighbors for label, count in counter.items()}
            probs.append(prob)
        return pd.DataFrame(probs, columns=self.classes_)

    def calculate_distances(self, x):
        if self.metric == "minkowski":
            distances = np.sum(np.abs(self.X_train - x) ** self.p, axis=1) ** (1/self.p)
        elif self.metric == "euclidean":
            distances = np.linalg.norm(self.X_train - x, axis=1)
        elif self.metric == "manhattan":
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        elif self.metric == "cosine":
            dot_product = np.dot(self.X_train, x)
            norm_x = np.linalg.norm(x)
            norm_data = np.linalg.norm(self.X_train, axis=1)
            distances = 1 - dot_product / (norm_x * norm_data)
        else:
            raise ValueError("Invalid metric. Supported metrics are 'minkowski', 'euclidean', 'manhattan', and 'cosine'.")
        return distances