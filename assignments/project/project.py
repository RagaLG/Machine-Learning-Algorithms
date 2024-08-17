from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


class my_model():
    def __init__(self) -> None:
        ohe = OneHotEncoder()
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, use_idf=True)
        self.preprocessor = make_column_transformer(
                (ohe, ['telecommuting', 'has_company_logo', 'has_questions']),
            (tfidf_vectorizer, 'text')
        )

        self.clf = SGDClassifier()
        # Define the pipeline with hyperparameters to tune
        self.model = make_pipeline(self.preprocessor, self.clf)
        # Define hyperparameters to tune
        param_grid = {
            'sgdclassifier__loss': ['log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
            'sgdclassifier__random_state': [0, 56, 88],
            'sgdclassifier__penalty': ['l2', 'l1', 'elasticnet'],
        }
        # Use GridSearchCV for hyperparameter tuning
        self.grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring='f1', n_jobs=-1)

    def fit(self, X, y):
        X_new = self.preprocess(X)
        self.grid_search.fit(X_new, y)
        best_params = self.grid_search.best_params_
        self.model.set_params(**best_params)
        self.model.fit(X_new, y)
        return

    def predict(self, X):
        X_new = self.preprocess(X)

        predictions = self.model.predict(X_new)
        return predictions

    @staticmethod
    def preprocess(X):
        X['text'] = X['title'] + X['location'] + X['description'] + X['requirements']
        X.drop(['requirements', 'title', 'location', 'description'], axis=1, inplace=True)
        X = X[['text', 'telecommuting', 'has_company_logo', 'has_questions']]
        X.fillna(" ", inplace=True)
        return X