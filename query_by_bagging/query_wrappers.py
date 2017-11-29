from sklearn.datasets import make_blobs
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, BaggingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from histogram_entropy import multiple_histogram_entropy

class AbstractQuery(object):
    def __init__(self, ensemble):
        self.ensemble_models = [m for m in ensemble.estimators_]

    def score(self, X):
        raise NotImplementedError

    def rank(self, X):
        raise NotImplementedError

class ClassifierQuery(AbstractQuery):
    def score_proba(self, X):
        raise NotImplementedError

    def rank_proba(self, X):
        raise NotImplementedError

class BinaryClassifierQuery(ClassifierQuery):
    def score(self, X):
        prediction_matrix = self._get_ensemble_predictions(X)
        results = []
        for prediction_row in prediction_matrix:
            entropy = self._prediction_distribution_entropy(prediction_row)
            results.append(entropy)
        return np.array(results)

    def _get_ensemble_predictions(self, X):
        # Produce a result matrix where each input row corresponds to a row of hard-margin predictions
        predictions = [m.predict(X) for m in self.ensemble_models]
        return np.vstack(predictions).T

    def _prediction_distribution_entropy(self, prediction_row):  # This is really entropy of the prediction distribution
        ctr = Counter(prediction_row)
        if 0 not in ctr or 1 not in ctr: return 0
        n = 1.0 * len(prediction_row)
        p = np.array([ctr[0] / n, ctr[1] / n])
        disagreement = self._entropy(p)
        return disagreement

    def score_proba(self, X):
        prediction_matrix = self._get_ensemble_predictions_proba(X)
        results = []
        for prediction_row in prediction_matrix:
            p = sum(prediction_row) / len(prediction_row)
            results.append(self._entropy(p))
        return np.array(results)

    def _get_ensemble_predictions_proba(self, X):
        # Produce a result matrix where each input row corresponds to a row of probabilistic predictions
        predictions = np.array([m.predict_proba(X) for m in self.ensemble_models])
        _, c, _ = predictions.shape
        return [predictions[:, i] for i in range(c)]

    def _entropy(self, p):
        return sum(- p * np.log(p))

class RegressionQuery(AbstractQuery):
    def score(self, X):
        prediction_matrix = self._get_ensemble_predictions(X)
        return multiple_histogram_entropy(prediction_matrix, nbins=100)

    def _get_ensemble_predictions(self, X):
        # Produce a result matrix where each input row corresponds to a row of hard-margin predictions
        predictions = [m.predict(X) for m in self.ensemble_models]
        return np.vstack(predictions).T

    def rank(self, X):
        raise NotImplementedError

def demo_overlap():
    # This is an example of the "extrapolation paradox" - we intuitively expect more uncertainty outside the sampling region, but we are _very_ certain outside it instead
    X, y = np.concatenate([np.random.normal(1, 1, 20), np.random.normal(2, 1, 20)]).reshape(-1, 1), [0] * 20 + [1] * 20
    # m = BaggingClassifier(LinearSVC(), n_estimators=1000)
    m = BaggingClassifier(LogisticRegression(), n_estimators=1000)
    # m = RandomForestClassifier(n_estimators=1000)
    m.fit(X, y)

    bc = BinaryClassifierQuery(m)
    scores = bc.score(X)
    # scores = bc.score_proba(X)
    print(scores)
    sns.distplot(X[0:,][:20], color='red')
    sns.distplot(X[0:,][20:], color='green')
    plt.show()
    plt.scatter(X[:, 0], scores)
    plt.show()


def demo_regression():
    X = np.random.normal(1, 1, 20)
    y = 2*X + 1 + np.random.normal(1, 1, 20)
    X_t = X.reshape(-1, 1)
    m = BaggingRegressor(LinearRegression(), n_estimators=1000)
    # m = RandomForestClassifier(n_estimators=1000)
    m.fit(X_t, y)

    bc = RegressionQuery(m)
    scores = bc.score(X_t)
    plt.scatter(X, y)
    plt.show()
    plt.scatter(X, scores)
    plt.show()

if __name__ == '__main__':
    # demo_overlap()
    demo_regression()