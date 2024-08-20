import numpy as np
import sklearn
import sklearn.metrics as sk_metrics
from sklearn.model_selection import (
    BaseCrossValidator,
    cross_val_score,
    LeaveOneOut)
from copy import deepcopy

from . import base as pea2_base
from . import validation as pea2_valid


def aucScore(y_pred: np.ndarray, y_true: np.ndarray):
    """ Wrapper for sklearn.metrics.roc_auc_score function """
    return sk_metrics.roc_auc_score(y_true=y_true, y_score=y_pred)


class FeatureSelectionFitness(pea2_base.FitnessStrategy):
    """ DESCRIPTION
    todo. add flexibility to the objective metrics. """
    VALID_SCORES = {  # only sklearn.metrics.SCORERS are supported
        'accuracy': sk_metrics.accuracy_score,
        'f1': sk_metrics.f1_score,
        'auc': aucScore,
        'explained_variance': sk_metrics.explained_variance_score,
        'r2': sk_metrics.r2_score,
        'mean_absolute_error': sk_metrics.mean_absolute_error,
        'mean_squared_error': sk_metrics.mean_squared_error,
    }

    def __init__(
            self,
            model: sklearn.base.BaseEstimator,
            score: str or callable,
            X: np.ndarray,
            y: np.ndarray,
            X_fixed: np.ndarray or None = None,
            cv: BaseCrossValidator or None = None,
            return_std: bool = False,
            n_jobs: int = 1, **_):

        super(FeatureSelectionFitness, self).__init__()

        # todo. add input type checking

        self.model = model
        self.cv = cv
        self.n_jobs = n_jobs
        self._X = deepcopy(X)  # avoid inplace modifications
        self._X_fixed = deepcopy(X_fixed)
        self._y = deepcopy(y)
        if isinstance(score, str):
            self._score = FeatureSelectionFitness.VALID_SCORES[score.lower()]
            self._score_repr = score.lower()
        else:
            self._score = score
            self._score_repr = '{}'.format(score)
        self._return_std = return_std

    def getNumFeatures(self) -> int:
        return self._X.shape[1]

    def __call__(self, features: np.ndarray or list) -> tuple:
        """
        (score, 0.0) if cv is None
        (mean score, std score) if cv is not None
        """
        assert self._X.shape[1] == len(features)
        assert np.max(features) <= 1 and np.min(features) >= 0   # check for correct binary representation

        # hack. use all features
        if np.sum(features) == 0:  # no feature was selected
            features = list(np.ones(len(features)))

        if self._X_fixed is not None:
            assert self._X.shape[0] == self._X_fixed.shape[0], 'unitary test'  # better prevent
            Xsub = np.hstack([self._X[:, np.array(features).astype(bool)], self._X_fixed])   # add fixed features
        else:
            Xsub = self._X[:, np.array(features).astype(bool)]

        # todo. check valid maximum value of features
        model_copy = sklearn.base.clone(self.model)

        if self.cv is not None:
            if isinstance(self.cv, LeaveOneOut):
                # apply cross validation schema
                y_preds = []
                y_trues = []
                for train_idx, test_idx in self.cv.split(Xsub):
                    X_train, y_train = Xsub[train_idx], self._y[train_idx]
                    X_test, y_test = Xsub[test_idx], self._y[test_idx]
                    y_preds.append(model_copy.fit(X_train, y_train).predict(X_test))
                    y_trues.append(y_test)
                y_preds = np.array(y_preds)
                y_trues = np.array(y_trues)
                scores = self._score(y_pred=y_preds, y_true=y_trues)
            else:
                # apply cross validation schema
                y_preds = []
                y_trues = []
                for train_idx, test_idx in self.cv.split(Xsub, self._y):
                    X_train, y_train = Xsub[train_idx], self._y[train_idx]
                    X_test, y_test = Xsub[test_idx], self._y[test_idx]
                    y_preds.append(np.array(model_copy.fit(X_train, y_train).predict(X_test)))
                    y_trues.append(np.array(y_test))
                y_preds = np.concatenate(y_preds)
                y_trues = np.concatenate(y_trues)
                scores = self._score(y_pred=y_preds, y_true=y_trues)

            if self._return_std:
                return np.mean(scores), np.std(scores)
            return np.mean(scores),
        else:
            # just fit and evaluate the model
            model_copy = model_copy.fit(Xsub, self._y)
            y_pred = model_copy.predict(Xsub)
            score = self._score(y_true=self._y, y_pred=y_pred)

            return score,


class FeatureSelectionFitnessMinFeats(FeatureSelectionFitness):
    """ DESCRIPTION
    todo. add possibility of exponentiate
    """
    def __init__(
            self,
            model: sklearn.base.BaseEstimator,
            score: str,
            X: np.ndarray,
            y: np.ndarray,
            scale_features: callable = None,
            X_fixed: np.ndarray or None = None,
            cv: BaseCrossValidator or None = None,
            return_std: bool = False,
            n_jobs: int = 1,
            **_):

        super(FeatureSelectionFitnessMinFeats, self).__init__(
            model=model, score=score, X=X, y=y, X_fixed=X_fixed, cv=cv, return_std=return_std, n_jobs=n_jobs)
        self._scale_features = scale_features

    def __call__(self, features: np.ndarray or list) -> tuple:
        """
        (score, 0.0) if cv is None
        (mean score, std score) if cv is not None
        """
        assert self._X.shape[1] == len(features)
        assert np.max(features) <= 1 and np.min(features) >= 0   # check for correct binary representation

        # hack. use all features
        if np.sum(features) == 0:  # no feature was selected
            features = list(np.ones(len(features)))

        if self._X_fixed is not None:
            assert self._X.shape[0] == self._X_fixed.shape[0], 'unitary test'  # better prevent
            Xsub = np.hstack([self._X[:, np.array(features).astype(bool)], self._X_fixed])   # add fixed features
        else:
            Xsub = self._X[:, np.array(features).astype(bool)]

        # todo. check valid maximum value of features
        model_copy = sklearn.base.clone(self.model)

        if self.cv is not None:
            if isinstance(self.cv, LeaveOneOut):
                # apply cross validation schema
                y_preds = []
                y_trues = []
                for train_idx, test_idx in self.cv.split(Xsub):
                    X_train, y_train = Xsub[train_idx], self._y[train_idx]
                    X_test, y_test = Xsub[test_idx], self._y[test_idx]
                    y_preds.append(model_copy.fit(X_train, y_train).predict(X_test))
                    y_trues.append(y_test)
                y_preds = np.array(y_preds)
                y_trues = np.array(y_trues)
                scores = self._score(y_pred=y_preds, y_true=y_trues)
            else:
                # apply cross validation schema
                y_preds = []
                y_trues = []
                for train_idx, test_idx in self.cv.split(Xsub, self._y):
                    X_train, y_train = Xsub[train_idx], self._y[train_idx]
                    X_test, y_test = Xsub[test_idx], self._y[test_idx]
                    y_preds.append(np.array(model_copy.fit(X_train, y_train).predict(X_test)))
                    y_trues.append(np.array(y_test))
                y_preds = np.concatenate(y_preds)
                y_trues = np.concatenate(y_trues)
                scores = self._score(y_pred=y_preds, y_true=y_trues)

            if self._scale_features is not None:
                return np.mean(scores), self._scale_features(np.sum(features))
            return np.mean(scores), np.sum(features)
        else:
            # just fit and evaluate the model
            model_copy = model_copy.fit(Xsub, self._y)
            y_pred = model_copy.predict(Xsub)
            score = self._score(y_true=self._y, y_pred=y_pred)

            if self._scale_features is not None:
                return score, self._scale_features(np.sum(features))
            return score, np.sum(features)

    @classmethod
    def createFromFeatureSelectionFitness(cls, obj: FeatureSelectionFitness, **kwargs):
        """ DESCRIPTION. acts like a factory
        """
        pea2_valid.checkInputType('obj', obj, [FeatureSelectionFitness])
        wrapped_obj = FeatureSelectionFitnessMinFeats(
            model=obj.model,
            score=obj._score_repr,
            X=obj._X,
            y=obj._y,
            X_fixed=obj._X_fixed,
            cv=obj.cv,
            return_std=False,
            n_jobs=obj.n_jobs,
            **kwargs
        )

        return wrapped_obj

