import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    mutual_info_classif,
    mutual_info_regression)


def getLearningProb(rank: int, population_size: int, min_prob: float = 0.05, max_prob: float = 0.5) -> float:
    """ Calculate the learning probability (based on the Conprehensive Learning PSO) """
    numerator = np.exp(10 * (rank - 1) / (population_size - 1))
    denominator = np.exp(10) - 1
    return min_prob + (max_prob - min_prob) * (numerator / denominator)


def pearsonCorrImportance(data: pd.DataFrame, x_feats: list, y_feat: str, **_) -> dict:
    """ Calculate feature importance based on Pearson's correlation """
    corrs = {}
    for feat in x_feats:
        corrs[feat] = abs(np.corrcoef(data[feat].values, data[y_feat].values)[0, 1])

    return corrs


def mutualInformationImportance(data: pd.DataFrame, x_feats: list, y_feat: str, task: str, **_) -> dict:
    """ Calculate feature importance based on Pearson's correlation """
    if task in ['c', 'classification']:
        mi = mutual_info_classif(  # use default parameters
            X=data[x_feats].values,
            y=data[y_feat].values)
    elif task in ['r', 'regression']:
        mi = mutual_info_regression(  # use default parameters
            X=data[x_feats].values,
            y=data[y_feat].values)
    else:
        raise TypeError('"task" argument must be one of: "classification" or "c", or "regression" or "r"')
    mi_scores = {f: mi_score for f, mi_score in zip(x_feats, mi)}

    return mi_scores


def rankFeatures(data: pd.DataFrame, x_feats: list, y_feat: str, rankFunction: str, **kwargs) -> pd.DataFrame:
    """ Function that returns a DataFrame with x_feats sorted by importance in descending order. """
    assert rankFunction in list(VALID_RANK_FUNCTIONS.keys())

    data = data.copy()

    # calculate feature importance
    feature_importance = VALID_RANK_FUNCTIONS[rankFunction](
        data=data,
        x_feats=x_feats,
        y_feat=y_feat,
        **kwargs)

    assert isinstance(feature_importance, dict)

    # sort features in descending order
    feature_importance = np.array(sorted(feature_importance, key=lambda v: feature_importance[v]))[::-1]

    return data[feature_importance]


VALID_RANK_FUNCTIONS = {
    'pearsonCorrImportance': pearsonCorrImportance,
    'mutualInformationImportance': mutualInformationImportance,
}


