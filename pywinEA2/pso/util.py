import pandas as pd
import numpy as np


def getLearningProb(rank: int, population_size: int) -> float:
    """ Calculate the learning probability (based on the Conprehensive Learning PSO) """
    numerator   = np.exp(10 * (rank - 1) / (population_size -1))
    denominator = np.exp(10) - 1
    return 0.05 + 0.45 * (numerator / denominator)


def pearsonCorrImportance(data: pd.DataFrame, x_feats: list, y_feat: str, **_) -> dict:
    """ Calculate feature importance based on Pearson's correlation """
    corrs = {}
    for feat in x_feats:
        corrs[feat] = abs(np.corrcoef(data[feat].values, data[y_feat].values)[0, 1])

    return corrs


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
    'pearsonCorrImportance': pearsonCorrImportance
}


