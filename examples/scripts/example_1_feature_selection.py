import os
import sys
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.svm import SVR

sys.path.append(os.path.join('..', '..'))
import pywinEA2


def getDataset() -> pd.DataFrame:
    """ Create the testing dataset """
    X, y = datasets.make_friedman1(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=NOISE,
        random_state=SEED)

    X = pd.DataFrame(X, columns=['feat_{}'.format(f) for f in range(X.shape[1])])
    y = pd.DataFrame(y, columns=['target'])
    data = pd.concat([X, y], axis=1)

    return data


if __name__ == '__main__':
    # genetic algorithm parameters
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 50
    OPTIM_SCORE = 'mean_squared_error'
    OPTIM = 'min'
    PROB_MUTATION = 0.25
    PROB_CROSSOVER = 0.75
    SELECTION_OP = 'tournament'
    MUTATION_OP = 'bit_flip'
    # dataset creation parameters
    N_SAMPLES = 200
    N_FEATURES = 90
    NOISE = 0.5
    # misc
    SEED = 1997

    # load the dataset to be optimized
    data = getDataset()
    # create the model
    model = SVR()
    # create the genetic algorithm instance
    obj = pywinEA2.MultiObjFeatureSelectionNSGA2(
        data=data,
        model=model,
        score=OPTIM_SCORE,
        y=['target'],
        population_size=POPULATION_SIZE,
        p_crossover=PROB_CROSSOVER,
        p_mutation=PROB_MUTATION,
        max_generations=MAX_GENERATIONS,
        target_feats=data.columns.tolist()[:-1],
        optim=OPTIM,
        selection_op=SELECTION_OP,
        mutation_op=MUTATION_OP,
        n_jobs=1
        )

    obj.multiprocessing(n_jobs=4)   # multiprocessing execution

    # execute the algorithm
    report = pywinEA2.run(obj, type='nsga2', verbose=True)

    # show convergence parameters
    report.displayConvergence(title='Convergence')
    report.displayMultiObjectiveConvergence(title='Convergence', objective_names=['MSE', 'Features'])
    report.displayParetoFront(objective_names=['MSE', 'Features'])

    # show features in the pareto front
    pareto_front = report.pareto_front
    for individual in pareto_front:
        print(data.columns.values[:-1][np.array(individual, dtype=bool)])
