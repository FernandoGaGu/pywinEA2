import random

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from deap import creator
from deap import base
from deap import tools
from functools import partial
from copy import deepcopy

from pywinEA2 import fitness as pea2_fitness
from pywinEA2 import base as pea2_base
from pywinEA2 import validation as pea2_valid


class FeatureSelectionGA(pea2_base.BaseWrapper):
    """
    Supervised learning

    todo. block access to internal attributes
    todo. option for minimization
    todo. add support for crossover operators
    todo. add support for other selection operators
    """
    MIN_NUM_FEATURES = 2     # todo. update

    VALID_MUTATION_OPERATORS = {   # allowed mutation operators
        'bit_flip': tools.mutFlipBit,
        'uniform': tools.mutUniformInt
    }
    VALID_CROSSOVER_OPERATORS = {  # allowed crossover operators
        'one_point': tools.cxOnePoint,
        'two_point': tools.cxTwoPoint
    }
    VALID_SEL_OPERATORS = {      # allowed selection operators
        'tournament': tools.selTournament,
        'sus': tools.selStochasticUniversalSampling
    }
    DEFAULT_MUTATION_ARGS = {
        'bit_flip': dict(
            indpb=0.05
        ),
        'uniform': dict(
            indpb=0.05,
            low=0,
            up=1,
        )
    }
    DEFAULT_CROSSOVER_ARGS = {
        'one_point': dict(),
        'two_point': dict(),
    }
    DEFAULT_SELECTION_ARGS = {
        'tournament': dict(
            tournsize=2
        ),
        'sus': dict()
    }

    def __init__(
            self,
            data: pd.DataFrame,
            model: sklearn.base.BaseEstimator,
            score: str,
            y: list,
            population_size: int,
            max_generations: int,
            optim: str,
            objective_kw: dict = None,
            # genetic algorithm operators
            p_crossover: float = 0.5,
            p_mutation: float = 0.2,
            mutation_op: str = 'bit_flip',
            mutation_kw: dict = None,
            crossover_op: str = 'one_point',
            crossover_kw: dict = None,
            selection_op: str = 'tournament',
            selection_kw: dict = None,
            # optimized features
            fixed_feats: list = None,
            target_feats: list = None,
            # cross-validation parameters
            cv: int or None = None,
            cv_reps: int or None = None,
            stratified: bool = False,
            # execution parameters
            random_seed: int or None = None,
            n_jobs: int = 1):

        input_parameters = FeatureSelectionGA.__checkInputParameters(
            data=data, model=model, score=score, y=y, population_size=population_size, max_generations=max_generations,
            optim=optim, objective_kw=objective_kw, p_crossover=p_crossover, p_mutation=p_mutation,
            mutation_op=mutation_op, mutation_kw=mutation_kw, crossover_op=crossover_op, crossover_kw=crossover_kw,
            selection_op=selection_op, selection_kw=selection_kw, fixed_feats=fixed_feats, target_feats=target_feats,
            cv=cv, cv_reps=cv_reps, stratified=stratified, random_seed=random_seed, n_jobs=n_jobs)

        self._set_parameters(input_parameters)

    def updateParam(self, **kwargs):
        """ Function used to update the algorithm parameters """
        new_params = deepcopy(self._input_parameters)
        if ('mutation_op' in kwargs) and \
                (kwargs['mutation_op'] != self._input_parameters['mutation_op']) and \
                ('mutation_kw' not in kwargs):
            # reset mutation optional arguments
            new_params['mutation_kw'] = None
        if ('crossover_op' in kwargs) and \
                (kwargs['crossover_op'] != self._input_parameters['crossover_op']) and \
                ('crossover_kw' not in kwargs):
            # reset crossover optional arguments
            new_params['crossover_kw'] = None
        if ('selection_op' in kwargs) and \
                (kwargs['selection_op'] != self._input_parameters['selection_op']) and \
                ('selection_kw' not in kwargs):
            # reset selection optional arguments
            new_params['selection_kw'] = None

        for k, v in kwargs.items():
            pea2_valid.inputParamUnitaryTest(
                k in self._input_parameters,
                'Unrecognized input parameter "{}". Available parameters are: {}'.format(
                    k, ', '.join(list(self._input_parameters.keys())))
            )
            new_params[k] = v

        new_params = FeatureSelectionGA.__checkInputParameters(**new_params)
        self._set_parameters(new_params)

    def getParams(self):
        """ Function used to get the algorithm parameters. Parameters can be accessed also as a dictionary. """
        return self._input_parameters

    def getToolbox(self) -> dict:
        """ Function used to get the configured toolbox (DEAP).  """
        return self._toolbox

    def _set_parameters(self, input_parameters: dict):
        """ Method used to set the algorithm parameters. """
        # model-data parameters
        self._model = input_parameters['model']
        self._data  = input_parameters['data']
        self._y = input_parameters['y']
        self._fixed_feats = input_parameters['fixed_feats']
        self._target_feats = input_parameters['target_feats']

        # algorithm parameters
        self._population_size = input_parameters['population_size']
        self._max_generations = input_parameters['max_generations']
        self._optim = input_parameters['optim']
        self._objective_kw = input_parameters['objective_kw']
        self._p_crossover = input_parameters['p_crossover']
        self._p_mutation = input_parameters['p_mutation']
        self._mutation_op = input_parameters['mutation_op']
        self._mutation_kw = input_parameters['mutation_kw']
        self._crossover_op = input_parameters['crossover_op']
        self._crossover_kw = input_parameters['crossover_kw']
        self._selection_op = input_parameters['selection_op']
        self._selection_kw = input_parameters['selection_kw']

        # other parameters
        self._random_seed = input_parameters['random_seed']
        self._n_jobs = input_parameters['n_jobs']

        # create objects
        # create cross-validation object
        if input_parameters['cv'] is not None:
            if input_parameters['stratified']:  # with class stratification
                cv_obj = RepeatedStratifiedKFold(n_splits=input_parameters['cv'], n_repeats=input_parameters['cv_reps'])
            else:
                cv_obj = RepeatedKFold(n_splits=input_parameters['cv'], n_repeats=input_parameters['cv_reps'])
        else:
            cv_obj = None

        # create fitness function
        self._fitness_obj = pea2_fitness.FeatureSelectionFitness(
            model=self._model,
            score=input_parameters['score'],
            X=self._data[self._target_feats].values,
            X_fixed=self._data[self._fixed_feats].values,
            y=self._data[self._y[0]].values,  # todo. current support for one dependent variable
            cv=cv_obj,
            n_jobs=self._n_jobs,
            **self._objective_kw
        )

        if hasattr(self, '_toolbox'):    # remove previous toolbox
            del self._toolbox

        self._toolbox = base.Toolbox()

        # save input parameters
        self._input_parameters = input_parameters

        self._register()

    @classmethod
    def __checkInputParameters(cls, data, model, score, y, population_size, max_generations, optim, objective_kw,
                               p_crossover, p_mutation, mutation_op, mutation_kw, crossover_op, crossover_kw,
                               selection_op, selection_kw, fixed_feats, target_feats, cv, cv_reps, stratified,
                               random_seed, n_jobs):
        """ Method used to check the input parameters """
        # check input parameters
        pea2_valid.checkMultiInputTypes(
            ('data',            data,            [pd.DataFrame]),
            ('model',           model,           [sklearn.base.BaseEstimator]),
            ('score',           score,           [str]),
            ('y',               y,               [str, list]),
            ('population_size', population_size, [int]),
            ('max_generations', max_generations, [int]),
            ('optim',           optim,           [str]),
            ('objective_kw',    objective_kw,    [dict, type(None)]),
            ('p_crossover',     p_crossover,     [float]),
            ('p_mutation',      p_mutation,      [float]),
            ('mutation_op',     mutation_op,     [str]),
            ('mutation_kw',     mutation_kw,     [dict, type(None)]),
            ('crossover_op',    crossover_op,    [str]),
            ('crossover_kw',    crossover_kw,    [dict, type(None)]),
            ('selection_op',    selection_op,    [str]),
            ('selection_kw',    selection_kw,    [dict, type(None)]),
            ('fixed_feats',     fixed_feats,     [list, type(None)]),
            ('target_feats',    target_feats,    [list, type(None)]),
            ('cv',              cv,              [int, type(None)]),
            ('cv_reps',         cv_reps,         [int, type(None)]),
            ('stratified',      stratified,      [bool]),
            ('random_seed',     random_seed,     [int, type(None)]),
            ('n_jobs',          n_jobs,          [int])
        )
        # input parameters transform
        y = [y] if not isinstance(y, list) else y
        optim = optim.lower()

        pea2_valid.inputParamUnitaryTest(len(y) == 1, 'Currently support for one "y" variable')
        pea2_valid.inputParamUnitaryTest(
            optim.lower() in ['max', 'min'],
            'Parameter "optim" must be one of "min" (for minimization) or "max" (for maximization)')

        # select data
        if fixed_feats is None and target_feats is None:   # use all features (no fixed features)
            fixed_feats = None
            target_feats = list(set(data.columns.tolist()) - set(y))
        elif fixed_feats is not None and target_feats is None:  # specified fixed features
            fixed_feats = fixed_feats
            target_feats = list(set(data.columns.tolist()) - set(y) - set(fixed_feats))
        elif fixed_feats is None and target_feats is not None:  # specified target features
            fixed_feats = list(set(data.columns.tolist()) - set(y) - set(target_feats))
            target_feats = target_feats
        else:  # specified fixed and target features
            fixed_feats = fixed_feats
            target_feats = target_feats

        # Check that "y" is not included in neither fixed nor target features
        for feat in y:
            if fixed_feats is not None:
                pea2_valid.inputParamUnitaryTest(
                    feat not in fixed_feats, 'feature "{}" cannot be in target nor fixed features'.format(feat))
            pea2_valid.inputParamUnitaryTest(
                feat not in target_feats, 'feature "{}" cannot be in target nor fixed features'.format(feat))

        pea2_valid.inputParamUnitaryTest(
            len(target_feats) >= FeatureSelectionGA.MIN_NUM_FEATURES,
            'The number of target features ("target_feats") cannot be less than {}'.format(
                FeatureSelectionGA.MIN_NUM_FEATURES))

        # Check that fixed and target features not intersect
        if fixed_feats is not None:
            pea2_valid.inputParamUnitaryTest(
                len(set(fixed_feats).intersection(set(target_feats))) == 0,
                '"fixed_feats" cannot intersect with "target_feats"')

        # Check mutation, crossover and selection operators
        pea2_valid.inputParamUnitaryTest(
            mutation_op in FeatureSelectionGA.VALID_MUTATION_OPERATORS,
            'Unrecognized mutation operator "{}". Valid operators are: {}'.format(
                mutation_op, ', '.join(FeatureSelectionGA.VALID_MUTATION_OPERATORS)))
        pea2_valid.inputParamUnitaryTest(
            crossover_op in FeatureSelectionGA.VALID_CROSSOVER_OPERATORS,
            'Unrecognized crossover operator "{}". Valid operators are: {}'.format(
                crossover_op, ', '.join(FeatureSelectionGA.VALID_CROSSOVER_OPERATORS)))
        pea2_valid.inputParamUnitaryTest(
            selection_op in FeatureSelectionGA.VALID_SEL_OPERATORS,
            'Unrecognized selection operator "{}". Valid operators are: {}'.format(
                selection_op, ', '.join(FeatureSelectionGA.VALID_SEL_OPERATORS)))

        # select default kwargs if operator kwargs were not defined
        mutation_kw = mutation_kw if mutation_kw is not None else FeatureSelectionGA.DEFAULT_MUTATION_ARGS[mutation_op]
        crossover_kw = crossover_kw if crossover_kw is not None else FeatureSelectionGA.DEFAULT_CROSSOVER_ARGS[crossover_op]
        selection_kw = selection_kw if selection_kw is not None else FeatureSelectionGA.DEFAULT_SELECTION_ARGS[selection_op]
        objective_kw = objective_kw if objective_kw is not None else {}

        return {
            'data': data,
            'model': model,
            'score': score,
            'y': y,
            'population_size': population_size,
            'max_generations': max_generations,
            'optim': optim,
            'objective_kw': objective_kw,
            'p_crossover': p_crossover,
            'p_mutation': p_mutation,
            'mutation_op': mutation_op,
            'mutation_kw': mutation_kw,
            'crossover_op': crossover_op,
            'crossover_kw': crossover_kw,
            'selection_op': selection_op,
            'selection_kw': selection_kw,
            'fixed_feats': fixed_feats,
            'target_feats': target_feats,
            'cv': cv,
            'cv_reps': cv_reps,
            'stratified': stratified,
            'random_seed': random_seed,
            'n_jobs': n_jobs}

    def _register(self):
        """ DESCRIPTION
        This method was created for encapsulation
        """
        self.clearToolbox()

        # register deap elements
        # -- create fitness function
        # todo. current support mono-objective, extend for multi-objective
        if self._optim == 'max':
            creator.create('Fitness', base.Fitness, weights=(1.0,))
        elif self._optim == 'min':
            creator.create('Fitness', base.Fitness, weights=(-1.0,))
        else:
            assert False   # better prevent

        # -- create individual
        creator.create('Individual', list, fitness=creator.Fitness)   # binary representation
        self._toolbox.register(
            'createIndividuals', tools.initRepeat, creator.Individual,
            partial(random.randint, a=0, b=1), len(self._target_feats))
        self._toolbox.register('createPopulation', tools.initRepeat, list, self._toolbox.createIndividuals)

        # -- create operators
        self._toolbox.register('evaluate', self._fitness_obj)
        self._toolbox.register('select', FeatureSelectionGA.VALID_SEL_OPERATORS[self._selection_op], **self._selection_kw)
        self._toolbox.register('mate', FeatureSelectionGA.VALID_CROSSOVER_OPERATORS[self._crossover_op], **self._crossover_kw)
        self._toolbox.register('mutate', FeatureSelectionGA.VALID_MUTATION_OPERATORS[self._mutation_op], **self._mutation_kw)


class MultiObjFeatureSelectionNSGA2(FeatureSelectionGA):
    """ todo. inform about the selection operator (not used)"""
    def _register(self):
        """ DESCRIPTION
        This method was created for encapsulation
        """
        self.clearToolbox()
        # create wrapper for multi-objective fitness
        # -- create fitness function
        # second weight fot pywinEA2.fitness.FeatureNumFitness objective
        if self._optim == 'max':
            creator.create('Fitness', base.Fitness, weights=(1.0, -1.0))
        elif self._optim == 'min':
            creator.create('Fitness', base.Fitness, weights=(-1.0, -1.0))
        else:
            assert False   # better prevent

        # -- create individual
        creator.create('Individual', list, fitness=creator.Fitness)   # binary representation
        self._toolbox.register(
            'createIndividuals', tools.initRepeat, creator.Individual,
            partial(random.randint, a=0, b=1), len(self._target_feats))
        self._toolbox.register('createPopulation', tools.initRepeat, list, self._toolbox.createIndividuals)

        # -- create operators
        self._toolbox.register(
            'evaluate', pea2_fitness.FeatureSelectionFitnessMinFeats.createFromFeatureSelectionFitness(
                self._fitness_obj, **self._objective_kw))
        self._toolbox.register('select', tools.selNSGA2)
        self._toolbox.register('mate', FeatureSelectionGA.VALID_CROSSOVER_OPERATORS[self._crossover_op], **self._crossover_kw)
        self._toolbox.register('mutate', FeatureSelectionGA.VALID_MUTATION_OPERATORS[self._mutation_op], **self._mutation_kw)






