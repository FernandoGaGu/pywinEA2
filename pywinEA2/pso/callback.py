import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List

from .particle import Particle


class Callback(object):
    """ Abstract class used to identify callbacks"""
    __metaclass__ = ABCMeta

    def __init__(self, name: str):
        self._name = name

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self):
        return self._name

    def __str__(self):
        return self.__repr__()


class EarlyStopping(Callback):
    VALID_STATS = {
        'all': None,
        'mean': np.mean,
        'std': np.std,
        'max': np.max,
        'min': np.min}

    def __init__(self, stat: str, max_iterations_without_improve: int, improvement_tol: float = 0.0,
                 maximization: bool = True):
        super(EarlyStopping, self).__init__(
            name='EarlyStopping(stat="{}", max_iterations_without_improve={}, improvement_tol={}, '
                 'maximization={})'.format(
                    stat, max_iterations_without_improve, improvement_tol, maximization)
        )

        assert stat in EarlyStopping.VALID_STATS.keys()

        self.max_iterations_without_improve = max_iterations_without_improve
        self.improvement_tol = improvement_tol
        self.maximization = maximization
        self._curr_agg_stat = None
        self._it_count = 0
        self._stat_fn = EarlyStopping.VALID_STATS[stat]

    def evaluate(self, particles: List[Particle]):

        # compute population statistic used to compare improvements
        fitness = [p.curr_fitness_value for p in particles]
        agg_stat = self._stat_fn(fitness)

        if self._curr_agg_stat is None:
            # save values for the first iteration
            self._curr_agg_stat = agg_stat
        else:
            improvement = True
            if self.maximization:
                if (agg_stat - self.improvement_tol) <= self._curr_agg_stat:
                    self._it_count += 1
                    improvement = False
            else:
                if (agg_stat + self.improvement_tol) >= self._curr_agg_stat:
                    self._it_count += 1
                    improvement = False

            if improvement:  # reset counter
                self._curr_agg_stat = agg_stat
                self._it_count = 0

        if self._it_count >= self.max_iterations_without_improve:
            return 'stop'

        return 'continue'




