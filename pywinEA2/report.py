import numpy as np
import matplotlib.pyplot as plt
import warnings
from deap import tools
from functools import partial

from pywinEA2 import base as pea2_base
from pywinEA2 import validation as pea2_valid


def _getFitnessValues(ind):
    return ind.fitness.values


class Report(object):
    """ DESCRIPTION

    track_distribution enables 'dist' key
    todo. separate multiobjective report from Report
    """
    TRACKING_METRICS = {
        'avg': np.nanmean,
        'std': np.nanstd,
        'min': np.nanmin,
        'max': np.nanmax,
        'sum': np.nansum
    }

    def __init__(self):

        # create and configure the statistics
        self._stats = tools.Statistics(_getFitnessValues)
        for k, v in Report.TRACKING_METRICS.items():
            self._stats.register(k, v)

        # variables modified by the pywinEA2.algorithm.run function
        self._logbook = None
        self._population = None
        self._fitness = None
        self._hof = None
        self._algorithm = None

        # add plotting functions
        setattr(self, 'displayConvergence', partial(displayConvergence, report=self))

    @property
    def stats(self):
        return self._stats

    @property
    def logbook(self):
        return self._logbook

    @property
    def population(self):
        return self._population

    @property
    def fitness(self):
        return self._fitness

    @property
    def hall_of_fame(self):
        return self._hof

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def keys(self):
        return list(self.TRACKING_METRICS.keys())

    @property
    def multiobjective(self):
        return self._multiobjective

    def get(self, key: str) -> list or None:
        valid_metrics = list(self.TRACKING_METRICS.keys())
        if key not in valid_metrics:
            raise KeyError('Key "{}" not found. Available keys are: {}'.format(key, list(valid_metrics)))
        if self._logbook is not None:
            return self._logbook.select(key)
        else:
            return None

    def addPopulation(self, population):
        self._population = population

    def addLogbook(self, logbook):
        self._logbook = logbook

    def addHallOfFame(self, hof):
        self._hof = hof

    def addAlgorithm(self, alg: pea2_base.BaseWrapper):
        pea2_valid.checkInputType('alg', alg, [pea2_base.BaseWrapper])
        self._algorithm = alg

    def addFitness(self, fitness: list):
        self._fitness = fitness


class MultiObjectiveReport(Report):
    """ DESCRIPTION """
    def __init__(self):
        super(MultiObjectiveReport, self).__init__()

        # save objetive values
        self._stats.register('multiobj_fitness_values_mean', partial(np.mean, axis=0))
        self._stats.register('multiobj_fitness_values_min', partial(np.min, axis=0))
        self._stats.register('multiobj_fitness_values_max', partial(np.max, axis=0))

        self._hypervolume = None

        # add plotting functions
        setattr(self, 'displayMultiObjectiveConvergence', partial(displayMultiObjectiveConvergence, report=self))
        setattr(self, 'displayParetoFront', partial(displayParetoFront, report=self))

    def get(self, key):
        if 'multiobj_fitness_values' in key:
            return self._logbook.select(key)
        else:
            return super().get(key)

    @property
    def hypervolume(self):
        return self._hypervolume

    @property
    def pareto_front(self):
        """ Returns the pareto front. """
        if self.population is None:
            return None

        pareto_front = tools.ParetoFront()
        pareto_front.update(self.population)

        return pareto_front

    def addHypervolume(self, hypervolume: list):
        self._hypervolume = hypervolume


def displayConvergence(
        report,
        figsize: tuple = (10, 6),
        grid_opacity: float = 0.2,
        avg_linecolor: str = '#C0392B',
        avg_lw: int = 4,
        opt_linecolor: str = '#27AE60',
        opt_lw: int = 2,
        fill_color: str = 'grey',
        fill_opacity: float = 0.4,
        legend_size: int = 15,
        title: str or None = None,
        title_size: int = 20,
        axes_label_size: int = 15):
    """ DESCRIPTION """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        np.arange(len(report.logbook)),
        report.get('avg'),
        label='Average',
        color=avg_linecolor,
        lw=avg_lw
    )
    ax.plot(
        np.arange(len(report.logbook)),
        report.get(report.algorithm.getParams()['optim']),
        label='Best',
        color=opt_linecolor,
        lw=opt_lw
    )

    ax.fill_between(
        np.arange(len(report.logbook)),
        np.array(report.get('avg')) - np.array(report.get('std')),
        np.array(report.get('avg')) + np.array(report.get('std')),
        color=fill_color,
        alpha=fill_opacity
    )

    # layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(grid_opacity)
    ax.legend(prop=dict(size=legend_size))
    if title is not None:
        ax.set_title(title, size=title_size)
    ax.set_ylabel('Objective function', size=axes_label_size)
    ax.set_xlabel('Iteration', size=axes_label_size)
    plt.show()


def displayMultiObjectiveConvergence(
    report,
    figsize: tuple = (12, 6),
    grid_opacity: float = 0.2,
    title: str or None = None,
    title_size: int = 20,
    axes_label_size: int = 15,
    legend_size: int = 15,
    hy_linecolor: str = '#C0392B',
    hy_lw: int = 4,
    cmap: str = 'tab10',
    objective_names: list or None = None):
    """ Description """
    fig, ax = plt.subplots(figsize=figsize)

    cmap = plt.cm.get_cmap(cmap)

    obj_values = np.array(report.get('multiobj_fitness_values_{}'.format(report.algorithm.getParams()['optim'])))

    if objective_names is None:
        objective_names = ['Objective {}'.format(i) for i in range(obj_values.shape[1])]

    if len(objective_names) != obj_values.shape[1]:
        raise TypeError('Incorrect number of objective names (number of names={}, number of objectives={})'.format(
            len(objective_names), obj_values.shape[1]))

    if obj_values.shape[1] > 2:
        warnings.warn(
            'Currently only two objectives can be represented. Number of objectives {}'.format(obj_values.shape[1]))
        objective_names = objective_names[:2]

    obj_values = obj_values[:, :2]  # only two objectives can be represented

    ax_obj1 = ax.twinx()
    ax_obj2 = ax.twinx()

    p1 = ax.plot(
        np.arange(len(report.hypervolume)),
        report.hypervolume,
        label='Hypervolume',
        color=hy_linecolor,
        lw=hy_lw)

    p2 = ax_obj1.plot(
        np.arange(obj_values.shape[0]),
        obj_values[:, 0],
        color=cmap(0),
        label=objective_names[0],
        lw=hy_lw * 0.5)

    p3 = ax_obj2.plot(
        np.arange(obj_values.shape[0]),
        obj_values[:, 1],
        color=cmap(1),
        label=objective_names[1],
        lw=hy_lw * 0.5)

    # merge legend
    ax.legend(handles=p1 + p2 + p3, loc='best', prop=dict(size=legend_size))

    # modify axes
    ax_obj1.set_ylabel(objective_names[0])
    ax_obj1.spines['right'].set_position(('outward', 10))
    ax_obj1.spines['top'].set_visible(False)
    ax_obj2.spines['right'].set_position(('outward', 60))
    ax_obj2.spines['top'].set_visible(False)
    ax_obj2.set_ylabel(objective_names[1])

    # layout
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(grid_opacity)
    ax.set_xlabel('Iteration', size=axes_label_size)
    ax.set_ylabel('Hypervolume', size=axes_label_size)

    if title is not None:
        plt.title(title, size=title_size)

    plt.show()


def displayParetoFront(
        report,
        figsize: tuple = (10, 6),
        grid_opacity: float = 0.2,
        color: str = '#C0392B',
        marker_size: int = 50,
        title_size: int = 20,
        axes_label_size: int = 15,
        objective_names: list or None = None):

    pareto_front = report.pareto_front
    fitness = np.array([ind.fitness.values for ind in pareto_front])

    if objective_names is None:
        objective_names = ['Objective {}'.format(i) for i in range(fitness.shape[1])]

    if fitness.shape[1] > 2:
        warnings.warn(
            'Currently only two objectives can be represented. Number of objectives {}'.format(fitness.shape[1]))
        objective_names = objective_names[:2]

    fitness = fitness[:, :2]  # only two objectives can be represented

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(x=fitness[:, 0], y=fitness[:, 1], color=color, s=marker_size)
    ax.plot(fitness[:, 0], fitness[:, 1], color=color, alpha=0.6)

    # layout
    ax.set_title('Number of solutions {}'.format(len(pareto_front)), size=title_size)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(grid_opacity)
    ax.set_ylabel(objective_names[0], size=axes_label_size)
    ax.set_xlabel(objective_names[1], size=axes_label_size)

    plt.show()



