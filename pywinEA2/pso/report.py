import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from copy import deepcopy
from functools import partial
from collections import defaultdict

from .particle import Particle


class PSOReport(object):
    VALID_STATS = {
        'all': None,
        'mean': np.mean,
        'std': np.std,
        'max': np.max,
        'min': np.min}
    DEFAULT_STATS = ['mean', 'max', 'std']

    def __init__(self, stats: list = None):

        # select default statistics to record
        stats = PSOReport.DEFAULT_STATS if stats is None else stats

        # check valid statistics
        for stat in stats:
            assert stat in PSOReport.VALID_STATS.keys()

        self._stats = stats
        self._curr_it = 0
        self._history = {}
        self._curr_particles = None
        self._metadata = {}

        # add plotting functions
        setattr(self, 'displayConvergence', partial(displayConvergence, report=self))

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def history(self) -> dict:
        return self._history

    @property
    def stats(self) -> list:
        return self._stats

    @property
    def particles(self) -> None or List[Particle]:
        return self._curr_particles

    def addMetadata(self, **kwargs):
        for k, v in kwargs.items():
            self._metadata[k] = v

    def record(self, particles: List[Particle], iteration: int = None):
        curr_it = self._curr_it if iteration is None else iteration
        assert curr_it not in self.history.keys(), 'overwriting history'

        particle_fitness = np.ma.masked_invalid([p.curr_fitness_value for p in particles])
        self._history[curr_it] = {}
        for stat in self._stats:
            if stat == 'all':
                self._history[curr_it][stat] = deepcopy(particles)
            else:
                self._history[curr_it][stat] = PSOReport.VALID_STATS[stat](particle_fitness)

        if iteration is None:  # if no iteration is specified the inner count will be used
            self._curr_it += 1

    def saveParticles(self, particles: List[Particle]):
        """ Save a deepcopy of the particles. """
        for p in particles:
            assert isinstance(p, Particle)

        self._curr_particles = deepcopy(particles)


def displayConvergence(
    report,
    figsize: tuple = (7, 4),
    dpi: int = 100,
    title: str = None
    ):
    """ DESCRIPTION """
    assert len(report.history) > 0
    report_df = pd.DataFrame(report.history).T

    # check report mean
    has_mean = 'mean' in report_df.columns
    has_std = 'std' in report_df.columns
    has_max = 'max' in report_df.columns
    has_min = 'min' in report_df.columns

    fig, ax = plt.subplots(figsize=figsize)
    fig.set_dpi(dpi)

    if has_mean:
        ax.plot(
            report_df.index.tolist(), report_df['mean'].values,
            label='mean', color='red', ls='solid'
        )
    if has_min:
        ax.plot(
            report_df.index.tolist(), report_df['min'].values,
            label='min', color='red', ls='dashed'
        )

    if has_max:
        ax.plot(
            report_df.index.tolist(), report_df['max'].values,
            label='max', color='red', ls='dotted'
        )

    if has_std and has_mean:
        ax.fill_between(
            report_df.index.tolist(),
            y1=report_df['mean'].values + report_df['std'].values,
            y2=report_df['mean'].values - report_df['std'].values,
            color='grey', alpha=0.2
        )

    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2)
    if title is None:
        ax.set_title('PSO convergence', size=15)
    else:
        ax.set_title(title, size=15)
    plt.show()

