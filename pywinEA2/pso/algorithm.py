import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from typing import List
from functools import partial
from copy import deepcopy

from .. import base as pea2_base
from .population import (
    evaluateBinary,
    generateParticles,
    vanillaReferencesUpdate,
    generateVariableLengthParticles,
    exemplarAssignment,
    lengthChanging
)
from .particle import (
    vanillaPositionUpdate,
    clpsoPositionUpdate
)
from .util import rankFeatures, getLearningProb
from .report import PSOReport
from .callback import Callback

# valid functions used to evaluate the particles fitness
PSO_VALID_EVALUATIONS = {
    'binary': evaluateBinary
}
# valid functions used to update pbest and gbest
PSO_VALID_REFERENCE_UPDATES = {
    'simple': vanillaReferencesUpdate,
}
# valid functions used to update the particle position and speed
PARTICLE_VALID_UPDATES = {
    'simple': vanillaPositionUpdate,
    'clpso': clpsoPositionUpdate
}

# valid actions returned by Callbacks
ALGORITHM_PRIMITIVES = [
    'stop', 'continue'
]


def psoSimple(
        population_size: int,
        fitness_function: pea2_base.FitnessStrategy,
        max_iterations: int,
        pso_evaluation: str = 'binary',
        pso_evaluation_kw: dict = None,
        pso_reference_update: str = 'simple',
        pso_reference_update_kw: dict = None,
        particle_update: str = 'simple',
        particle_update_kw: dict = None,
        particle_init_position: str = 'random',
        particle_init_speed: str = 'random',
        particle_init_position_kwargs: dict = None,
        particle_init_speed_kwargs: dict = None,
        report: PSOReport = None,
        callbacks: List[Callback] = None,
        verbose: bool = True,
        seed: int = None
):
    """ Algorithm steps:

    1. Generate initial population (generateParticles)
    2. Evaluate particles (PSO_VALID_EVALUATIONS)
    3. Update particle pbest/gbest
    4. Update particle position
    5. Back step 3 until convergence


    """
    assert isinstance(population_size, int)
    assert isinstance(max_iterations, int)
    assert issubclass(type(fitness_function), pea2_base.FitnessStrategy)
    assert pso_evaluation in PSO_VALID_EVALUATIONS.keys()
    assert pso_reference_update in PSO_VALID_REFERENCE_UPDATES.keys()
    assert particle_update in PARTICLE_VALID_UPDATES.keys()

    pso_evaluation_kw       = {} if pso_evaluation_kw       is None else pso_evaluation_kw
    pso_reference_update_kw = {} if pso_reference_update_kw is None else pso_reference_update_kw
    particle_update_kw      = {} if particle_update_kw      is None else particle_update_kw

    if seed is None:
        np.random.seed(seed)
        random.seed(seed)

    progress_bar, updateProgressBar = None, None
    if verbose:  # show progress bar
        # TODO. Fix progress bar
        # progress_bar, updateProgressBar = pea2_io.getProgressBar(max_iterations)
        progress_bar = tqdm(total=max_iterations)

    # 1. Initialize particles
    particles = generateParticles(
        population_size=population_size,
        num_features=fitness_function.getNumFeatures(),
        particle_init_position=particle_init_position,
        particle_init_speed=particle_init_speed,
        particle_init_position_kwargs=particle_init_position_kwargs,
        particle_init_speed_kwargs=particle_init_speed_kwargs,
        seed=seed)

    curr_iteration = 0
    while curr_iteration < max_iterations:
        # 2. Evaluate particles
        particles = PSO_VALID_EVALUATIONS[pso_evaluation](
            particles=particles,
            fitness_function=fitness_function,
            **pso_evaluation_kw)

        # 3. Update particles pbest/gbest
        particles = PSO_VALID_REFERENCE_UPDATES[pso_reference_update](particles, **pso_reference_update_kw)

        # 4. Update particles position and speed
        particles = list(map(
            partial(PARTICLE_VALID_UPDATES[particle_update], seed=seed, **particle_update_kw), particles))

        # save particle stats
        if report is not None:
            report.record(particles)

        curr_iteration += 1

        if verbose:  # update progress bar
            # TODO. Fix progress bar
            # updateProgressBar(progress_bar)
            progress_bar.update(1)

        # evaluate callbacks
        if callbacks is not None:
            for callback in callbacks:
                directive = callback.evaluate(particles)
                assert directive in ALGORITHM_PRIMITIVES

                # stop the algorithm
                if directive == 'stop':
                    print('Early stopping activated by "{}"'.format(callback))
                    curr_iteration = np.inf
                    break

    # save population in the report
    if report is not None:
        report.saveParticles(particles)

    return report



def vlpso(
        population_size: int,
        num_population_div: int,
        fitness_function: pea2_base.FitnessStrategy,
        max_iterations: int,
        rank_function: str,
        alpha: int = 99,   # TODO. remove None
        beta: int = 99,    # TODO. remove None
        rank_function_kw: dict = None,

        pso_evaluation: str = 'binary',
        pso_evaluation_kw: dict = None,
        pso_reference_update: str = 'simple',
        pso_reference_update_kw: dict = None,
        particle_update_kw: dict = None,
        particle_init_position: str = 'random',
        particle_init_speed: str = 'random',
        particle_init_position_kwargs: dict = None,
        particle_init_speed_kwargs: dict = None,
        report: PSOReport = None,
        callbacks: list = None,
        verbose: bool = True,
        seed: int = None
):
    """ Algorithm steps:


    """
    assert isinstance(population_size, int)
    assert isinstance(num_population_div, int)
    assert isinstance(rank_function, str)
    assert isinstance(max_iterations, int)
    assert issubclass(type(fitness_function), pea2_base.FitnessStrategy)
    assert pso_evaluation in PSO_VALID_EVALUATIONS.keys()
    assert pso_reference_update in PSO_VALID_REFERENCE_UPDATES.keys()

    # select optional arguments
    rank_function_kw        = {} if rank_function_kw      is None else rank_function_kw
    pso_evaluation_kw       = {} if pso_evaluation_kw       is None else pso_evaluation_kw
    pso_reference_update_kw = {} if pso_reference_update_kw is None else pso_reference_update_kw
    particle_update_kw      = {} if particle_update_kw      is None else particle_update_kw
    report = PSOReport() if report is None else report

    if seed is None:
        np.random.seed(seed)
        random.seed(seed)

    progress_bar, updateProgressBar = None, None
    if verbose:  # show progress bar
        # TODO. Fix progress bar
        #progress_bar, updateProgressBar = pea2_io.getProgressBar(max_iterations)
        progress_bar = tqdm(total=max_iterations)

    # 1. Sort features using an importance measurement
    x_data = pd.DataFrame(fitness_function._X)
    feature_names = ['feat_%d' % i for i in range(x_data.shape[1])]
    x_data.columns = feature_names
    x_data['target'] = fitness_function._y
    x_data_ordered = rankFeatures(
        data=x_data,
        x_feats=feature_names,
        y_feat='target',
        rankFunction=rank_function,
        **rank_function_kw)
    fitness_function._X = x_data_ordered.values
    feature_order = x_data_ordered.columns.tolist()

    # 2. Initialize particles (of variable length)
    particles = generateVariableLengthParticles(
        population_size=population_size,
        num_features=fitness_function.getNumFeatures(),
        num_divisions=num_population_div,
        particle_init_position=particle_init_position,
        particle_init_speed=particle_init_speed,
        particle_init_position_kwargs=particle_init_position_kwargs,
        particle_init_speed_kwargs=particle_init_speed_kwargs,
        seed=seed)

    # 3. Initialize exemplars
    for p in particles:
        p.exemplar = deepcopy(p.position)

    # 4. Evaluate particles
    particles = PSO_VALID_EVALUATIONS[pso_evaluation](
        particles=particles,
        fitness_function=fitness_function,
        **pso_evaluation_kw)

    # 5. Update particles pbest/gbest
    particles = PSO_VALID_REFERENCE_UPDATES[pso_reference_update](particles, **pso_reference_update_kw)

    # 6. Calculate learning probability
    fitness_values = [p.best_fitness_value for p in particles]  # get particle rank
    particle_rank = np.argsort(fitness_values)[::-1] + 1
    for p, p_rank in zip(particles, particle_rank):
        p.rank = p_rank
        p.learning_prob = getLearningProb(p_rank, population_size)

    # 7. assign exemplars
    particles = exemplarAssignment(particles)

    # save particle stats
    if report is not None:
        report.record(particles)

    curr_iteration = 0
    num_length_changing = 0
    invalid_solutions = {}
    while curr_iteration < max_iterations:
        # 9b. Check gbest improvements
        if particles[0].gbest_count > beta:
            num_length_changing += 1

            # length changing
            particles = lengthChanging(particles)

            # re-evaluate particles
            particles = PSO_VALID_EVALUATIONS[pso_evaluation](
                particles=particles,
                fitness_function=fitness_function,
                **pso_evaluation_kw)

            # calculate learning probability for all particles
            fitness_values = [p.best_fitness_value for p in particles]  # get particle rank
            particle_rank = np.argsort(fitness_values)[::-1] + 1
            for p, p_rank in zip(particles, particle_rank):
                p.rank = p_rank
                p.learning_prob = getLearningProb(p_rank, population_size)

            # renew exemplars as True
            for p in particles:
                p.renew_exemplar = True
        else:
            # renew exemplars if true
            for p in particles:
                if p.renew_exemplar:
                    particles = exemplarAssignment(particles)
                    p.renew_exemplar = False

            # evaluate particles
            particles = PSO_VALID_EVALUATIONS[pso_evaluation](
                particles=particles,
                fitness_function=fitness_function,
                **pso_evaluation_kw)

            # 3. Update particles pbest/gbest
            particles = PSO_VALID_REFERENCE_UPDATES[pso_reference_update](particles, **pso_reference_update_kw)

            # 8a. Update particles position and speed
            particles = list(map(
                partial(PARTICLE_VALID_UPDATES['clpso'], seed=seed, **particle_update_kw), particles))

            # Check particle improvement
            for p in particles:
                if p.pbest_count > alpha:
                    p.renew_exemplar = True

        curr_iteration += 1

        # check for the number of -np.inf values
        inf_count = 0
        for p in particles:
            if p.curr_fitness_value == -np.inf:
                inf_count += 1
        invalid_solutions[curr_iteration] = inf_count

        # save particle stats
        if report is not None:
            report.record(particles)

        if verbose:  # update progress bar
            # TODO. Fix progress bar
            #updateProgressBar(progress_bar)
            progress_bar.update(1)

        # evaluate callbacks
        if callbacks is not None:
            for callback in callbacks:
                directive = callback.evaluate(particles)
                assert directive in ALGORITHM_PRIMITIVES

                # stop the algorithm
                if directive == 'stop':
                    print('Early stopping activated by "{}"'.format(callback))
                    curr_iteration = np.inf
                    break

    # save population in the report
    if report is not None:
        report.saveParticles(particles)
        report.addMetadata(num_length_changing_updates=num_length_changing)
        report.addMetadata(invalid_solutions=invalid_solutions)

    return report, feature_order
