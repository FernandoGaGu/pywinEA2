import numpy as np
import random
import joblib
import multiprocessing as mp
from typing import List, Tuple
from collections import defaultdict
from copy import deepcopy

from .particle import Particle
from .. import base as pea2_base


def removeParticleFeatures(
        particles: list,
        mask: np.ndarray = None,
        threshold: float = None,
        min_freq: int = 1):
    """ Function that eliminates those features that have not been selected by more than min_freq particles
    considering a threshold of threshold as binarization for feature selection. """
    new_particles = deepcopy(particles)
    if mask is None:
        particle_masks = np.array([particle.position < threshold for particle in particles])
        feature_mask = (particle_masks.sum(axis=0) >= min_freq)
    else:
        assert isinstance(mask, np.ndarray)
        feature_mask = mask

    # reset particle arguments
    for particle in new_particles:
        particle.speed = deepcopy(particle.speed[feature_mask])
        particle.position = deepcopy(particle.position[feature_mask])
        particle.pbest = deepcopy(particle.pbest[feature_mask])
        particle.gbest = deepcopy(particle.gbest[feature_mask])
        particle.prev_fitness_value = -np.inf
        particle.curr_fitness_value = -np.inf
        particle.best_fitness_value = -np.inf
        particle.gbest_fitness_value = -np.inf
        particle.pbest_count = 0
        particle.gbest_count = 0
        particle.learning_prob = None
        particle.exemplar = None
        particle.renew_exemplar = False
        particle.rank = 0

    return new_particles


def generateVariableLengthParticles(
        population_size: int,
        num_features: int,
        num_divisions: int,
        particle_init_position: str = 'random',
        particle_init_speed: str = 'random',
        particle_init_position_kwargs: dict = None,
        particle_init_speed_kwargs: dict = None,
        seed: int = None) -> List[Particle]:
    """ Population division subroutine from 0.1109/TEVC.2018.2869405 """
    assert population_size >= num_divisions, '"num_divisions" cannot be greater than "population_size"'
    assert num_divisions <= num_features, '"num_features" cannot be greater than "num_divisions"'
    assert population_size % num_divisions == 0, '"population_size" must be a multiple of "num_divisions"'
    assert (num_features / num_divisions) > 1, \
        'num_features / num_features cannot be 0. This will generate particles of size 1'

    if seed is not None:
        np.random.seed(seed)

    # calculate particles sizes
    div_size = int(population_size / num_divisions)   # calculate number of particles per division
    sizes = [int(num_features * (v / num_divisions)) for v in range(1, num_divisions + 1)]

    # initialize particles
    particles = [
        Particle(
            size=size,
            init_position=particle_init_position,
            init_position_kwargs=particle_init_position_kwargs,
            init_speed=particle_init_speed,
            init_speed_kwargs=particle_init_speed_kwargs)
        for size in np.repeat(sizes, div_size)]

    return particles


def generateParticles(
        population_size: int,
        num_features: int,
        particle_init_position: str = 'random',
        particle_init_speed: str = 'random',
        particle_init_position_kwargs: dict = None,
        particle_init_speed_kwargs: dict = None,
        seed: int = None) -> List[Particle]:
    """ Standard swarm initialization """
    if seed is not None:
        np.random.seed(seed)

    # initialize particles
    particles = [
        Particle(
            size=num_features,
            init_position=particle_init_position,
            init_position_kwargs=particle_init_position_kwargs,
            init_speed=particle_init_speed,
            init_speed_kwargs=particle_init_speed_kwargs)
        for _ in range(population_size)]

    return particles


def evaluateBinary(
        particles: List[Particle],
        fitness_function: pea2_base.FitnessStrategy,
        threshold: float = 0.6,
        invert_objective: bool = False,
        n_jobs: int = None) -> List[Particle]:
    """ Evaluate the particle's fitness
    invert_objective if True apply fitness = -1 * fitness
    """
    def __worker__(
            _index: int,
            _features: list or np.ndarray,
            _invert_objective: str,
            _fitness: pea2_base.FitnessStrategy) -> Tuple[int, float]:
        """ Evaluation subroutine. """
        assert isinstance(_features, (list, np.ndarray))
        assert issubclass(type(_fitness), pea2_base.FitnessStrategy)

        if isinstance(_features, list):
            _features = np.array(_features)

        # penalize when no feature is selected
        if np.all(_features == 0):
            return _index, -np.inf

        # get number of features
        _num_feats = _fitness.getNumFeatures()

        # pad right-side features (allow variable length particles)
        if len(_features) < _num_feats:
            _features = np.append(_features, np.zeros(_num_feats - len(_features)))

        assert len(_features) == _num_feats, 'Check'

        # evaluate features cost
        _fit_val = _fitness(_features)

        # when a cross validation scheme is applied in the fitness function a tuple is returned,
        # consider only the first value of the cost function
        if isinstance(_fit_val, tuple):
            _fit_val = _fit_val[0]

        if _invert_objective:
            _fit_val = -1 * _fit_val

        return _index, _fit_val

    # determine the number of jobs used to evaluate the particles
    if n_jobs is None:
        n_jobs = 1
    elif n_jobs == -1:
        n_jobs = mp.cpu_count()
    elif n_jobs <= 0:
        raise TypeError('"n_jobs" cannot be 0 or less than -1')

    if n_jobs == 1:
        fitness_values = dict([
            __worker__(
                _index=i,
                _features=(particle.position > threshold).astype(int),
                _invert_objective=invert_objective,
                _fitness=fitness_function)
            for i, particle in enumerate(particles)
        ])
    else:
        fitness_values = dict(
            joblib.Parallel(n_jobs=n_jobs, backend='loky')(
                joblib.delayed(__worker__)(
                    _index=i,
                    _features=(particle.position > threshold).astype(int),
                    _invert_objective=invert_objective,
                    _fitness=fitness_function)
                for i, particle in enumerate(particles)))

    # sort fitness values
    fitness_values = [fitness_values[k] for k in sorted(fitness_values.keys())]

    # assign fitness values to particles
    for particle, fitness in zip(particles, fitness_values):
        particle.prev_fitness_value = particle.curr_fitness_value
        particle.curr_fitness_value = fitness
        # for the first evaluation select the current fitness value as the best fitness value
        if particle.best_fitness_value is None:
            particle.best_fitness_value = fitness

    return particles


def vanillaReferencesUpdate(
        particles: List[Particle],
        particle_init_position: str = 'random',
        particle_init_position_kwargs: dict = {},
        **_) -> List[Particle]:
    """ Update particle's pbest and gbest based on the fitness function.
    The particles need to have been previously evaluated and therefore the curr_fitness_value must not be None.
    """
    gbest_fitness = max([p.gbest_fitness_value for p in particles])  # the algorithm will perform an optimization
    gbest = None
    for particle in particles:
        assert isinstance(particle, Particle)
        assert particle.curr_fitness_value is not None, \
            'Before updating the reference values, the particle must have been previously evaluated.'

        # if the current fitness value is greater than the best fitness value update the best position
        if particle.curr_fitness_value > particle.best_fitness_value:
            particle.pbest = deepcopy(particle.position)
            particle.best_fitness_value = particle.curr_fitness_value
            particle.pbest_count = 0   # update pbest changing count
        elif particle.curr_fitness_value == -np.inf:
            # duplicate particle
            valid_particles = [p for p in particles if len(p) == len(particle) and p.curr_fitness_value != -np.inf]

            if len(valid_particles) > 0:
                particle.position = deepcopy(random.choice(valid_particles).position)
            else:  # all particles contain invalid values
                particle.position = Particle.VALID_PARTICLE_INIT[particle_init_position](
                    size=len(particle), **particle_init_position_kwargs)
        else:
            particle.pbest_count += 1  # update pbest changing count

        # calculate gbest
        if particle.best_fitness_value > gbest_fitness:
            gbest_fitness = particle.best_fitness_value
            gbest = deepcopy(particle.pbest)

    # update particles gbest
    for particle in particles:
        if gbest is not None:
            particle.gbest = gbest
            particle.gbest_fitness_value = gbest_fitness
            particle.gbest_count = 0
        else:
            particle.gbest_count += 1

    return particles


# TODO. Parallelization
def exemplarAssignment(particles: List[Particle]):
    """ Exemplar assignment based on the VLPSO method """
    for i, p in enumerate(particles):
        if p.renew_exemplar:
            p_length = len(p)

            for d in range(p_length):
                if random.uniform(0, 1) < p.learning_prob:
                    # randomly pick a particle that is different from i and has a length longer than d;
                    valid_length_particles = [
                        ii for ii, pp in enumerate(particles) if ii != i and len(pp) >= p_length]
                    rdn_idx_p1 = random.choice(valid_length_particles)
                    p1 = particles[rdn_idx_p1]
                    valid_length_particles.pop(valid_length_particles.index(rdn_idx_p1))
                    p2 = particles[random.choice(valid_length_particles)]
                    if p1.best_fitness_value >= p2.best_fitness_value:
                        p.exemplar[d] = p1.pbest[d]
                    else:
                        p.exemplar[d] = p2.pbest[d]
                else:
                    p.exemplar[d] = p.pbest[d]

            p.renew_exemplar = False

    return particles


# TODO. Parallelization
def lengthChanging(
        particles: List[Particle],
        particle_init_position: str = 'random',
        particle_init_speed: str = 'random',
        particle_init_position_kw: dict = None,
        particle_init_speed_kw: dict = None
) -> List[Particle]:
    """ Length changing procedure as described in algorithm 2 of VLPSO. """
    # calculate the average fitness value per division
    fitness_per_division = defaultdict(list)
    particles_by_division = defaultdict(list)
    for p in particles:
        fitness_per_division[len(p)].append(p.best_fitness_value)
        particles_by_division[len(p)].append(p)
    fitness_per_division = {k: np.mean(v) for k, v in fitness_per_division.items()}

    # get the number of divisions of the current swarm
    nbr_div = len(fitness_per_division)
    # get the max length of particles in the current swarm
    max_len = max(fitness_per_division.keys())
    # particles' length of the best division
    best_len = max(fitness_per_division, key=lambda v: fitness_per_division[v])
    # sort the different divisions
    divisions = sorted(list(fitness_per_division.keys()))

    new_particles = []
    if best_len != max_len:
        k = 1
        for division in divisions:
            if division != best_len:
                new_len = int(np.ceil(best_len * k / nbr_div))
                # update particles in the current division
                for p in particles_by_division[division]:
                    if len(p) < new_len:
                        # append more dimensions to particles in division v to have
                        # new_len dimensions
                        p.position = np.append(p.position,
                                               Particle.VALID_PARTICLE_INIT[particle_init_position](
                                                   size=(new_len - len(p)), **particle_init_position_kw))
                        p.speed = np.append(p.speed,
                                            Particle.VALID_PARTICLE_INIT[particle_init_speed](
                                                   size=(new_len - len(p)), **particle_init_speed_kw))
                        p.pbest = np.append(p.pbest,
                                            Particle.VALID_PARTICLE_INIT[particle_init_position](
                                                size=(new_len - len(p)), **particle_init_position_kw))
                        p.exemplar = np.append(p.exemplar,
                                               Particle.VALID_PARTICLE_INIT[particle_init_speed](
                                                   size=(new_len - len(p)), **particle_init_speed_kw))
                        p.pbest_count = 0
                        p.gbest_count = 0

                    elif len(p) > new_len:
                        # remove the last dimensions of particles in division to have
                        # new_len dimensions
                        p.position = p.position[:new_len]
                        p.speed = p.speed[:new_len]
                        p.pbest = p.pbest[:new_len]
                        p.exemplar = p.exemplar[:new_len]
                        p.pbest_count = 0
                        p.gbest_count = 0
                    else:
                        p.pbest_count = 0
                        p.gbest_count = 0

                    new_particles.append(p)
                k += 1
            else:
                for p in particles_by_division[division]:
                    p.pbest_count = 0
                    p.gbest_count = 0
                    new_particles.append(p)
    else:
        for p in particles:
            p.pbest_count = 0
            p.gbest_count = 0
        new_particles = particles

    return new_particles



