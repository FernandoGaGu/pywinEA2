import numpy as np
import random


def randomInit(
        size: int,
        bounds: tuple or list or np.ndarray = None) -> np.ndarray:
    """ Random value initialization of particle positions and speed """

    # default initialization to the range [0, 1]
    bounds = (0, 1) if bounds is None else bounds

    # initialize all values in the same range
    if isinstance(bounds, tuple):  # same min-max for all features
        assert len(bounds) == 2, '"bounds" specified as tuple must contain only 2 elements.'
        assert bounds[0] < bounds[1], 'lower bound greater than upper bound.'
        init_values = np.random.uniform(low=bounds[0], high=bounds[1], size=size)

    # initialize values using a different range
    elif isinstance(bounds, (list, np.ndarray)):
        bounds = np.array(bounds) if isinstance(bounds, list) else bounds  # convert to numpy array
        assert bounds.shape[0] == size, '"bounds" specified as list or array must be of the same length than size'
        assert bounds.shape[1] == 2, 'Elements in "bounds" must be of size 2.'
        assert np.all(bounds[:, 0] < bounds[:, 1]), 'lower bound greater than upper bound.'
        init_values = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=size)

    else:
        raise TypeError(
            'Variable "bounds" must be a tuple, list or numpy array. Provided type "{}"'.format(type(bounds)))

    return init_values


class Particle(object):
    """ DESCRIPTION """
    VALID_PARTICLE_INIT = {
        'random': randomInit
    }

    def __init__(
            self,
            size: int,
            init_position: str = 'random',
            init_position_kwargs: dict = None,
            init_speed: str = 'random',
            init_speed_kwargs: dict = None,
            seed: int = None
    ):
        # check initialization method
        assert init_position in Particle.VALID_PARTICLE_INIT
        assert init_speed    in Particle.VALID_PARTICLE_INIT

        if seed is not None:
            np.random.seed(seed)

        # default parameters
        init_position_kwargs = {} if init_position_kwargs is None else init_position_kwargs
        init_speed_kwargs     = {} if init_speed_kwargs is None else init_speed_kwargs

        # initialize parameters
        self.position = Particle.VALID_PARTICLE_INIT[init_position](**{**{'size': size}, **init_position_kwargs})
        self.speed    = Particle.VALID_PARTICLE_INIT[init_speed](**{**{'size': size}, **init_speed_kwargs})

        # default parameters
        self.prev_fitness_value = -np.inf
        self.curr_fitness_value = -np.inf
        self.best_fitness_value = -np.inf
        self.gbest_fitness_value = -np.inf
        self.pbest = None
        self.gbest = None
        # parameters used for VLPSO
        self.pbest_count = 0
        self.gbest_count = 0
        self.learning_prob = None
        self.exemplar = None
        self.renew_exemplar = False
        self.rank = 0

    def __len__(self) -> int:
        return len(self.position)


def _clipParticleValues(particle: Particle, clip_values: tuple or np.ndarray) -> Particle:
    if isinstance(clip_values, tuple):  # same values for all the positions
        assert len(clip_values) == 2, 'If "clip_values" is provided as tuple, it must be a 2 element tuple'
        particle.position[particle.position < clip_values[0]] = clip_values[0]
        particle.position[particle.position > clip_values[1]] = clip_values[1]
    elif isinstance(clip_values, np.ndarray):  # position-specific values
        assert (clip_values.shape[0] == len(particle.position)) and clip_values.shape[1] == 2, \
            'If "clip_values" is provided as numpy array, it must be of shape (particle_size, 2)'
        particle.position[particle.position < clip_values[:, 0]] = clip_values[:, 0]
        particle.position[particle.position > clip_values[:, 1]] = clip_values[:, 1]
    else:
        raise TypeError(
            'Parameter "clip_values" must be a tuple or numpy array. Provided: "{}"'.format(type(clip_values)))

    return particle


def vanillaPositionUpdate(
        particle: Particle,
        inertia: float,
        acc_const1: float,
        acc_const2: float,
        clip_values: tuple or np.ndarray = None,
        seed: int = None) -> Particle:
    """ Calculate new particles speed and position using the standard PSO update rule """
    assert particle.pbest is not None, 'particle.pbest is None. Before calculate new position evaluate the particle.'
    assert particle.gbest is not None, 'particle.gbest is None. Before calculate new position evaluate the particle.'

    if seed is not None:
        random.seed(seed)

    v_t = (
        inertia * particle.speed +
        acc_const1 * random.uniform(0, 1) * (particle.pbest - particle.position) +
        acc_const2 * random.uniform(0, 1) * (particle.gbest - particle.position)
    )
    x_t = particle.position + v_t

    # update particle parameters
    particle.position = x_t
    particle.speed = v_t

    if clip_values is not None:   # select maximum and minimum values for position
        particle = _clipParticleValues(particle, clip_values)

    return particle


def clpsoPositionUpdate(
        particle: Particle,
        inertia: float,
        acc_const1: float,
        clip_values: tuple or np.ndarray = None,
        seed: int = None) -> Particle:
    """ Calculate new particles speed and position using the standard PSO update rule """
    assert particle.exemplar is not None, \
        'particle.exemplar is None. Before calculate new position evaluate the particle.'

    if seed is not None:
        random.seed(seed)

    v_t = (
        inertia * particle.speed +
        acc_const1 * random.uniform(0, 1) * (particle.exemplar - particle.position)
    )
    x_t = particle.position + v_t

    # update particle parameters
    particle.position = x_t
    particle.speed = v_t

    if clip_values is not None:   # select maximum and minimum values for position
        particle = _clipParticleValues(particle, clip_values)

    return particle


def particleContainsAttribute(
        particle: Particle,
        attr_name: str) -> bool:
    """ Function that checks if a particle contains a given attribute """
    return attr_name in particle.__dict__.keys()

