import random
import numpy as np
from deap import algorithms
from deap import tools
from deap.benchmarks.tools import hypervolume
from functools import partial

from pywinEA2 import base as pea2_base
from pywinEA2 import validation as pea2_valid
from pywinEA2 import report as pea2_report


def run(alg: pea2_base.BaseWrapper, type: str = 'eaSimple', verbose: bool = True, hof_size: int = 1,
        **kwargs) -> pea2_report.Report:
    """ Function used to execute the algorithms.
    
    Parameters
    ----------
    :param alg: 
    :param type: str, default='eaSimple'
        
        Implemented mono-objective genetic algorithms:
            - 'eaSimple'
            - 'eaMuPlusLambda'
            - 'eaMuCommaLambda'
            - 'eaSimpleWithElitism'
            - 'eaMuCommaLambdaWithElitism'
        
        Implemented multi-objective genetic algorithms:
            - 'nsga2'

        For more information see https://deap.readthedocs.io/en/master/api/algo.html
        
    :param verbose: bool, default=True
    :param hof_size: int, default=1
        Hall of fame size.

    **kwargs
    lambda_ and mu for 'eaMuCommaLambda'
    lambda_ and mu for 'eaMuPlusLambda'

    """
    IMPLEMENTED_MONO_ALG = [  # https://deap.readthedocs.io/en/master/api/algo.html
        'eaSimple',
        'eaMuPlusLambda',
        'eaMuCommaLambda',
        'eaSimpleWithElitism',
        'eaMuCommaLambdaWithElitism',
    ]
    IMPLEMENTED_MULTI_ALG = [
        'nsga2'
    ]

    # Check input parameters
    pea2_valid.checkMultiInputTypes(
        ('alg',      alg,      [pea2_base.BaseWrapper]),
        ('type',     type,     [str]),
        ('hof_size', hof_size, [int]),
        ('verbose',  verbose,  [bool])
    )
    pea2_valid.checkImplementation('getToolbox',        alg,              'getToolbox',        True)
    pea2_valid.checkImplementation('createIndividuals', alg.getToolbox(), 'createIndividuals', True)
    pea2_valid.checkImplementation('createPopulation',  alg.getToolbox(), 'createPopulation',  True)
    pea2_valid.checkImplementation('select',            alg.getToolbox(), 'select',            True)
    pea2_valid.checkImplementation('mate',              alg.getToolbox(), 'mate',              True)
    pea2_valid.checkImplementation('mutate',            alg.getToolbox(), 'mutate',            True)
    pea2_valid.inputParamUnitaryTest('population_size' in alg.getParams(), 'Missing parameter "population_size"')
    pea2_valid.inputParamUnitaryTest('max_generations' in alg.getParams(), 'Missing parameter "max_generations"')
    pea2_valid.inputParamUnitaryTest('p_crossover'     in alg.getParams(), 'Missing parameter "p_crossover"')
    pea2_valid.inputParamUnitaryTest('p_mutation'      in alg.getParams(), 'Missing parameter "p_mutation"')
    pea2_valid.inputParamUnitaryTest(
         type in IMPLEMENTED_MONO_ALG + IMPLEMENTED_MULTI_ALG,
        'Unrecognized algorithm type "{}". Currently supported are: {}'.format(
            type, IMPLEMENTED_MONO_ALG + IMPLEMENTED_MULTI_ALG))

    if 'random_seed' in alg.getParams():
        random.seed(alg['random_seed'])
        np.random.seed(alg['random_seed'])

    # create hall of fame
    hof = tools.HallOfFame(hof_size) if hof_size is not None or hof_size > 0 else None

    if type in IMPLEMENTED_MONO_ALG:
        report = pea2_report.Report()
    elif type in IMPLEMENTED_MULTI_ALG:
        report = pea2_report.MultiObjectiveReport()
    else:
        assert False   # better prevent

    population = alg.getToolbox().createPopulation(alg['population_size'])

    if type == 'eaSimple':
        population, logbook = algorithms.eaSimple(
            population=population,
            toolbox=alg.getToolbox(),
            cxpb=alg['p_crossover'],
            mutpb=alg['p_mutation'],
            ngen=alg['max_generations'],
            stats=report.stats,
            halloffame=hof,
            verbose=verbose)
    elif type == 'eaSimpleWithElitism':
        population, logbook = eaSimpleWithElitism(
            population=population,
            toolbox=alg.getToolbox(),
            cxpb=alg['p_crossover'],
            mutpb=alg['p_mutation'],
            ngen=alg['max_generations'],
            stats=report.stats,
            halloffame=hof,
            verbose=verbose)
    elif type == 'eaMuCommaLambda':
        mu = kwargs.get('mu', len(population))             # by default mu will be equal to the population size
        lambda_ = kwargs.get('lambda_', len(population))   # by default lambda_ will be equal to the population size
        population, logbook = algorithms.eaMuCommaLambda(
            population=population,
            toolbox=alg.getToolbox(),
            mu=mu,
            lambda_=lambda_,
            cxpb=alg['p_crossover'],
            mutpb=alg['p_mutation'],
            ngen=alg['max_generations'],
            stats=report.stats,
            halloffame=hof,
            verbose=verbose)
    elif type == 'eaMuCommaLambdaWithElitism':
        mu = kwargs.get('mu', len(population))             # by default mu will be equal to the population size
        lambda_ = kwargs.get('lambda_', len(population))   # by default lambda_ will be equal to the population size
        population, logbook = eaMuCommaLambdaWithElitism(
            population=population,
            toolbox=alg.getToolbox(),
            mu=mu,
            lambda_=lambda_,
            cxpb=alg['p_crossover'],
            mutpb=alg['p_mutation'],
            ngen=alg['max_generations'],
            stats=report.stats,
            halloffame=hof,
            verbose=verbose)
    elif type == 'eaMuPlusLambda':
        mu = kwargs.get('mu', len(population))             # by default mu will be equal to the population size
        lambda_ = kwargs.get('lambda_', len(population))   # by default lambda_ will be equal to the population size
        population, logbook = algorithms.eaMuCommaLambda(
            population=population,
            toolbox=alg.getToolbox(),
            mu=mu,
            lambda_=lambda_,
            cxpb=alg['p_crossover'],
            mutpb=alg['p_mutation'],
            ngen=alg['max_generations'],
            stats=report.stats,
            halloffame=hof,
            verbose=verbose)
    elif type == 'nsga2':
        compute_hypervolume = kwargs.get('compute_hypervolume', True)
        population, logbook, hypervolume_l = nsga2(
            population=population,
            toolbox=alg.getToolbox(),
            cxpb=alg['p_crossover'],
            mutpb=alg['p_mutation'],
            ngen=alg['max_generations'],
            stats=report.stats,
            halloffame=hof,
            compute_hypervolume=compute_hypervolume,
            verbose=verbose)
        report.addHypervolume(hypervolume_l)   # add hypervolume to the report
    else:
        assert False   # better prevent

    report.addPopulation(population)
    report.addLogbook(logbook)
    report.addHallOfFame(hof)
    report.addAlgorithm(alg)

    # evaluate population
    toolbox = alg.getToolbox()
    report.addFitness(toolbox.map(toolbox.evaluate, population))

    return report


def eaSimpleWithElitism(population, toolbox, cxpb, mutpb, ngen, halloffame, stats=None,
                        verbose: bool = False):
    """This algorithm is similar to DEAP eaSimple() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.

    Code from: https://github.com/PacktPublishing/Hands-On-Genetic-Algorithms-with-Python

    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - hof_size)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # add the best back to population:
        offspring.extend(halloffame.items)

        # Update the hall of fame with the generated individuals
        halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def eaMuCommaLambdaWithElitism(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, halloffame, stats=None,
                               verbose: bool = False):
    """This algorithm is similar to DEAP eaMuCommaLambda() algorithm, with the modification that
    halloffame is used to implement an elitism mechanism. The individuals contained in the
    halloffame are directly injected into the next generation and are not subject to the
    genetic operators of selection, crossover and mutation.
    """
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is None:
        raise ValueError("halloffame parameter must not be empty!")

    halloffame.update(population)
    hof_size = len(halloffame.items) if halloffame.items else 0

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        offspring = toolbox.select(offspring, mu - hof_size)
        offspring.extend(halloffame.items)
        halloffame.update(offspring)
        population[:] = offspring

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


def nsga2(population, toolbox, cxpb, mutpb, ngen, halloffame=None, stats=None, compute_hypervolume: bool = True,
          verbose: bool = False):
    """ DESCRIPTION
    code based on: https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py
    """
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # This is just to assign the crowding distance to the individuals
    # no actual selection is done
    population = toolbox.select(population, len(population))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    record = stats.compile(population)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    hypervolumes_l = []
    for gen in range(1, ngen):
        # Vary the population
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        try:
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        except Exception as ex:
            print('Exception {}'.format(ex))
            print('fitness {}'.format(fitnesses))
            print('ind {}'.format(ind))
            print('fit {}'.format(fit))

        # Select the next generation population
        population = toolbox.select(population + offspring, len(population))

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(population)

        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # compute the hypervolume
        if compute_hypervolume:
            hypervolumes_l.append(hypervolume(population))

    return population, logbook, hypervolumes_l





