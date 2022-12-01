import multiprocessing as mp
from abc import ABCMeta, abstractmethod


class BaseWrapper(object):
    __metaclass__ = ABCMeta

    def __getitem__(self, key: str):
        try:
            return self.getParams()[key]
        except KeyError as kerr:
            raise KeyError('The method get_params() not contain "{}"'.format(key))
        except Exception as ex:
            raise ex

    @abstractmethod
    def getParams(self) -> dict:
        """ DESCRIPTION """
        raise NotImplementedError

    @abstractmethod
    def getToolbox(self) -> dict:
        """ DESCRIPTION """
        raise NotImplementedError

    @abstractmethod
    def _register(self):
        """ DESCRIPTION """
        raise NotImplementedError

    def multiprocessing(self, n_jobs: int):
        pool = mp.Pool(n_jobs)
        toolbox = self.getToolbox()
        toolbox.register('map', pool.map)

    def clearToolbox(self):
        """ DESCRIPTION """
        toolbox = self.getToolbox()
        implemented = list(toolbox.__dict__.keys())
        for key in implemented:
            if key not in ['clone', 'map']:
                del toolbox.__dict__[key]


class FitnessStrategy(object):
    """ DESCRIPTION
    todo. convert to abstract class
    """
    __metaclass__ = ABCMeta

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return self.__repr__()

    @abstractmethod
    def __call__(self, **kwargs) -> tuple:
        # todo. convert to abstract method
        raise NotImplementedError


