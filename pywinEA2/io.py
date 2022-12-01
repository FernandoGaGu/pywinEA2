# Module containing the elements necessary to serialise objects.
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org
#
import pickle
import os
import traceback
from datetime import datetime


def save(obj: object, path: str, extension: str = None):
    """
    Method to serialize any type of object received as an argument.
    Parameters
    ----------
    obj: object
        Object to be serialized.
    path: str
        Path where the object will be saved.
    extension: str (default None)
        Extension to be added.
    """
    abs_path, file_name = os.path.split(os.path.abspath(path))

    if os.path.exists(os.path.join(abs_path, file_name)):  # If exists update the name
        file_name = '{}_{}'.format(datetime.now().strftime("%m-%d-%Y-%H-%M-%S"), file_name)
        print('File already exists, name changed to "{}"'.format(os.path.join(abs_path, file_name)))

    # Add extension
    if extension is not None:
        file_name = '{}.{}'.format(file_name, extension)

    file_abs_path = os.path.join(abs_path, file_name)

    with open(file_abs_path, 'ab') as out:
        pickle.dump(obj, out)

    print('{} object serialised correctly in "{}'.format(type(obj), file_abs_path))


def load(path: str) -> object:
    """
    Method to deserialize the object stored in the file received as argument.
    Parameters
    ----------
    :param path: str
        Path where the object was saved.
    """
    abs_path = os.path.abspath(path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError('File "{}" not found'.format(abs_path))
    try:
        with open(path, 'rb') as input_file:
            obj = pickle.load(input_file)

        return obj
    except pickle.UnpicklingError as e:
        print(traceback.format_exc(e))
        raise
    except (AttributeError, EOFError, ImportError, IndexError) as e:
        print(traceback.format_exc(e))
        raise
    except Exception as e:
        print(traceback.format_exc(e))
        raise


def addTimePrefix(s: str) -> str:
    return '{}_{}'.format(datetime.now().strftime('%Y%m%d'), s)
