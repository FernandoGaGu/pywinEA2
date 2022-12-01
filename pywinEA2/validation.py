# Module containing functions used to validate input parameters
#
# Author: Fernando García Gutiérrez
# Email: fgarcia@fundacioace.org

def checkInputType(input_var_name: str, input_var: object, valid_types: list or tuple):
    """ Check if the input variable is in the allowed types ("valid_types"). """
    if not isinstance(input_var_name, str):
        raise TypeError(f'input_var_name must be a string. Provided {str(input_var_name)}')
    if not isinstance(valid_types, (tuple, list)):
        raise TypeError(f'valid_types must be a list or a tuple. Provided {str(valid_types)}')

    if not isinstance(input_var, tuple(valid_types)):
        raise TypeError(
            f'Input variable "{input_var_name}" of type {str(type(input_var))} not in available types: '
            f'{",".join([str(v) for v in valid_types])}.')


def checkMultiInputTypes(*args):
    """ Wrapper of "checkInputType" for checking multiple input parameters. """
    for element in args:
        if not ((isinstance(element, tuple) or isinstance(element, list)) and len(element) == 3):
            raise TypeError('The arguments of this function must consist of tuples or lists of three arguments '
                            'following the signature of the adhoc.utils.checkInputType() function.')

        checkInputType(*element)


def checkImplementation(input_var_name: str, obj: object, attr: str, is_callable: bool = False):
    """ Check if an input instance implements an attribute. """
    checkMultiInputTypes(
        ('input_var_name', input_var_name, [str]),
        ('attr', attr, [str]),
        ('is_callable', is_callable, [bool]))

    if getattr(obj, attr, None) is None:
        raise TypeError(f'The method or attribute {attr} is not accessible for variable {input_var_name}')

    if is_callable and not callable(getattr(obj, attr, None)):
        raise TypeError(f'The method or attribute {attr} for {input_var_name} must be a callable.')


def inputParamUnitaryTest(completed: bool, msg: str):
    if not completed:
        raise TypeError(msg)










