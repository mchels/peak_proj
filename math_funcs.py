import inspect
import numpy as np

def sum_of_func(func, x, *params):
    """
    Returns a sum of the function func over the parameters params applied to
    the x-values in x.

    Arguments:
    func: function
        Function to sum. func must take a numpy array as the first argument
        and a variable number of parameters as the following arguments.
        It must Not take any keyword arguments.
    x: numpy array
        x-values to apply func to.
    params: list
        List of parameters to use when applying func to x. The length of
        params defines how many funcs are summed so it must be divisible
        by the number of parameters of func.
    """
    n_supplied_params = len(params)
    args_for_func = inspect.getargspec(func).args
    # Subtract 1 because x is not a parameter.
    n_params_for_func = len(args_for_func) - 1
    assert np.isclose(n_supplied_params % n_params_for_func, 0)
    y = np.zeros_like(x, dtype=float)
    i = 0
    while i < n_supplied_params:
        params_to_use = params[i:i+n_params_for_func]
        y += func(x, *params_to_use)
        i += n_params_for_func
    return y

def lorentzian(x, x0, gamma, height, baseline):
    numerator = height
    denominator = 1 + (x-x0)**2/(gamma/2)**2
    y = numerator/denominator + baseline
    return y

def lorentzian_no_base(x, x0, gamma, height):
    return lorentzian(x, x0, gamma, height, 0)

def polynomial(x, *args):
    """
    Evaluate a polynomial on the values in x and return the result as a list.
    The degree of the polynomial is given by the number of variable arguments,
    so that degree = len(args) - 1.
    The arguments are in order of increasing degree, i.e. the first element
    is the coefficient of x^0.
    """
    assert len(args) != 0
    y = np.zeros_like(x, dtype=float)
    for i, parameter in enumerate(args):
        assert np.isfinite(parameter)
        exponent = i
        y += parameter * x**exponent
    return y
