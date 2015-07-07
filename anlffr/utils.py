"""
Utility functions for the ANLffr package

@author: Hari Bharadwaj
"""
import warnings
import logging
import inspect
import sys
from functools import wraps

from .externals.decorator import decorator as markDecorator

__all__ = ['logger', 'verbose', 'deprecated', 'set_log_level']

logger = logging.getLogger('anlffr')  # Used across all code
logger.propagate = False  # What to do in case of multiple imports
logger.addHandler(logging.StreamHandler(sys.stdout))


# force show of DeprecationWarning even on python 2.7
warnings.simplefilter('default')


class deprecated(object):
    """Decorator to mark a function or class as deprecated.

    Issue a warning when the function is called/the class is instantiated and
    adds a warning to the docstring.

    The optional extra argument will be appended to the deprecation message
    and the docstring. Note: to use this with the default value for extra, put
    in an empty of parentheses:

    >>> from mne.utils import deprecated
    >>> deprecated() # doctest: +ELLIPSIS
    <anlffr.utils.deprecated object at ...>

    >>> @deprecated()
    ... def some_function(): pass

    """
    # Adapted from http://wiki.python.org/moin/PythonDecoratorLibrary,
    # but with many changes.

    def __init__(self, extra=''):
        """
        Parameters
        ----------
        extra: string
          to be added to the deprecation messages

        """
        self.extra = extra

    def __call__(self, obj):
        if isinstance(obj, type):
            return self._decorate_class(obj)
        else:
            return self._decorate_fun(obj)

    def _decorate_class(self, cls):
        msg = "Class %s is deprecated" % cls.__name__
        if self.extra:
            msg += "; %s" % self.extra

        # FIXME: we should probably reset __new__ for full generality
        init = cls.__init__

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return init(*args, **kwargs)
        cls.__init__ = wrapped

        wrapped.__name__ = '__init__'
        wrapped.__doc__ = self._update_doc(init.__doc__)
        wrapped.deprecated_original = init

        return cls

    def _decorate_fun(self, fun):
        """Decorate function fun"""

        msg = "Function %s is deprecated" % fun.__name__
        if self.extra:
            msg += "; %s" % self.extra

        def wrapped(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning)
            return fun(*args, **kwargs)

        wrapped.__name__ = fun.__name__
        wrapped.__dict__ = fun.__dict__
        wrapped.__doc__ = self._update_doc(fun.__doc__)

        return wrapped

    def _update_doc(self, olddoc):
        newdoc = "DEPRECATED"
        if self.extra:
            newdoc = "%s: %s" % (newdoc, self.extra)
        if olddoc:
            newdoc = "%s\n\n%s" % (newdoc, olddoc)
        return newdoc


@markDecorator
def verbose(function, *args, **kwargs):
    """Improved verbose decorator to allow functions to override log-level

    Do not call this directly to set global verbosrity level, instead use
    set_log_level().

    Parameters
    ----------
    function - function
        Function to be decorated to allow for overriding global verbosity
        level

    Returns
    -------

    dec - function
        The decorated function

    """
    arg_names = inspect.getargspec(function).args

    if len(arg_names) > 0 and arg_names[0] == 'self':
        default_level = getattr(args[0], 'verbose', None)
    else:
        default_level = None

    if('verbose' in arg_names):
        verbose_level = args[arg_names.index('verbose')]
    else:
        verbose_level = default_level

    if verbose_level is not None:
        old_level = set_log_level(verbose_level, True)
        # set it back if we get an exception
        try:
            ret = function(*args, **kwargs)
        except:
            set_log_level(old_level)
            raise
        set_log_level(old_level)
        return ret
    else:
        ret = function(*args, **kwargs)
        return ret


@deprecated('This is DEPRECATED since the inclusion of the decorator module.'
            '\nUse of this decorator fetches the right docstring\n'
            'but not the signature.')
def verbose_old(function):
    """Decorator to allow functions to override default log level

    Do not call this function directly to set the global verbosity level,
    instead use set_log_level().


    Parameters
    ----------

    function - function
        The function to be decorated to allow for overriding global verbosity
        level
    Returns
    -------

    dec - function
        The decorated function

    """
    arg_names = inspect.getargspec(function).args
    # this wrap allows decorated functions to be pickled

    @wraps(function)
    def dec(*args, **kwargs):
        # Check if the first arg is "self", if it has verbose, make it default
        if len(arg_names) > 0 and arg_names[0] == 'self':
            default_level = getattr(args[0], 'verbose', None)
        else:
            default_level = None
        verbose_level = kwargs.get('verbose', default_level)
        if verbose_level is not None:
            old_level = set_log_level(verbose_level, True)
            # set it back if we get an exception
            try:
                ret = function(*args, **kwargs)
            except:
                set_log_level(old_level)
                raise
            set_log_level(old_level)
            return ret
        else:
            return function(*args, **kwargs)

    # set __wrapped__ attribute so ?? in IPython gets the right source
    dec.__wrapped__ = function

    return dec


def set_log_level(verbose=None, return_old_level=False):
    """Convenience function for setting the logging level

    Parameters
    ----------

    verbose - bool, str, int, or None
        The verbosity of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        If None, defaults to WARNING.
    return_old_level - bool
        If True, return the old verbosity level.

    Returns
    -------

    old_verbose - Old Verbosity Level
        Returned if return_old_level is True

    """
    if verbose is None:
        verbose = 'WARNING'
    elif isinstance(verbose, bool):
        if verbose is True:
            verbose = 'INFO'
        else:
            verbose = 'WARNING'
    if isinstance(verbose, basestring):
        verbose = verbose.upper()
        logging_types = dict(DEBUG=logging.DEBUG, INFO=logging.INFO,
                             WARNING=logging.WARNING, ERROR=logging.ERROR,
                             CRITICAL=logging.CRITICAL)
        if verbose not in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]
    logger = logging.getLogger('anlffr')
    old_verbose = logger.level
    logger.setLevel(verbose)
    return (old_verbose if return_old_level else None)
