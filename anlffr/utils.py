"""
Utility functions for the ANLffr package

Created on Thu Dec  5 14:56:34 2013

@author: Hari Bharadwaj
"""
import logging, inspect, re, sys
from functools import wraps

__all__ = ['logger', 'verbose', 'set_log_level']

logger = logging.getLogger('anlffr') # Used across all code
logger.propagate = False # What to do in case of multiple imports
logger.addHandler(logging.StreamHandler(sys.stdout))


################## Code from decorator module 3.4.0 ##########################
if sys.version >= '3':
    from inspect import getfullargspec
    def get_init(cls):
        return cls.__init__
else:
    class getfullargspec(object):
        "A quick and dirty replacement for getfullargspec for Python 2.X"
        def __init__(self, f):
            self.args, self.varargs, self.varkw, self.defaults = \
                inspect.getargspec(f)
            self.kwonlyargs = []
            self.kwonlydefaults = None
        def __iter__(self):
            yield self.args
            yield self.varargs
            yield self.varkw
            yield self.defaults
    def get_init(cls):
        return cls.__init__.im_func

DEF = re.compile('\s*def\s*([_\w][_\w\d]*)\s*\(')

# basic functionality
class FunctionMaker(object):
    """
    An object with the ability to create functions with a given signature.
    It has attributes name, doc, module, signature, defaults, dict and
    methods update and make.
    """
    def __init__(self, func=None, name=None, signature=None,
                 defaults=None, doc=None, module=None, funcdict=None):
        self.shortsignature = signature
        if func:
            # func can be a class or a callable, but not an instance method
            self.name = func.__name__
            if self.name == '<lambda>': # small hack for lambda functions
                self.name = '_lambda_' 
            self.doc = func.__doc__
            self.module = func.__module__
            if inspect.isfunction(func):
                argspec = getfullargspec(func)
                self.annotations = getattr(func, '__annotations__', {})
                for a in ('args', 'varargs', 'varkw', 'defaults', 'kwonlyargs',
                          'kwonlydefaults'):
                    setattr(self, a, getattr(argspec, a))
                for i, arg in enumerate(self.args):
                    setattr(self, 'arg%d' % i, arg)
                if sys.version < '3': # easy way
                    self.shortsignature = self.signature = \
                        inspect.formatargspec(
                        formatvalue=lambda val: "", *argspec)[1:-1]
                else: # Python 3 way
                    allargs = list(self.args)
                    allshortargs = list(self.args)
                    if self.varargs:
                        allargs.append('*' + self.varargs)
                        allshortargs.append('*' + self.varargs)
                    elif self.kwonlyargs:
                        allargs.append('*') # single star syntax
                    for a in self.kwonlyargs:
                        allargs.append('%s=None' % a)
                        allshortargs.append('%s=%s' % (a, a))
                    if self.varkw:
                        allargs.append('**' + self.varkw)
                        allshortargs.append('**' + self.varkw)
                    self.signature = ', '.join(allargs)
                    self.shortsignature = ', '.join(allshortargs)
                self.dict = func.__dict__.copy()
        # func=None happens when decorating a caller
        if name:
            self.name = name
        if signature is not None:
            self.signature = signature
        if defaults:
            self.defaults = defaults
        if doc:
            self.doc = doc
        if module:
            self.module = module
        if funcdict:
            self.dict = funcdict
        # check existence required attributes
        assert hasattr(self, 'name')
        if not hasattr(self, 'signature'):
            raise TypeError('You are decorating a non function: %s' % func)

    def update(self, func, **kw):
        "Update the signature of func with the data in self"
        func.__name__ = self.name
        func.__doc__ = getattr(self, 'doc', None)
        func.__dict__ = getattr(self, 'dict', {})
        func.func_defaults = getattr(self, 'defaults', ())
        func.__kwdefaults__ = getattr(self, 'kwonlydefaults', None)
        func.__annotations__ = getattr(self, 'annotations', None)
        callermodule = sys._getframe(3).f_globals.get('__name__', '?')
        func.__module__ = getattr(self, 'module', callermodule)
        func.__dict__.update(kw)

    def make(self, src_templ, evaldict=None, addsource=False, **attrs):
        "Make a new function from a given template and update the signature"
        src = src_templ % vars(self) # expand name and signature
        evaldict = evaldict or {}
        mo = DEF.match(src)
        if mo is None:
            raise SyntaxError('not a valid function template\n%s' % src)
        name = mo.group(1) # extract the function name
        names = set([name] + [arg.strip(' *') for arg in 
                             self.shortsignature.split(',')])
        for n in names:
            if n in ('_func_', '_call_'):
                raise NameError('%s is overridden in\n%s' % (n, src))
        if not src.endswith('\n'): # add a newline just for safety
            src += '\n' # this is needed in old versions of Python
        try:
            code = compile(src, '<string>', 'single')
            # print >> sys.stderr, 'Compiling %s' % src
            exec code in evaldict
        except:
            print >> sys.stderr, 'Error in generated code:'
            print >> sys.stderr, src
            raise
        func = evaldict[name]
        if addsource:
            attrs['__source__'] = src
        self.update(func, **attrs)
        return func

    @classmethod
    def create(cls, obj, body, evaldict, defaults=None,
               doc=None, module=None, addsource=True, **attrs):
        """
        Create a function from the strings name, signature and body.
        evaldict is the evaluation dictionary. If addsource is true an attribute
        __source__ is added to the result. The attributes attrs are added,
        if any.
        """
        if isinstance(obj, str): # "name(signature)"
            name, rest = obj.strip().split('(', 1)
            signature = rest[:-1] #strip a right parens            
            func = None
        else: # a function
            name = None
            signature = None
            func = obj
        self = cls(func, name, signature, defaults, doc, module)
        ibody = '\n'.join('    ' + line for line in body.splitlines())
        return self.make('def %(name)s(%(signature)s):\n' + ibody, 
                        evaldict, addsource, **attrs)
  
def decorator(caller, func=None):
    """
    decorator(caller) converts a caller function into a decorator;
    decorator(caller, func) decorates a function using a caller.
    """
    if func is not None: # returns a decorated function
        evaldict = func.func_globals.copy()
        evaldict['_call_'] = caller
        evaldict['_func_'] = func
        return FunctionMaker.create(
            func, "return _call_(_func_, %(shortsignature)s)",
            evaldict, undecorated=func, __wrapped__=func)
    else: # returns a decorator
        if inspect.isclass(caller):
            name = caller.__name__.lower()
            callerfunc = get_init(caller)
            doc = 'decorator(%s) converts functions/generators into ' \
                'factories of %s objects' % (caller.__name__, caller.__name__)
            fun = getfullargspec(callerfunc).args[1] # second arg
        elif inspect.isfunction(caller):
            name = '_lambda_' if caller.__name__ == '<lambda>' \
                else caller.__name__
            callerfunc = caller
            doc = caller.__doc__
            fun = getfullargspec(callerfunc).args[0] # first arg
        else: # assume caller is an object with a __call__ method
            name = caller.__class__.__name__.lower()
            callerfunc = caller.__call__.im_func
            doc = caller.__call__.__doc__
            fun = getfullargspec(callerfunc).args[1] # second arg
        evaldict = callerfunc.func_globals.copy()
        evaldict['_call_'] = caller
        evaldict['decorator'] = decorator
        return FunctionMaker.create(
            '%s(%s)' % (name, fun), 
            'return decorator(_call_, %s)' % fun,
            evaldict, undecorated=caller, __wrapped__=caller,
            doc=doc, module=caller.__module__)


@decorator
def verbose(function, *args, **kwargs):
    """Improved verbose decorator to allow functions to override log-level
    
    Do not call this directly to set global verbosrity level, instead use
    set_log_level().
    
    Parameters
    ----------
    function - function
        Function to be decorated by setting the verbosity level.
    
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
        verbose_level =  default_level
        
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
        
        
def verbose_old(function):
    """Decorator to allow functions to override default log level

    Do not call this function directly to set the global verbosity level,
    instead use set_log_level().
    
    This is DEPRECATED since the inclusion of the decorator module

    Parameters
    ----------
    
    verbose - bool, str, int, or None
        The level of messages to print. If a str, it can be either DEBUG,
        INFO, WARNING, ERROR, or CRITICAL. Note that these are for
        convenience and are equivalent to passing in logging.DEBUG, etc.
        For bool, True is the same as 'INFO', False is the same as 'WARNING'.
        None defaults to using the current log level [e.g., set using
        mne.set_log_level()].
        
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
        if not verbose in logging_types:
            raise ValueError('verbose must be of a valid type')
        verbose = logging_types[verbose]
    logger = logging.getLogger('anlffr')
    old_verbose = logger.level
    logger.setLevel(verbose)
    return (old_verbose if return_old_level else None)


