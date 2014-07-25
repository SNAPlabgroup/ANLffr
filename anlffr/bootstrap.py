# -*- coding: utf-8 -*-
"""
Bootstrap helper functions for FFR data

@author: Lenny Varghese
"""
from __future__ import print_function
import numpy as np
import time
import multiprocessing
import errno

def bootfunc(inputFunction, x, params, verbose = None):
    """
    Performs bootstrapping over trials for spectral functions, utilizing
    multiple cores (threads) for a speed increase. Designed to work with the
    spectral analysis functions provided in ANLFFR, but in theory should work
    with any function that takes 3D numpy arrays and a suitable params
    structure as inputs.

    Inputs
    ------
    inputFunction - Function to run on x (the actual function, not the string
    name of the function). Function must take numpy 3D array of form chan x rep
    x time as the first argument and a paramater dictionary as the second
    argument. Function must return a dictionary.

    x - a 3D numpy array or list/tuple of 3D numpy arrays (chan x rep x time).
    If list/tuple with N elements, an equal number of trial repetitions from
    each array in the list will be used in each pass of the computation; i.e.,
    results will be computed by combining nPerDraw/N trials from per each pool
    bootstrap repetition. The obvious use case is to sample an equal number of
    trials from positive and negative polarity FFR trials.

    params - dictionary of parameters. The following fields are required for
    boostrapping, but others may be required based on the specifics of
    inputFunction:

          params['nDraws'] - number of draws for bootstrapping

          params['nPerDraw'] - number of trials per draw for bootstrapping

          params['threads'] - number of threads to spawn; only really useful if
          a multi-core CPU is available.

    Returns
    -------
    out - Dictionary of dictionaries. Level 1 keys corrspond to
    function outputs, level 2 keys correspond to individual draw results, mean
    of draw results, and variance of draw results for each key in level 1.

    Example usage
    -------
    from anlffr import bootstrap
    resultsDict = bootstrap.bootfunc(spectral.mtcpca_complete,
        [positivePolarityData, negativePolarityData], params)

    Notes
    -------
    This is not implemented in a particularly clever or elegant way. But it
    does provide roughly a factor of N speedup (where N is the number of CPU
    cores dedicated to this task). The data array is copied in memory for each
    core that is utilized, and will thus eat quite a bit of RAM when more than
    1-2 threads are spawned. RAM is cheap, so this shouldn't be too much of a
    concern in well funded labs.  But if you want something more memory
    efficient, go code it yourself.

    Tested using Debian 7, Python 2.7.3 virtual environment with numpy 1.8.1,
    scipy 0.13.3, nitime 0.5. Code should be platform independent if
    dependencies are satistied, but no effort has gone into checking.

    Last updated: 7/25/2014
    Auditory Neuroscience Laboratory, Boston University
    Contact: lennyv@bu.edu
    """

    _validate_bootstrap_params(params)

    startTime = time.time()
    theQueue = multiprocessing.Queue()
    results = {}
    output = {}

    print('Using {} threads...'.format(params['threads']))

    # split the loads roughly equally:
    drawSplit = _compute_thread_split(params)

    processList = []
    for proc in range(params['threads']):
        if ('debugMode' in params) and (params['debugMode']):
            print('setting fixed random seeds!')
            randomState = np.random.RandomState(proc)
        else:
            randomState = np.random.RandomState()

        processList.append(multiprocessing.Process(target = _multiprocess_wrapper,
                           args = (inputFunction, x, params, drawSplit[proc], theQueue,
                                   randomState)))
        processList[proc].start()

    numRetrieved = 0
    trialsUsed = []

    while numRetrieved < params['nDraws']:
        try:
            retrievedData = theQueue.get(True)
            numRetrieved = numRetrieved + 1

            for k in retrievedData[0].keys():
                if 1 == numRetrieved:
                    results[k] = dict(runningSum = 0, runningSS = 0, indivDraw = [])

                results[k]['indivDraw'].append(retrievedData[0][k])
                results[k]['runningSum'] += retrievedData[0][k]
                results[k]['runningSS'] += retrievedData[0][k]**2

            trialsUsed.append(retrievedData[1])

        # the following should be OK, as per:
        # http://stackoverflow.com/questions/
        #        4952247/interrupted-system-call-with-processing-queue
        except IOError, e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise

    for k in results.keys():
        output[k] = {}
        output[k]['nDraws'] = int(params['nDraws'])
        output[k]['nPerDraw'] = int(params['nPerDraw'])
        output[k]['indivDraw'] = np.array(results[k]['indivDraw'])
        output[k]['bootMean'] = results[k]['runningSum'] / params['nDraws']
        output[k]['bootVariance'] = _compute_variance(output[k]['mean'],
                                                      results[k]['runningSS'],
                                                      params['nDraws'])
    output['trialsUsed'] = list(trialsUsed)
    
    print('Completed in: {} s'.format(time.time() - startTime))

    return output

def _multiprocess_wrapper(inputFunction, inputData, params, nDraws, resultsQueue,
        randomState):
    """
    internal function. places results from spectral functions in queue.
    """
    # set random seed here, otherwise this might draw the same pool
    # of trials across all processes

    errorString = 'Data should be 3D numpy array or list/tuple of 3D numpy arrays'
    # allows you to specify a list of data from each polarity (or other things you want to
    # combine across) so that the number of trials going into a computation from different
    # sources can be fixed
    if type(inputData) == list or type(inputData) == tuple:
        for x in inputData:
            if type(x) != np.ndarray and x.ndim !=3:
                raise TypeError(errorString)
    elif type(inputData) == np.ndarray:
        inputData = [inputData]
    else:
        raise TypeError(errorString)

    for _ in range(nDraws):
        theseData, trialsUsed = _combine_random_trials(inputData, 
                                                       params['nPerDraw'],
                                                       randomState)
        out = (inputFunction(theseData, params, verbose = False), trialsUsed)
        resultsQueue.put(out)

def _compute_variance(dataMean, dataSumOfSquares, n):
    """
     internal function. computes variance from running SS, mean, and n.
    """
    return (dataSumOfSquares - ((dataMean*n)**2) / n ) / (n-1)

def _combine_random_trials(inputData, nPerDraw, randomState = None):
    """
    internal function. creates a new data array from a series of randomly 
    sampled old ones. random sample is with replacement.
    """

    assert type(inputData) == list, 'inputData should be a list of arrays'

    if randomState is None:
        randomState = np.random.RandomState()

    warnString = ('warning: number of trials requested in draw ' +
        '> trials available for pool {}')
    numPools = len(inputData)
    useTrialsPerPool = int(nPerDraw) / numPools

    tempData = []
    pickTrials = []

    print('\n\nChoosing trials...\n\n ')
    for pool in range(numPools):
        if useTrialsPerPool > inputData[pool].shape[1]:
            print(warnString.format(pool))

        pickTrials.append(randomState.randint(0,inputData[pool].shape[1],useTrialsPerPool))
        tempData.append(inputData[pool][:,pickTrials[-1],:])

    useData = np.concatenate(tuple(tempData), axis = 1)

    return useData, pickTrials

def _compute_thread_split(params):
    """
    internal function. computes how many draws each process will take on.

    """

    drawSplit = []
    distribute = int(params['nDraws']) / int(params['threads'])
    leftover = int(params['nDraws']) % int(params['threads'])
    for _ in range(params['threads']):
        drawSplit.append(distribute)
    for r in range(leftover):
        drawSplit[r] = drawSplit[r] + 1

    return drawSplit

def _validate_bootstrap_params(params):
    """
    internal function. checks parameters required for bootfunc.
    """
    if ('bootstrap_params_validated' not in params) or (params['bootstrap_params_validated']
            is False):

        # check/fix bootStrap input

        if ('nDraw' in params) or ('nPerDraw' in params):
            assert 'nDraws' in params, ('when params[''nPerDraw''] is specified, ' +
                    'params[''nDraw''] must be too')
            assert 'nPerDraw' in params, ('when params[''nDraws''] is specified, ' +
                    'params[''nPerDraw''] must be too')

            assert params['nDraws'] == int(params['nDraws']), ('params[''nDraws''] ' +
                'must be an integer')
            assert params['nPerDraw'] == int(params['nPerDraw']), ('params[''nPerDraw''] ' +
                'must be an integer')

            assert params['nDraws'] > 0, 'params[''nDraws''] must be positive'
            assert params['nPerDraw'] > 0, 'params[''nPerDraw''] must be positive'

        if 'threads' in params:
            numCpu = multiprocessing.cpu_count()
            assert params['threads'] == int(params['threads']), ('params[''threads''] must be an ' +
                'integer')
            assert 0 < params['threads'], 'params[''threads''] should be > 0'
            assert params['threads'] <= numCpu, ('params[''threads''] should be ' +
                ' <= {}'.format(numCpu))

        params['bootstrap_params_validated'] = True

    return params
