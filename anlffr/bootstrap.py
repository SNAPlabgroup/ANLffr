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
from .utils import logger
from .utils import verbose as verbose_decorator
import platform


@verbose_decorator
def bootfunc(inputFunction, x, params, verbose=True):
    """Performs bootstrapping over trials for spectral functions.

    Inputs
    ------
    inputFunction - Function to run on x (the actual function, not the string
    name of the function). Function must take numpy 3D array of form chan x rep
    x time as the first argument and a paramater dictionary as the second
    argument. Function must return a dictionary.

    x - a 3D numpy array or list/tuple of 3D numpy arrays (note: must be chan x
    trial x time). If list/tuple with N elements, an equal number of trials
    will be selected from each pool. Those will then be sampled with
    replacement to obtain bootstrapped mean/variance, i.e., results will be
    computed by combining nPerDraw/N trials from each pool per bootstrap
    repetition. An example use case is to sample an equal number of trials from
    positive and negative polarity FFR trials.

    params - dictionary of parameters. The following fields are required for
    boostrapping, but others may be required based on the specifics of
    inputFunction:

          params['nDraws'] - number of draws for bootstrapping

          params['nPerDraw'] - number of trials per draw for bootstrapping

          params['threads'] - number of threads to spawn; only really useful if
          a multi-core CPU is available. When a multi-core CPU is available,
          computations should be speeded by a factor of approximately
          params['threads'].

    Returns
    -------
    out - Dictionary of dictionaries. Level 1 keys corrspond to
    function outputs, level 2 keys correspond to individual draw results, mean
    of draw results, and variance of draw results for each key in level 1.

    Examples
    --------
    Just a simple illustration of usage.

    >>> # Package imports and initialize parameters
    >>> from anlffr import bootstrap, spectral
    >>> params = spectral.generate_params(...,
                                         threads = 4,
                                         nPerDraw = 250,
                                         nDraws = 100)

    >>> # load preprocessed data sets with 3D arrays nPerDraw is 250, so will
    >>> # select 125 trials from each of these mat files
    >>> positivePolarityData = io.loadmat(...)['data']
    >>> negativePolarityData = io.loadmat(...)['data']

    >>> # create a list of data
    >>> dataList = [positivePolaritydata, negativePolaritydata]

    >>> # run the boostrap function
    >>> results = bootstrap.bootfunc(spectral.mtcpca_complete, dataList,
                                     params)

    Notes
    -------
    Utilizes multiple cores (threads) for a speed increase.
    Designed to work with the spectral analysis functions provided in ANLffr,
    but in theory should work with any function that takes 3D numpy arrays
    and a suitable params structure as inputs.

    Bootstrapping is not implemented in a particularly clever or elegant way -
    it's an "emarassingly parallel" problem, and thus was coded in the
    bare-minimum fashion so that boostrap computations can proceed in parallel
    and change the amount of time you wait for a final result by 1/nThreads.
    Note that this speedup will only take place on Linux/Mac (POSIX in
    general?) systems - not on Windows machines. The current implementation
    does not play well with process forks on Windows, and as such, any attempt
    to run things in multithreaded mode will fall back to a single thread.

    The data array is copied in memory for each core that is utilized, and will
    thus eat quite a bit of RAM when more than 1-2 threads are spawned. RAM is
    cheap, so this shouldn't be too much of a concern. But if you want
    something more memory efficient, go code it yourself.

    Last updated: 10/05/2014
    Auditory Neuroscience Laboratory, Boston University
    Contact: lennyv@bu.edu
    """

    _validate_bootstrap_params(params)

    sanitizedData = _validate_data(x, params)

    startTime = time.time()
    theQueue = multiprocessing.Queue()
    results = {}
    output = {}

    # split the loads roughly equally:
    drawSplit = _compute_thread_split(params)

    # set up the processes
    processList = []
    for proc in range(len(drawSplit)):
        if 'debugMode' in params and params['debugMode']:
            logger.warn('Warning: setting fixed random seeds!')
            randomState = np.random.RandomState(proc)
        else:
            randomState = np.random.RandomState(None)

        if platform.system() != 'Windows':
            processList.append(
                multiprocessing.Process(
                    target=_multiprocess_wrapper,
                    args=(inputFunction,
                          sanitizedData,
                          params,
                          drawSplit[proc],
                          theQueue,
                          randomState)
                    )
                )

            processList[proc].start()
        else:
            _multiprocess_wrapper(inputFunction,
                                  sanitizedData,
                                  params,
                                  drawSplit[proc],
                                  theQueue,
                                  randomState)

    numRetrieved = 0
    trialsUsed = []
    frequencyVector = None

    while numRetrieved < params['nDraws']:
        try:
            retrievedData = theQueue.get(True)
            numRetrieved = numRetrieved + 1

            usefulKeys = retrievedData[0].keys()

            # only need to retrieve frequency vector once
            # otherwise remove it from key values to store
            if 'f' in usefulKeys:
                if 1 == numRetrieved:
                    frequencyVector = retrievedData[0]['f']
                else:
                    if np.any(frequencyVector != retrievedData[0]['f']):
                        logger.error('Internal error: ' +
                                     'frequency axes are different ' +
                                     'across draws')
                usefulKeys.remove('f')

            # now run through the other keys and store the results
            for k in usefulKeys:
                # set up the dictionary fields with the first retrieved piece
                if 1 == numRetrieved:
                    results[k] = dict(runningSum=0, runningSS=0, indivDraw=[])

                if params['returnIndividualBootstrapResults']:
                    results[k]['indivDraw'].append(retrievedData[0][k])

                results[k]['runningSum'] += retrievedData[0][k]
                results[k]['runningSS'] += retrievedData[0][k]**2

            trialsUsed.append(retrievedData[1])

        # the following should be OK, as per:
        # http://stackoverflow.com/questions/
        #        4952247/interrupted-system-call-with-processing-queue
        except IOError as e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise

    # ensure the queue is empty:
    if int(theQueue.qsize()) > 0:
        logger.error('Internal error: retrieved all data, but Queue is ' +
                     'nonempty: {}'.format(int(theQueue.qsize())))

    for k in usefulKeys:
        output[k] = {}
        output[k]['nDraws'] = int(params['nDraws'])
        output[k]['nPerDraw'] = int(params['nPerDraw'])
        if params['returnIndividualBootstrapResults']:
            output[k]['indivDraw'] = np.array(results[k]['indivDraw'])
        output[k]['bootMean'] = results[k]['runningSum'] / params['nDraws']
        output[k]['bootVariance'] = _compute_variance(output[k]['bootMean'],
                                                      results[k]['runningSS'],
                                                      params['nDraws'])
    output['trialsUsed'] = list(trialsUsed)

    if frequencyVector is not None:
        output['f'] = frequencyVector

    logger.info('\nCompleted in: {} s'.format(time.time() - startTime))

    return output


@verbose_decorator
def _multiprocess_wrapper(
        inputFunction,
        inputData,
        params,
        nDraws,
        resultsQueue,
        randState,
        verbose=True):
    """
    internal function. places results from spectral functions in queue.
    """

    for _ in range(nDraws):

        theseData, theseParams, trialsUsed = _select_trials_with_replacement(
            inputData, params, randState)

        out = (inputFunction(theseData, theseParams, verbose=False,
                             bootstrapMode=True),
               trialsUsed)

        resultsQueue.put(out)  # block by default...


@verbose_decorator
def _compute_variance(dataMean, dataSumOfSquares, n, verbose=None):
    """
     internal function. computes variance from running SS, mean, and n.
    """
    return (dataSumOfSquares - ((dataMean * n) ** 2) / n) / (n - 1)


@verbose_decorator
def _select_trials_with_replacement(inputData,
                                    params,
                                    randomState=None,
                                    verbose=True):
    """
    internal function. creates a new data array from a series of randomly
    sampled old ones. random sample is with replacement.
    """

    if not isinstance(inputData, list):
        logger.error('Internal error: inputData should be a list of arrays.')

    # should make a copy of params to avoid original being modified
    modifiedParams = dict(params)

    if randomState is None:
        randomState = np.random.RandomState()

    numPools = len(inputData)
    useTrialsPerPool = int(params['nPerDraw']) / numPools

    tempData = []
    pickTrials = []
    tempSelected = []

    for pool in range(numPools):

        randTrials = randomState.randint(
            0,
            inputData[pool].shape[1],
            useTrialsPerPool)

        tempData.append(inputData[pool][:, randTrials, :])
        pickTrials.append(randTrials)

        # because things will be concatenated, add useTrialsPerPool*pool
        # to each value in randTrials. Functions in spectral.py will only
        # use even-labeled trials When computing the noise floor.
        tempSelected.append(randTrials + useTrialsPerPool*pool)

    useData = np.concatenate(tuple(tempData), axis=1)

    modifiedParams['bootstrapTrialsSelected'] = np.concatenate(
        tuple(tempSelected))

    return (useData, modifiedParams, pickTrials)


@verbose_decorator
def _compute_thread_split(params, verbose=None):
    """
    internal function. computes how many draws each process will take on.
    """
    drawSplit = []
    if platform.system() != 'Windows':
        distribute = int(params['nDraws']) / int(params['threads'])
        leftover = int(params['nDraws']) % int(params['threads'])

        if distribute != 0:
            for _ in range(params['threads']):
                drawSplit.append(distribute)

            for r in range(leftover):
                drawSplit[r] = drawSplit[r] + 1
        else:
            for _ in range(int(params['nDraws'])):
                drawSplit.append(1)
    else:
        drawSplit.append(params['nDraws'])

    return drawSplit


def _validate_data(inputData, params):
    """
    shuffles the trials and ensures that each pool to draw from has the same
    number of trials by selecting the minimum number in common across all pools
    """

    # allows you to specify a list of data from each polarity (or other things
    # you want to combine across) so that the number of trials going into a
    # computation from different sources can be fixed
    try:
        if isinstance(inputData, list) or isinstance(inputData, tuple):
            for x in inputData:
                if not isinstance(x, np.ndarray) and x.ndim != 3:
                    raise TypeError

        elif isinstance(inputData, np.ndarray):
            inputData = [np.array(inputData)]

    except TypeError:
        logger.error('Data should be 3D numpy array or list/tuple ' +
                     'of 3D numpy arrays')

    poolSizes = []

    # select the smallest number in common between sizes of inputData and
    # nPerDraw / len(inputData)
    for x in inputData:
        poolSizes.append(x.shape[1])

    poolSizes.append(params['nPerDraw'] / len(inputData))
    minimumAcrossPools = min(poolSizes)

    # make sure this is always an even number...everything is nicer that way
    if minimumAcrossPools % 2 == 1:
        minimumAcrossPools -= 1

    # make sure the user knows this is what is going on
    warnStr1 = ('Selecting {} per pool '.format(minimumAcrossPools) +
                'WITHOUT replacement from original data supplied')

    warnStr2 = ('Will select from this subset WITH replacement to ' +
                'compute bootstrap mean/variances')

    logger.critical(warnStr1 + '\n' + warnStr2)

    validatedData = []

    if 'debugMode' in params and params['debugMode']:
        randState = np.random.RandomState(31415)
    else:
        randState = np.random.RandomState(None)

    # shuffle the trials within each pool and equate the number of trials:
    for x in inputData:
        # select the same number of trials from each pool without replacement
        randomOrder = randState.permutation(x.shape[1])
        randomOrder = randomOrder[0:minimumAcrossPools]
        validatedData.append(x[:, randomOrder, :])

    return validatedData


@verbose_decorator
def _validate_bootstrap_params(params, verbose=True):
    """
    internal function. checks parameters required for bootfunc and throws an
    error if conditions are violated
    """

    if platform.architecture()[0] != '64bit':
        logger.warning('Python is running in 32 bit mode. ' +
                       'You may encounter out-of-memory issues.')

    if ('nDraw' in params) or ('nPerDraw' in params):
        if 'nDraws' not in params:
            logger.error('when params[''nPerDraw''] is specified, ' +
                         'params[''nDraw''] must be too')

        if 'nPerDraw' not in params:
            logger.error('when params[''nDraws''] is specified, ' +
                         'params[''nPerDraw''] must be too')

        if params['nDraws'] != int(params['nDraws']):
            logger.error('params[''nDraws''] must be an integer')

        if params['nPerDraw'] != int(params['nPerDraw']):
            logger.error('params[''nPerDraw''] must be an integer')

        if params['nDraws'] <= 0:
            logger.error('params[''nDraws''] must be positive')

        if params['nPerDraw'] <= 0:
            logger.error('params[''nPerDraw''] must be positive')

    if platform.system() == 'Windows':
        logger.warn('Windows system detected...' +
                    'will only use one execution thread.')
        checkThreads = False
    else:
        checkThreads = True

    if checkThreads and 'threads' in params:
        print('Attempting to use {} threads...'.format(params['threads']))
        numCpu = multiprocessing.cpu_count()

        if 0 > params['threads']:
            logger.error('params[''threads''] should be > 0')

        if params['threads'] > numCpu:
            logger.warn(
                'params[''threads''] should optimally be <= {}'.format(numCpu))

    return
