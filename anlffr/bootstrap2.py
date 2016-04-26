import numpy as np
from anlffr import spectral
from joblib import Parallel, delayed
from .utils import logger
from .utils import verbose
import time

@verbose
def bootfunc(inputFunction, x1, params, verbose=True):
    startTime = time.time()
    try:
        nJobs = int(params['threads'])
    except KeyError:
        nJobs = 1
    nDraws = int(params['nDraws'])

    x1, n = _equate_within_pool(x1)
    if nJobs == 1:
        results = []
        for i in range(nDraws):
            results.append(_run_bootfunc(inputFunction, x1, params))
    else:
        P = Parallel(n_jobs=nJobs)
        P(delayed(_run_bootfunc)(inputFunction, x1, params) for i in range(nDraws))

    concatenated = _dict_concatenate(results)

    output = {}
    usefulKeys = concatenated.keys()
    output['f'] = concatenated['f']
    usefulKeys.remove('f')

    for k in usefulKeys:
        output[k]['bootMean'] = np.mean(concatenated[k], axis=0)
        output[k]['bootVar'] = np.var(concatenated[k], axis=0)
        output[k]['nDraws'] = nDraws
        output[k]['nPerDraw'] = n
        if params['returnIndividualBootstrapResults']:
            output[k]['indivDraw'] = concatenated[k]

    logger.info('\nCompleted in: {} s'.format(time.time() - startTime))
    return output


@verbose
def permutation_distributions(inputFunction, x1, x2, params, verbose=True):

    '''
    Returns x1-x2 when each is computed using inputFunction, and also the
    distribution of differences when the labels are shuffled at random.
    inputFunction is expected to be one of the functions from anlffr.spectral.
    '''
    try:
        nJobs = int(params['threads'])
    except KeyError:
        nJobs = 1
    nDraws = params['nDraws']

    if len(x1) != len(x2):
        raise ValueError('x1 and x2 must have same first dimension')

    x1, n1 = _equate_within_pool(x1)
    x2, n2 = _equate_within_pool(x2)
    
    x1Res = inputFunction(x1, params, bootstrapMode=True)
    x2Res = inputFunction(x2, params, bootstrapMode=True)
    difference = _dict_diff(x1Res, x2Res)

    if nJobs == 1:
        results = []
        for i in range(nDraws):
            results.append(get_null_difference(inputFunction, x1, x2, params))

    else:
        P = Parallel(n_jobs=nJobs)
        P(delayed(_get_null_difference)(inputFunction, x1, x2, params) for i in range(nDraws))

    nullDifferenceDistribution = _dict_concatenate(results)

    return difference, nullDifferenceDistribution


@verbose
def _dict_diff(x1Res, x2Res, verbose=True):
    '''
    for each key in a dictionary, subtract x2 from x1. both assumed to be
    arrays.
    '''
    difference = {}
    usefulKeys = x1Res.keys()
    usefulKeys.remove('f')

    for k in usefulKeys:
        difference[k] = x1Res[k] - x2Res[k]

    difference['f'] = x1Res['f']

    return difference


@verbose
def _dict_concatenate(resList, verbose=True):
    '''
    Concatenates arrays in a list of dictionaries, as would be returned when
    running anlffr.spectrum functions using joblib parallel instance.
    '''
    concatenated = {}
    usefulKeys = resList[0].keys()
    concatenated['f'] = reslist[0]['f']
    usefulKeys.remove('f')
    
    for k in usefulKeys:
        toConcatenate = []
        for y in range(len(resList)):
            toConcatenate.append(resList[y][k])
        concatenated[k] = np.vstack(toConcatenate)

    return concatenated


@verbose
def _run_bootfunc(inputFunction, x1, params, verbose=True):
    '''
    the function actually being fed to joblib parallel for bootstrap
    computation of mean and variance
    '''
    x1s = _sample_with_replacement(x1)
    x1sRes = inputFunction(np.concatenate(x1s, axis=1), params)

    return x1sRes 


@verbose
def _get_null_difference(inputFunction, x1, x2, params, verbose=True):
    '''
    the function actually being fed to joblib parallel for permutation testing 
    '''
    x1s, x2s = _label_shuffler(x1, x2)
    x1sRes = inputFunction(np.concatenate(x1s, axis=1), params)
    x2sRes = inputFunction(np.concatenate(x2s, axis=1), params)

    nullDiff = _dict_diff(x1sRes, x2sRes) 

    return nullDiff


@verbose
def _label_shuffler(x1, x2, verbose=True):
    '''
    randomly reassigns the trials belonging to x1 and x2, and returns two
    arrays the same size as the original input
    '''
    r = np.random.RandomState(None)
    x1s = np.empty(x1.shape)
    x2s = np.empty(x2.shape)
    for y in range(len(x1)):
        temp = np.concatenate([x1[y], x2[y]], 1)
        tempOrder = r.permutation(temp.shape[1])
        temp = temp[:, tempOrder, :]
        x1s[y] = temp[:, 0:x1.shape[1], :]
        x2s[y] = temp[:, x1.shape[1]:, :]

    return x1s, x2s


@verbose
def _equate_within_pool(inputData, verbose=True):
    '''
    Sets the number of trials per list element to be equal For example: useful
    when computing EFRs, and need an equal number of +/- polarity trials. 
    '''
    r = np.random.RandomState(None)
    errorStr = 'Expecting a 4d array, or list/tuple of 3D arrays'
    if isinstance(inputData, np.ndarray):
        if len(inputData.shape) == 3:
            inputData = [inputData]
        else:
            raise ValueError(errorStr)

    minAvail = np.Inf
    for x in inputData:
        if not isinstance(x, np.ndarray):
            raise ValueError(errorStr)
        if x.shape[1] < minAvail:
            minAvail = x.shape[1]
    n = 0
    for x in range(len(inputData)):
        keepTrials = r.permutation(inputData[x].shape[1])[0:minAvail]
        n += len(keepTrials)
        inputData[x] = inputData[x][:, keepTrials, :]

    return inputData, n


@verbose
def _sample_with_replacement(inputData, verbose=True):
    '''
    generates a dataset the same size as the original that was constructed by
    sampling the original with replacement
    '''
    r = np.random.RandomState(None)
    errorStr = 'Expecting a list/tuple of 3D arrays'
    if isinstance(inputData, np.ndarray):
        if len(inputData.shape) == 3:
            inputData = [inputData]
        else:
            raise ValueError(errorStr)

    resampled = list(inputData)

    for x in range(len(inputData)):
        if not isinstance(inputData[x], np.ndarray):
            raise ValueError(errorStr)
        tr = r.randint(inputData[x].shape[1], size=inputData[x].shape[1])
        resampled[x] = inputData[x][:, tr, :]

    return resampled
