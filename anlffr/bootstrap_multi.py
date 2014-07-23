import numpy as np
import time
import multiprocessing
import errno

def bootfunc(inputFunction, x, params, verbose = None):
    """
    Performs bootstrapping over trials for spectral functions, utilizing  
    multiple cores (threads) for a speed increase. Designed to work with the 
    spectral analysis functions provided in ANLffr.

    Inputs
    ------
    inputFunction - Function to run on x. Function must take numpy 3D array of
      form chan x rep x time as the first argument and a paramater dictionary as
      the second argument. Function must return a dictionary.

    x - a 3D numpy array or list/tuple of 3D numpy arrays (chan x rep x time).
      If list/tuple with N elements, an equal number of trial repetitions from 
      each element in the list will be used in each pass of the computation; 
      i.e., results will be computed by combining nPerDraw/N trials per 
      bootstrap repetition.
    
    params - dictionary of parameters. The following fields are required for
      boostrapping, but others may be required based on the specifics of 
      'inputFunction':

          params['nDraws'] - number of draws for bootstrapping

          params['nPerDraw'] - number of trials per draw for bootstrapping
          
          params['nThreads'] - number of threads to spawn 
    
    Returns
    -------
    out - Dictionary of dictionaries. Level 1 keys corrspond to 
      function outputs, level 2 keys correspond to mean, variance,
      and individual draws for each key in level 1.
    
    Notes 
    ----- 
    This is not implemented in a particularly clever or elegant way, as it is
    meant to "just work". But it does provide roughly a factor of N speedup
    (where N is the number of CPU cores dedicated to this task). The data array
    is copied in memory for each core that is utilized, and will thus eat quite
    a bit of RAM when more than 1-2 threads are spawned. RAM is cheap, so this
    shouldn't be too much of a concern in poorly funded labs. But if you want
    something more memory efficient, go code it yourself.

    Last updated: 7/22/2014
    Auditory Neuroscience Laboratory, Boston University
    Contact: lennyv@bu.edu 
    """

    _validate_bootstrap_params(params)

    startTime = time.time()
    theQueue = multiprocessing.Queue()
    output = {}

    print('Using {} threads...'.format(params['nThreads']))
    
    # split the loads roughly equally:
    drawSplit = []
    distribute = int(params['nDraws']) / params['nThreads'] 
    leftover = int(params['nDraws']) % params['nThreads'] 
    for _ in range(params['nThreads']):
        drawSplit.append(distribute)
    for r in range(leftover):
        drawSplit[r] = drawSplit[r] + 1
            
    processList = []
    
    for proc in range(params['nThreads']):
        processList.append(multiprocessing.Process(target = _multiprocess_wrapper,
                           args = (inputFunction, x, params, drawSplit[proc], theQueue)))
        processList[proc].start()
   
    results = {}    
    numRetrieved = 0

    while numRetrieved < params['nDraws']:
        try:
            retrievedData = theQueue.get(True) 
            numRetrieved = numRetrieved + 1
            print('Retrieved data from draw {}'.format(numRetrieved))

            for k in retrievedData.keys():
                if 1 == numRetrieved:
                    results[k] = dict(runningSum = 0, runningSS = 0, indivDraw = [])

                results[k]['indivDraw'].append(retrievedData[k])
                results[k]['runningSum'] += retrievedData[k]
                results[k]['runningSS'] += retrievedData[k]**2

        # the following should be OK, as per:
        # http://stackoverflow.com/questions/
        #        4952247/interrupted-system-call-with-processing-queue
        except IOError, e:
            if e.errno == errno.EINTR:
                continue
            else:
                raise

    for k in results.keys():
        output[k]['indivDraw'] = np.array(results[k]['indivDraw'])
        output[k]['bootMean'] = results[k]['runningSum'] / params['nDraws']
        output[k]['bootVariance'] = _compute_variance(output[k]['mean'], 
                                                      results[k]['runningSS'],
                                                      params['nDraws'])
    
    elapsedTime = 'Completed in: {} s'.format(time.time() - startTime)
    
    print(elapsedTime)

    return results

def _multiprocess_wrapper(inputFunction, inputData, inputParams, nDraws, resultsQueue):
    """ 
    internal function. places results from spectral functions in queue.
    """
    # set random seed here, otherwise this might draw the same pool 
    # of trials across all processes
    np.random.seed()
    
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

    for _ in range(inputParams['nDraws']):
        out = inputFunction(_combine_random_subset(inputData, inputParams), 
                            inputParams, 
                            verbose = False)
        resultsQueue.put(out)

def _compute_variance(dataMean, dataSumOfSquares, n):
    """
     internal function. computes variance from running SS, mean, and n.
    """
    return ((dataSumOfSquares - ((dataMean*n)**2) / n ) / (n-1))

def _combine_random_subset(inputData, params):
    """
    internal function. creates a new array from a randomly sampled old one.
    """
    assert type(inputData) == list, 'inputData should be a list of arrays'

    warnString = ('warning: number of trials requested in draw ' + 
        '> trials available for pool {}')
    numPools = len(inputData)
    useTrialsPerPool = int(params['nPerDraw'] / numPools)

    tempData = []
    
    for pool in range(numPools):
        if useTrialsPerPool > inputData[pool].shape[1]:
            print(warnString.format(pool))
        
        pickTrials = np.random.randint(0,inputData.shape[1],useTrialsPerPool)
        tempData.append(inputData[pool][:,picktrials,:])
        
        useData = np.concatenate(tuple(tempData), axis = 1)

        return useData
    
    print('using {} trials from each data pool supplied'.format(useTrialsPerPool))

    return useData


def _validate_boostrap_params(params):
    """
    internal function. checks/fixes parameters required for bootfunc.
    """
    # check/fix bootStrap input
    if ('nDraw' in params) or ('nPerDraw' in params):
        assert 'nDraw' in params, ('when params[''nPerDraw''] is specified, ' +
                'params[''nDraw''] must be too')
        assert 'nPerDraw' in params, ('when params[''nDraws''] is specified, ' +
                'params[''nPerDraw''] must be too')

        assert params['nDraw'] = int(params['nDraw']), 'params[''nDraw''] ' +
            'must be an integer'
        assert params['nPerDraw'] = int(params['nPerDraw']), 'params[''nPerDraw''] ' + 
            'must be an integer'

        assert params['nDraw'] > 0, 'params[''nDraw''] must be positive'
        assert params['nPerDraw'] > 0, 'params[''nPerDraw''] must be positive'
    
    if 'nThreads' in params:
        numCpu = multiprocessing.cpu_count()
        assert(params['nThreads'] = int(params['nThreads']))
        assert 0 < params['nThreads'], 'params[''nThreads''] should be > 0'
        assert params['nThreads'] <= numCpu, ('params[''nThreads''] should be ' + 
            ' <= {}'.format(numCpu))
    else:
        params['nThreads'] = 1

    return
