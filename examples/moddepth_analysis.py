#!/usr/bin/env python

'''
Example python script using anlffr functions, meant to be called from the
command line.  This was originally created to analyze data from a Brainvision
EEG system that was already preprocessed in Matlab. There were three conditions
with two polarities each, stored in files of the form
'subjectname_S_trgger.mat', where trigger is 1:12, and only triggers 1-3
(positive polarity) and 7-9 (negative polarity) are of interest. Epochs were
400 ms long.

This generates a .csv file with the results after bootstrapping in an
"Excel-friendly" format. See moddepth_analysis_results_example.csv for sample
output.

command line usage:
$ python moddepth_analysis.py dataDir saveDir subject001 [...]

where [...] are inputs for additional subjects.

Last updated: 10/05/2014
Auditory Neuroscience Laboratory, Boston University
Contact: lennyv@bu.edu
'''

from __future__ import print_function
import os
import sys
from scipy import io
from anlffr import spectral, bootstrap
from anlffr.utils import logger


# prints all info messages from ANLffr to stdout
logger.setLevel('INFO')

def _check_filename(inSaveName, inSaveDir):
    '''
    just checks the filenames and increments a counter after filename if it
    already exists
    '''

    counter = 0

    origFullFilename = os.path.join(inSaveDir, inSaveName)

    fullFilename = str(origFullFilename)

    while os.path.exists(fullFilename):
        counter = counter+1
        fullFilename = (os.path.splitext(origFullFilename)[0] + '_' +
                        str(counter) + '.csv')

    return fullFilename

dataDir = sys.argv[1]
saveDir = sys.argv[2]
subjectList = sys.argv[3:]

# use an auto-calculated nfft length
# but results will only include freqs between 70-1000
# warning: will take a long time, even with multiple threads

# note: nPerDraw should be set to the same number across conditions for a
# single subject. Here, we assume the minimum number of trials available for
# all subjects is 500/condition, split across polarities (i.e., 250 positive,
# 250 negative). If the total number of trials available is > 500 (i.e., >
# 250/polarity), the bootstrap program will fix the number of trials per
# polarity at 250.
params = spectral.generate_parameters(sampleRate=5000,
                                      fpass=[70.0, 1000.0],
                                      tapers=[2, 3],
                                      nDraws=100,
                                      nPerDraw=500,
                                      threads=4,
                                      returnIndividualBootstrapResults=False,
                                      debugMode=False)

# cycle through each subject, then conditions 1-3
for s in subjectList:
    for c in range(1, 4):

        loadName = {}
        print('condition: {}'.format(c))
        loadName['positive'] = s + '_S_' + str(c) + '.mat'
        loadName['negative'] = s + '_S_' + str(c+6) + '.mat'

        # open/check the save file name
        saveName = s + '_condition_' + str(c) + '.csv'
        outputFilename = _check_filename(saveName, saveDir)
        outputFile = open(outputFilename, 'w')

        try:
            # create a list of data to sample from:
            combinedData = []

            # loads the positive and negative mat files for this condition (c)
            for l in loadName:

                wholeMat = io.loadmat(os.path.join(dataDir, loadName[l]))
                mat = wholeMat['data']
                sampleRateFromFile = wholeMat['sampleRate']

                if sampleRateFromFile != params['Fs']:
                    logger.error('sample rate mismatch')

                combinedData.append(mat)

            # call the bootsrapping function using mtcpca_complete
            # this will handle the data shuffling and making sure that
            # things are sampled evenly from each polarity data set
            result = bootstrap.bootfunc(spectral._mtcpca_complete,
                                        combinedData,
                                        params)

            # here is where you can change the format to better suit whatever
            # you're doing see the first outputFile.write command to see the
            # current column headers
            #
            # there are 13 columns
            printStr = ('{0},{1},{2},{3},{4},{5},{6},{7},' +
                        '{8},{9},{10},{11},{12}\n')

            # write the column headers
            outputFile.write(printStr.format(
                'Subject Name',
                'Condition',
                'Number of Draws',
                'Trials per Draw',
                'Frequency',
                'PLV**2: bootstrapped mean',
                'PLV**2: bootstrapped variance',
                'Spectrum: bootstrapped mean',
                'Spectrum: bootstrapped variance',
                'Noise floor PLV**2: bootstrapped mean',
                'Noise floor PLV**2: bootstrapped variance',
                'Noise floor Spectrum: bootstrapped mean',
                'Noise floor Spectrum: bootstrapped variance'))

            # now write all the information to file
            for plvF in range(len(result['f'])):
                toWrite = printStr.format(
                    s,
                    c,
                    params['nDraws'],
                    params['nPerDraw'],
                    result['f'][plvF],
                    (result['mtcpcaPLV_normalPhase']
                           ['bootMean']
                           [plvF]),
                    (result['mtcpcaPLV_normalPhase']
                           ['bootVariance']
                           [plvF]
                     ),
                    (result['mtcpcaSpectrum_normalPhase']
                           ['bootMean']
                           [plvF]
                     ),
                    (result['mtcpcaSpectrum_normalPhase']
                           ['bootVariance']
                           [plvF]
                     ),
                    (result['mtcpcaPLV_noiseFloorViaPhaseFlip']
                           ['bootMean']
                           [plvF]
                     ),
                    (result['mtcpcaPLV_noiseFloorViaPhaseFlip']
                           ['bootVariance']
                           [plvF]
                     ),
                    (result['mtcpcaSpectrum_noiseFloorViaPhaseFlip']
                           ['bootMean']
                           [plvF]
                     ),
                    (result['mtcpcaSpectrum_noiseFloorViaPhaseFlip']
                           ['bootVariance']
                           [plvF]
                     )
                    )

                outputFile.write(toWrite)
            # make sure to close the file when finished
            outputFile.close()

        except IOError:
            print(('\nCannot find file: {}, skipping,' +
                   'condition {} \n').format(loadName[l],
                                             c))
            continue
