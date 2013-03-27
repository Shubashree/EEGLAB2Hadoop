#!/oasis/scratch/csd181/mdburns/python/bin/python
# Copyright (C) 2012 Matthew Burns  <mdburns@ucsd.edu>

from datetime import datetime
import helpers
import helpers.float_open as fop
from scipy import stats, zeros, ones, signal
from scipy.io import loadmat, savemat
import numpy as np
from numpy import array
import os
import sys
import pickle
import argparse
import base64
import gc
import multiprocessing as mp

from hadoop.io import SequenceFile, Text

NUMFOLDS = 5
NUM_SAMPLES = 0
SAMPLE_RATE = 0
NUM_EVENTS = 0
NUM_POINTS = 0
pool=None

def get_eeg(path, file_name):
    print 'get_eeg:  reading EEGLAB .set file '+ file_name

    fullpath = path+file_name

    try:
        f = loadmat(fullpath+'.set', appendmat=False)
    except:
        print >> sys.stderr, 'get_eeg: could not load '+ file_name + '.set'
        return 1

    EEG = f['EEG']
    events = {}
    eeg = {}
    label = []
    latencyInFrame=[]
    uniqueLabel=[]
    event = EEG['event'][0][0][0]

    gc.disable()
    for t_event in event:
        this_latency = str(t_event[0][0][0])
        this_label = str(t_event[1][0][0])
        latencyInFrame.append(this_latency)
        label.append(this_label)

        if this_label not in uniqueLabel:
            uniqueLabel.append(this_label)
    gc.enable()

    uniqueLabel=[int(x) for x in uniqueLabel]

    events['uniqueLabel'] = [str(x) for x in sorted(uniqueLabel)]
    #-1 for Matlab indexing conversion
    events['latencyInFrame'] = [(int(x)-1) for x in latencyInFrame]
    events['label'] = label

    eeg['events']=events
    eeg['num_events']=len(events.keys())
    eeg['sample_rate']=EEG['srate'][0][0][0][0]
    eeg['num_samples']=EEG['pnts'][0][0][0][0]
    eeg['num_channels']=EEG['nbchan'][0][0][0][0]
    eeg['trials']=EEG['trials'][0][0][0][0]
    eeg['ica_weights']=EEG['icaweights'][0][0]
    eeg['ica_sphere']=EEG['icasphere'][0][0]
    eeg['ica_winv=EEG']=['icawinv'][0][0]
    eeg['file_name']=file_name
    eeg['path']=path;
    eeg['channel_locations']=EEG['chanlocs'][0][0]
    eeg['prior_data_path']=EEG['data'][0][0][0]

    return eeg

def find_artifact_indexes(eeg, data):
    windowTimeLength = 200;# in ms.
    windowFrameLength = int(round((eeg['sample_rate'] * windowTimeLength/1000)));
    coefs = ones((windowFrameLength,))

    threshold = 2.1
    args=[data[:,i] for i in np.arange(data.shape[1])]
    result = pool.map(tied_rank, args)
    tdrnk = array(result)/data.shape[0]
    twosidep = np.minimum(tdrnk, 1-tdrnk)
    logliklihood = -np.log(twosidep)

    meanLogLikelihood = np.mean(np.transpose(logliklihood),1)

    windowFrame = np.arange((int(round(-windowFrameLength/2))),int(round((windowFrameLength/2)))).reshape((1,-1))
    meanLogLikelihood = np.nan_to_num(meanLogLikelihood)
    meanLogLikelihood[meanLogLikelihood > 1e20]=1e20
    smoothMeanLogLikelihood =  signal.filtfilt(coefs, array([1]), meanLogLikelihood)/(np.power(windowFrameLength,2))

    isArtifactWindowCenter = np.where(smoothMeanLogLikelihood > threshold)[0].reshape((-1,1))
    print 'clean indexes: number of artifact frames detected = %d' % len(isArtifactWindowCenter)

    artifactFrames = np.tile(windowFrame, (isArtifactWindowCenter.shape[0], 1)) + np.tile(isArtifactWindowCenter, (1 , windowFrame.shape[0]))
    artifactFrames = np.maximum(artifactFrames, 1)
    artifactFrames = np.minimum(artifactFrames, meanLogLikelihood.shape[0])
    artifactFrames = np.unique(artifactFrames[:])-1

    return artifactFrames

def tied_rank(x):
    """
    from: https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py

    Computes the tied rank of elements in x.

    This function computes the tied rank of elements in x.

    Parameters
    ----------
    x : list of numbers, numpy array

    Returns
    -------
    score : list of numbers
            The tied rank f each element in x

    """

    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1):
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r

"""
compile_data: input_str is the path to your imput files, with the character '?' inserted where you want to specify different values.
For instance: 'X:\RSVP\exp?\realtime\exp?_continuous_with_ica' with substitute = range(44,61) will process files
'X:\RSVP\exp44\realtime\exp44_continuous_with_ica.set' ... all the way to 60. The sequence files will be created in the outputpath and
automatically uploaded to hdfs_target_path in the HDFS file system if those are specified. Assumes you are running this on the head node.
"""
def compile_data(input_str, substitute, outputpath='', compression=False, test_file=False, p=None):
    temp = input_str.rpartition(os.sep)
    path_temp = temp[0]
    file_temp = temp[2]

    if outputpath is not '':
        try:
            os.mkdir(outputpath)
        except: pass

    if not p==None:
        global pool
        pool=p

    ica_key, ica_val, raw_key, raw_val = Text(), Text(), Text(), Text()

    for i, v in enumerate(substitute):

        path_to_data = path_temp.replace('?', str(v))
        filename = file_temp.replace('?', str(v))

        eeg = get_eeg(path_to_data + os.sep, filename)

        if eeg is not 1:
            raw_data, ica_act = read_full_float(eeg)
        else:
            continue
        if raw_data is None:
            continue

        print(filename + ': identifying outliers')
        artifact_indexes = find_artifact_indexes(eeg, ica_act)
        eeg['artifact_indexes'] = artifact_indexes;

        f=open('..\\artifact_indexes', 'w')
        pickle.dump(artifact_indexes,f)
        f.close()

        eegstr = pickle.dumps(eeg, protocol=2)

        print(filename + ': compiling dataset into hadoop sequence file')

        if outputpath is '':
            outputpath = path_to_data;

        #Enable compression if requested
        if compression:
            comp_type=SequenceFile.CompressionType.RECORD
        else:
            comp_type=SequenceFile.CompressionType.NONE

        writer = SequenceFile.createWriter(outputpath + os.sep + filename + '.seq', Text, Text, compression_type=comp_type)

        for i in range(raw_data.shape[1]):
            if test_file and i > 3:
                break

            this_raw = np.ascontiguousarray(raw_data[:,i], dtype=raw_data.dtype)
            this_ica = np.ascontiguousarray(ica_act[:,i], dtype=ica_act.dtype)

            ica_key.set(outputpath + os.sep + filename + '.ica.' + str(i+1))
            raw_key.set(outputpath + os.sep + filename + '.raw.' + str(i+1))

            ica_temp = pickle.dumps((this_ica, eegstr), protocol=2)
            raw_temp = pickle.dumps((this_raw, eegstr), protocol=2)

            ica = base64.b64encode(ica_temp)
            raw = base64.b64encode(raw_temp)

            ica_val.set(ica)
            raw_val.set(raw)

            writer.append(raw_key, raw_val)
            writer.append(ica_key, ica_val)

            print(filename + ': '+str(i+1))

        writer.close()
        print  filename + ': finished writing file'

    return 0

def read_full_float(eeg):
    print(eeg['file_name'] + ': reading full float file')
    fn =eeg['path'] + eeg['file_name'] + '.fdt';


    try:
        f = fop.fopen(fn, 'r', 'l')
    except:
        print eeg['file_name']+': could not open ' +  fn
        return None, None

    raw_data = f.read((eeg['num_samples'], eeg['num_channels']), 'float32')
    f.close();

    #Recompute ICA activations
    print (eeg['file_name'] + ': recomputing ICA activations')
    ica_act= np.transpose(np.float32(np.dot(np.dot(eeg['ica_weights'], eeg['ica_sphere']), np.transpose(raw_data))))

    return raw_data, ica_act

def create_file_manifest(input_str, substitute, outputpath=''):
    temp = input_str.rpartition(os.sep)
    path_temp = temp[0]
    file_temp = temp[2]

    f=open(outputpath+os.sep+'manifest.txt','w')

    if outputpath is not '':
        try:
            os.mkdir(outputpath)
        except: pass

    ica_key, ica_val, raw_key, raw_val = Text(), Text(), Text(), Text()

    for i, v in enumerate(substitute):

        path_to_data = path_temp.replace('?', str(v))
        filename = file_temp.replace('?', str(v))

def hadoop2mat(directory):
    result={}
    for fl in os.listdir(directory):
        if fl.split('-')[0]=='part':
            current = os.path.join(directory, fl)
            print current
            if os.path.isfile(current):
                f = open(current, 'rb')
                result_str = f.read().strip('\n')
                f.close()
                if not result_str=='':
                    experiments = result_str.split('\n')
                    kvps = [exp.split('\t') for exp in experiments]
                    for kvp in kvps:
                        this_result = pickle.loads(base64.b64decode(kvp[1]))
                        path, name = kvp[0].rsplit('/', 1)
                        print name
                        result[name]=this_result

    savemat(directory+os.sep+'result.mat', result)

def main():
    parser = argparse.ArgumentParser(description='Recompile EEGLAB files into sequence files for Hadoop')
    parser.add_argument('file_str', type=str)
    parser.add_argument('range', type=int, nargs=2)
    parser.add_argument('outputpath', type=str)
    parser.add_argument('--hdfs_target_path', type=str,default='', dest='hdfs_path')
    parser.add_argument('--compression', help='compression on',action='store_true')
    parser.add_argument('--manifest', help='compile the output as a list of file locations',action='store_true')
    parser.add_argument('--sequencefile', help='compile the output as a hadoop sequencefile for use with hdfs',action='store_true')
    parser.add_argument('--testfile', help='compile a small sequencefile for testing (~10 channels)',action='store_true')
    parser.add_argument('--hadoop2mat', help='collect hadoop output files into a single .mat file',action='store_true')
    #b= ['X:\RSVP\exp?\\realtime\exp?_continuous_with_ica','54' ,'54', 'X:\RSVP\hadoop\\']

    theseargs = parser.parse_args()

    if theseargs.range[0] is theseargs.range[1]:
        trange = [theseargs.range[0]]
    else:
        trange = range(theseargs.range[0] ,theseargs.range[1]+1)

    global pool
    pool = mp.Pool(mp.cpu_count()-1)

    ts = datetime.now()

    #Creates full sequencefiles
    if theseargs.sequencefile:
        print 'eeglab2hadoop: creating sequence file'
        compile_data(theseargs.file_str, trange, theseargs.outputpath, compression=theseargs.compression)

    #Creates list of file locations to bypass hdfs
    if theseargs.manifest:
        print 'eeglab2hadoop: creating manifest'
        create_file_manifest(theseargs.file_str, trange, theseargs.outputpath)

    #Creates small sequencefile for testing purposes
    if theseargs.testfile:
        print 'eeglab2hadoop: creating test file'
        compile_data(theseargs.file_str, trange, theseargs.outputpath, compression=theseargs.compression, test_file=True)

    #Puts the files created by hadoop into a JSON string for Matlab
    if theseargs.hadoop2mat:
        print 'eeglab2hadoop: consolidating hadoop parts into result.mat'
        hadoop2mat(theseargs.file_str)

    c=datetime.now()-ts

    print ' '
    print 'eeglab2hadoop: Completed processing in ' + str(c.seconds) + ' seconds'

    pool.close()
    pool.join()

    return 0


if __name__ == "__main__":
    mp.freeze_support()
    main()