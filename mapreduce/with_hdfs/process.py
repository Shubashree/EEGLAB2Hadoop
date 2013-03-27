'''
Created on Mar 1, 2013

@author: user
'''
import numpy as np
import multiprocessing as mp
from numpy import ndarray, zeros, ones, concatenate, array, arange, ma, append
import scipy as sp
from scipy import ceil,floor, io, stats, var
from scipy.io import loadmat, savemat
import os
import pickle
import math

from cvxopt import spmatrix, matrix, sparse, lapack, solvers, spdiag, base
import cvxopt
import sys
import gc
import time
import base64
import gc

def process(y, eeg, EPOCH_LENGTH, EPOCH_OFFSET, NUM_FOLDS, p=None):
    sr = eeg['sample_rate']
    events = eeg['events']
    event_types = events['uniqueLabel']

    ns = int(ceil(EPOCH_LENGTH*sr))

    #Identify artifacts
    artifact_indexes = zeros((y.shape[0],1))
    artifact_indexes[eeg['artifact_indexes']]=1
    num_occurances, events = remove_corrupted_events(event_types, events, artifact_indexes, ns)

    #Shift signal to account for negative response
    zpadpre=zeros((int(ceil(EPOCH_OFFSET*sr)), 1))
    zpadpost=zeros((int(ceil((EPOCH_LENGTH-EPOCH_OFFSET)*sr)), 1))
    y = concatenate((zpadpre, y, zpadpost))
    artifact_indexes = concatenate((zpadpre, artifact_indexes, zpadpost))

    result = np.empty((2, NUM_FOLDS, len(event_types), 2))
    if not p==None:

        reg_parent_conn, reg_child_conn = mp.Pipe()
        av_parent_conn, av_child_conn = mp.Pipe()
        these_args = (y, events, artifact_indexes, ns, num_occurances, NUM_FOLDS,)
        res_reg = p.apply_async(cross_validate_regression, these_args)
        res_av = p.apply_async(cross_validate_average, these_args)

        result[0,:,:,:]=res_av.get()
        result[1,:,:,:]=res_reg.get()

    else:
        result[0,:,:,:] = cross_validate_average(y, events, artifact_indexes, ns, num_occurances, NUM_FOLDS);
        result[1,:,:,:] = cross_validate_regression(y, events, artifact_indexes, ns, num_occurances, NUM_FOLDS);

    return result

def remove_corrupted_events(event_types, events, artifact_indexes, ns):
    count=0
    num_occurances = zeros((len(event_types),1))
    labels = events['label']
    lif = events['latencyInFrame']
    for i, this_event in enumerate(labels):
        this_latency = lif[i]
        this_mask = artifact_indexes[arange(this_latency, min(this_latency+ns, artifact_indexes.shape[0]), dtype=np.int)]

        if any(this_mask):
            events['label'][i] = 'corrupted'
            count = count + 1

        else:
            for j, this_event_type in enumerate(event_types):
                if this_event == this_event_type:
                    num_occurances[j] = num_occurances[j] + 1

    mes = 'process.remove_corrupted_events: rejected %d events because of outliers\n' % count
    sys.stderr.write(mes)

    return num_occurances, events

def cross_validate_regression(y, events, artifact_indexes, ns, num_occurances, NUM_FOLDS):
    gc.disable()

    mes = 'process.cross_validate_regression: generating predictor\n'
    sys.stderr.write(mes)
    predictor = predictor_gen(events, ns, num_occurances, y.shape[0])

    rov_reg= np.empty((NUM_FOLDS, len(num_occurances), 2))
    testmask = extract_folds(y, NUM_FOLDS)
    for i in arange(NUM_FOLDS):
        mes = 'process.cross_validate_regression: processing fold %d\n' % (i+1)
        sys.stderr.write(mes)

        test_idx = testmask[i]
        res = regress_erp(y, test_idx, predictor, events,  ns)[0]
        rov_reg[i,:,:]=res

    gc.enable()
    return rov_reg

def cross_validate_average(y, events, artifact_indexes, ns, num_occurances, NUM_FOLDS):
    gc.disable()

    rov_av= np.empty((NUM_FOLDS, len(num_occurances), 2))
    testmask = extract_folds(y, NUM_FOLDS)
    for i in arange(NUM_FOLDS):
        mes = 'process.cross_validate_average: processing fold %d\n' % (i+1)
        sys.stderr.write(mes)

        test_idx = testmask[i]
        res = average_erp(y, test_idx, events, ns)[0]
        rov_av[i,:,:]=res

    gc.enable()
    return rov_av

def extract_folds(y, NUM_FOLDS):
    dlength = y.shape[0]
    split_length = int(floor(dlength/NUM_FOLDS))
    testmask= []

    for i in arange(0, NUM_FOLDS):
        start_idx = split_length*i
        end_idx = split_length*(i+1)
        this_mask = zeros(dlength, dtype=bool)
        this_mask[start_idx:end_idx] = True
        testmask.append(this_mask)

    return testmask

def predictor_gen(events, ns, num_occurances, dlength):
    gc.disable()
    event_types = events['uniqueLabel']
    matsize = dlength, int(len(event_types)*ns)

    ii = np.empty(0, dtype=np.int)
    jj = np.empty(0, dtype=np.int)

    for j, this_type in enumerate(event_types):
        j_start_idx = j*ns;
        j_end_idx = j_start_idx + ns;

        j_vec = array(np.tile(arange(j_start_idx, j_end_idx), num_occurances[j]),dtype=np.int)
        i_vec = zeros(num_occurances[j]*ns, dtype=np.int)

        m=0
        for i, this_event in enumerate(events['label']):
            if this_event == this_type:
                i_start_idx = m*ns
                i_end_idx = i_start_idx + ns
                this_latency = events['latencyInFrame'][i]
                i_vec[arange(i_start_idx, i_end_idx, dtype=np.int)] = arange(this_latency,this_latency+ns,dtype=np.int);
                m=m+1

        mask = i_vec < dlength

        ii = append(ii, i_vec[mask], axis=0)
        jj = append(jj, j_vec[mask], axis=0)

    ii = ii.tolist()
    jj = jj.tolist()

    retval = spmatrix(1.0, ii, jj , matsize)
    gc.enable()
    return retval

def regress_erp(y, test_idx, predictor, events,  ns):
    event_types = events['uniqueLabel']
    labels = events['label']
    latencies = events['latencyInFrame']

    train_idx = ~test_idx
    ytrn = matrix(y[train_idx].tolist()).T

    #There is a specific test_set to use
    if (len(np.where(test_idx)[0])!=0):
        tst_start_idx = min(np.where(test_idx)[0])
        tst_end_idx = max(np.where(test_idx)[0])

    #Test on all the data
    else:
        tst_start_idx = min(np.where(~test_idx)[0])
        tst_end_idx = max(np.where(~test_idx)[0])

    train_idx_list= np.where(train_idx==1)[0]
    train_idx_list = array(train_idx_list, dtype=np.int).tolist()

    #Solve the system of equations y = Ax
    P = predictor[train_idx_list,:].T*predictor[train_idx_list,:]
    q = -predictor[train_idx_list, :].T*ytrn
    rerp_vec = solvers.coneqp(P, q)['x']

    yestimate = array(predictor*rerp_vec)
    y_temp = matrix(y.tolist()).T
    noise = y_temp-yestimate


    events_to_test = np.where((array(latencies)<tst_end_idx) & (array(latencies)>tst_start_idx))[0]
    gc.disable()
    #Compute performance stats
    stats = np.empty((len(event_types),2))
    for i, this_type in enumerate(event_types):
        this_stat = np.empty((0,2))
        for j, event_idx in enumerate(events_to_test):
            this_event=labels[event_idx]
            if this_event==this_type:
                start_idx = latencies[event_idx];
                end_idx = np.minimum(tst_end_idx, start_idx+ns)

                yblock = y[start_idx:end_idx]
                noiseblock = noise[start_idx:end_idx]
                this_stat = np.append(this_stat, array([[sp.var(yblock)], [sp.var(noiseblock)]]).T, axis=0)

        rov_raw = this_stat[:,0]-this_stat[:,1]
        rov_nor = rov_raw/this_stat[:,0]
        rov = array([sp.mean(rov_raw), sp.mean(rov_nor)])
        stats[i,:] =  rov

    gc.enable()
    return stats, np.reshape(array(rerp_vec),(-1, ns)).T

def average_erp(y, test_idx, events, ns):

    event_types = events['uniqueLabel']
    labels = events['label']
    latencies = events['latencyInFrame']
    y = np.reshape(y, (-1,))

    train_idx = ~test_idx

    #There is a specific test_set to use
    if (len(np.where(test_idx)[0])!=0):
        tst_start_idx = min(np.where(test_idx)[0])
        tst_end_idx = max(np.where(test_idx)[0])

    #Test on all the data
    else:
        tst_start_idx = 0
        tst_end_idx = max(np.where(~test_idx)[0])

    #Sort the events based on whether in train or testset
    events_to_test=[]
    events_to_train=[]
    for i, latency in enumerate(latencies):
        if (latency < tst_end_idx) & (latency > tst_start_idx):
            events_to_test.append(i)
        else:
            events_to_train.append(i)

    if len(events_to_train)==0:
        events_to_train = range(len(latencies))
    gc.disable()
    #Calc average erp
    aerp_vec = zeros((ns, len(event_types)))
    for i, this_type in enumerate(event_types):
        event_idx = [j for j, this_latency in enumerate(latencies) if (j in events_to_train)&(labels[j]==this_type)]
        yblock = zeros((ns, len(event_idx)))
        for k, idx in enumerate(event_idx):
            start_idx = latencies[idx]
            end_idx = start_idx+ns

            if y.shape[0]-1 < end_idx:
                continue

            yblock[:,k] = y[start_idx:end_idx]

        aerp_vec[:,i] = np.mean(yblock,axis=1)

    #Calc stats
    stats = np.empty((len(event_types),2))
    for i, this_type in enumerate(event_types):
        this_stat = np.empty((0,2))
        for j, event_idx in enumerate(events_to_test):
            this_event=labels[event_idx]
            if this_event==this_type:
                start_idx = latencies[event_idx];
                end_idx = np.minimum(tst_end_idx, start_idx+ns)

                yblock = y[start_idx:end_idx]
                block_range = arange(len(yblock))
                noiseblock = yblock - aerp_vec[block_range,i]
                this_stat = np.append(this_stat, array([[sp.var(yblock)], [sp.var(noiseblock)]]).T, axis=0)

        rov_raw = this_stat[:,0]-this_stat[:,1]
        rov_nor = rov_raw/this_stat[:,0]
        rov = array([sp.mean(rov_raw), sp.mean(rov_nor)])
        stats[i,:] =  rov

    gc.enable()
    return stats, aerp_vec

def main():
    from mapreduce.with_hdfs.process import *
    import helpers.eeglab2hadoop
    from hadoop.io import SequenceFile, Text

    p=mp.Pool()
    clean_indexes = io.loadmat('C:\\Users\\user\\Google Drive\\SCCN\\ERP Regression Project\\clean_indexes.mat')['clean_indexes']
    ai = np.logical_not(clean_indexes)

    EPOCH_OFFSET=.125
    EPOCH_LENGTH=1.0
    NUM_FOLDS = 5

    reader = SequenceFile.Reader('X:\\RSVP\\hadoop_input\\exp53_continuous_with_ica.seq')
    key_class = reader.getKeyClass()
    value_class = reader.getValueClass()

    key = key_class()
    value = value_class()
    desired='X:\\RSVP\\hadoop_input\\exp53_continuous_with_ica.ica.1'

    while(1):
        a = reader.next(key, value)
        if key._bytes==desired:
            break;

    v = pickle.loads(base64.decodestring(value._bytes))
    y = v[0].reshape((-1,1))
    eeg = pickle.loads(v[1])
    sr= eeg['sample_rate']

    events = eeg['events']
    event_types = events['uniqueLabel']
    EPOCH_LENGTH = 1
    ns = EPOCH_LENGTH*eeg['sample_rate']

    artifact_indexes = zeros((y.shape[0],1))
    artifact_indexes[eeg['artifact_indexes'],0]=1

    zpadpre=zeros((int(ceil(EPOCH_OFFSET*sr)), 1))
    zpadpost=zeros((int(ceil((EPOCH_LENGTH-EPOCH_OFFSET)*sr)), 1))
    y = np.concatenate((zpadpre, y, zpadpost))
    artifact_indexes = np.concatenate((zpadpre, artifact_indexes, zpadpost))

    #Convert to masked array
    ytrain = np.ma.array(y,mask=np.logical_not(ones(y.shape[0])))
    num_occurances, events = remove_corrupted_events(event_types, events, artifact_indexes, ns)
    dlength = ytrain.shape[0]

##    ts = time.time()
##    print 'generating predictor'
##    predictor = predictor_gen(events, ns, num_occurances, dlength)
##    print 'predictor time = %f minutes' % ((time.time()-ts)/60.0)
##    print 'predictor dimension = ' + str(predictor.size)
##
##
##    ts = time.time()
##    print 'starting regression'
##    rstats, rerps = regress_erp(ytrain, predictor, events, ns )
##    print 'regression time = %f minutes' % ((time.time()-ts)/60.0)
##
##    ts = time.time()
##    print 'starting averaging'
##    astats, aerps = average_erp(ytrain, events, ns)
##    print 'averaging time = %f minutes' % ((time.time()-ts)/60.0)
##
##
##
##    ts = time.time()
##    rov_reg = cross_validate_regression(y, events, artifact_indexes, ns, num_occurances, NUM_FOLDS)
##    print 'main: regression time = %f minutes' % ((time.time()-ts)/60.0)
##
##    ts = time.time()
##    rov_av = cross_validate_average(y, events, artifact_indexes, ns, num_occurances, NUM_FOLDS)
##    print 'main: averaging time = %f minutes' % ((time.time()-ts)/60.0)

    ts = time.time() #Runtime ~ 3 min.
    rov = process(y, eeg, EPOCH_LENGTH, EPOCH_OFFSET, NUM_FOLDS, p)
    print 'main: processing time = %f minutes' % ((time.time()-ts)/60.0)
    pass

if __name__ == "__main__":
    mp.freeze_support()