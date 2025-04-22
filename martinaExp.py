import numpy
import subprocess
import os
import glob
import pickle
seed = 1234
numpy.random.seed(seed)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor


import qiskit 
import matplotlib.pyplot as plt

"""
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime import EstimatorV2 as Estimator

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi
from qiskit.circuit.random import random_circuit
from qiskit import qasm2

# fake noise all machines support 127 qubits
from qiskit_ibm_runtime.fake_provider import *
"""

from functions import *

check = 0
loadData = 0
prepareData = 1
computeDistances = 1
run = 0

distances = ['KL', 'BC', 'TV', 'hotelling', 'FID']#[:3]
featureMaps = ['none']#, 'average shots', 'random'][:2] #'pca', 
methods = ['ratio']#, 'hist']
dataNames = ['train', 'calibration', 'test']
distanceFunctions = {
    'KL' : KL, 
    'BC' : BC, 
    'TV' : TV, 
    'hotelling' : hotelling, 
    'FID' : FID
    }



devices = ['athens', 'casablanca', 'lima', 'quito', 'santiago', '5_yorktown'][:5]
prefixes = ['ibmq_', 'fake_', 'ideal_']


nShots = 1000
nSteps = 9
nExecutions = 5
nShots = 1000
alpha = 0.1
datadir = 'Data/Output/Martina/Steps/'
outdir = 'Outputs/Martina/'
##################################3
# Load dataset from Output
if loadData:    
    datasets = {}
    for device in devices:
        for nE in range(1, nExecutions):
            for iStep in range(nSteps):
                #key = device+'-'+circuit+'-'+str(nQ)
                print('device:', device, 
                      ', step:', iStep, 
                      ', execution:', nE)
                for prefix in prefixes:
                    key = prefix+device+'_execution_'+str(nE)+'_step_'+str(iStep)
                    key = prefix+device+'_step_'+str(iStep)+'_'+'execution_' +str(nE)
                    fileOut = datadir+key+'.out'
                    datasets[key] =numpy.loadtxt(fileOut, dtype=int, delimiter=' ') 

    with open(outdir+'datasets.npy', 'wb') as f:
        pickle.dump(datasets, f)

if computeDistances:
    with open(outdir+'datasets.npy', 'rb') as f:
        datasets = pickle.load(f)
    
    allDeviceScores = []
    for device in devices:
        # select ideal exp and device
        keys = []
        for key, val in datasets.items():
            k = key.split(sep='_')
            print(k)
            if k[0] == 'ideal' and k[1] != device:#k[2] != device.split(sep='_')[1]:
                keys.append(key)
        print('len keys', len(keys)) 
        # create measure keys
        measureKeys = []
        for distance in distances:
            for method in methods:
                key = distance+'_'+method
                measureKeys.append(key)
        
        M = {}
        for mkey in measureKeys:
            print(mkey)
            distance, method = mkey.split(sep = '_')
            f = distanceFunctions[distance]
            for key in keys:
                ibmkey = 'ibmq'+key[5:]
                comb = mkey+'&'+ibmkey
                s = [datasets[key], datasets[ibmkey]]
                #X1, X2 =reduceDimensions(s[0], s[1], -1, None)
                M[comb] = [f(s[0], s[1], method, None)]

        print(len(M))
        # key splits
        modes = ['all', 'mondrian', 'shift', 'shift + mondrian']
        data = [[{}, {}, {}] for mode in modes]
        shots = [[{}, {}, {}] for mode in modes]
        for imode in range(len(modes)):
            for idata in [0, 1, 2]:
                for key in measureKeys:
                    data[imode][idata][key] = []
                    shots[imode][idata][key] = []
        # test is the same for everyone
        for imode in range(len(modes)):
            for comb, val in M.items():
                key, dkey = comb.split(sep='&')
                if int(dkey.split(sep='_')[3]) == 8: # last step
                    data[imode][2][key].append(val)
                    shots[imode][2][key].append(dkey)
            print('len test', len(shots[imode][2]))
        # cal = all, nqubits[-2], nquabits[-2] 
        for imode in range(len(modes)):
            mode  = modes[imode]
            for comb, val in M.items():
                key, dkey = comb.split(sep='&')
                print(dkey)#ibmq_santiago_step_8_execution_4
                if mode == 'all':
                    if int(dkey.split(sep='_')[3]) < 8:
                        data[imode][1][key].append(val)
                        shots[imode][1][key].append(dkey)
                    if int(dkey.split(sep='_')[3]) < 8:#nQubits[-1]:
                        data[imode][0][key].append(val)
                        shots[imode][0][key].append(dkey)
                
                if mode in ['mondrian', 'shift + mondrian']:
                    if int(dkey.split(sep='_')[3]) in [7, 6]:#nQubits[-1], nQubits[-2]]:
                        #nQubits[-2]:#smax - 1:
                        data[imode][1][key].append(val) #calibration
                        shots[imode][1][key].append(dkey) #calibration
                    if int(dkey.split(sep='_')[3]) < 6:
                        data[imode][0][key].append(val) #train
                        shots[imode][0][key].append(dkey) #train
                
                if mode in ['shift']:
                    if int(dkey.split(sep='_')[3]) < 8:
                        r = numpy.random.rand() > 0.5
                        if r:
                            data[imode][1][key].append(val) # calibration
                            shots[imode][1][key].append(dkey) # calibration
                        else:
                            data[imode][0][key].append(val)
                            shots[imode][0][key].append(dkey)
               
        with open(outdir+'data.npy', 'wb') as f:
            pickle.dump(data, f)
        with open(outdir+'shots.npy', 'wb') as f:
            pickle.dump(shots, f)
        
        with open(outdir+'datasets.npy', 'rb') as f:
            datasets = pickle.load(f)
        
        with open(outdir+'data.npy', 'rb') as f:
            data = (pickle.load(f))
        with open(outdir+'shots.npy', 'rb') as f:
            allshots = (pickle.load(f))
        
        S = []
        for imode in range(len(modes)):
            mode = modes[imode]
            high = {}
            Scores = {}
            good = {}
            RFs = {}
            normalizations = {}
            Q = {}
            #data[imode][idata][key] = []
            for idata in range(len(dataNames)):
                dataname = dataNames[idata]
                
                shotkeys = allshots[imode][idata] 
                measures = data[imode][idata]

                for key, val in measures.items():
                    if len(measures[key]):
                        print(key, ':', numpy.max(measures[key]))
                    else: print(key, 'problem ', dataname, mode)
                    # retrive ibm shots 
                    shots = []
                    for datakeys in shotkeys[key]:
                        #print(datakeys)
                        shots.append(datasets[datakeys])

                    y = numpy.array(measures[key]).squeeze()
                    print('y->',y.shape)
                    cut = 2
                    means = [numpy.mean(x[:, :cut], axis=0).tolist() for x in shots]
                    covs  = [numpy.cov(x[:, :cut].T).reshape(cut**2).tolist() for x in shots]
                    X = [numpy.array(means[i] + covs[i]) for i in range(len(covs))]
                    
                    if dataname == 'train' and mode in ['shift', 'shift + mondrian']:
                        # train model (training set only)
                        RFs[key] = sklearn.ensemble.RandomForestRegressor() 
                        RFs[key].fit(X, y)
                        haty = RFs[key].predict(X) 
                        print('train ER', numpy.mean((haty - y)**2))
                    
                    if dataname == 'calibration':
                        # CP calibration (calibration set only)
                        if mode in ['shift', 'shift + mondrian']: 
                            haty = RFs[key].predict(X) 
                            print('cal ER', numpy.mean((haty - y)**2))
                            a = y - haty
                        else:
                            a = y
                        print('a->', a.shape)
                        na = int(numpy.ceil((1 - alpha) * (len(a) + 1)))
                        Q[key] = numpy.sort(a)[na-1] 
                        print('q, max:', Q[key], numpy.max(a))
                    
                    if dataname == 'test':
                        # CP evaluation (test set only)
                        if mode in ['shift', 'shift + mondrian']: 
                            haty = RFs[key].predict(X) 
                            print('test ER', numpy.mean((haty - y)**2))
                            a = y - haty
                        else:
                            haty = 0 * numpy.ones(len(y))
                            a = y
                        high[key] = haty + Q[key]
                        good[key] = (a <=Q[key])

                        size = numpy.mean(high[key])#/abs(numpy.mean(measures[key]))
                        val = numpy.mean(1 * good[key])
                        Scores[key] = [size, val]

            for key, v in Scores.items():
                    print(key, ':', v)
            S.append(Scores)

        allDeviceScores.append(S)
    print('\n', modes)
    S0 = allDeviceScores[0][0] # for the keys
    for key, v in S0.items():
        for imode in range(len(modes)):
            mode = modes[imode]
            scores = [allDeviceScores[imachine][imode][key] for imachine in range(len(devices))]
            print(mode)
            sizes, vals = [[x[i] for x in scores] for i in [0, 1]]
            print(key, 'val:', numpy.mean(vals), numpy.std(vals))
            print(key, 'size:', numpy.mean(sizes), numpy.std(sizes))



"""

if prepareData:
    with open(outdir+'datasets.npy', 'rb') as f:
        datasets = pickle.load(f)
    
    # select ideal exp
    keys = []
    for key, val in datasets.items():
        if key.split(sep='_')[0] == 'ideal':
            keys.append(key)
    #print(keys)
    
    # split ideal keys
    #choice = numpy.random.choice(keys, len(keys), replace=False)
    #trainKeys = choice[:int(len(choice) * 1/3)]
    #calKeys = choice[int(len(choice) * 1/3):int(len(choice) * 2/3)]
    #testKeys = choice[int(len(choice) * 2/3):]

    # non-exchangeable split ideal keys
    #choice = numpy.random.choice(keys, len(keys), replace=False)
    print(keys)
    q = nSteps - 1 
    testKeys = [key for key in keys if int(key.split(sep='_')[3]) == q]
    calKeys = [key for key in keys if int(key.split(sep='_')[3]) 
                                       in [q-1]]
    trainKeys = [key for key in keys if int(key.split(sep='_')[3]) < q-1]
    print(testKeys[:10])
    print(calKeys[:10])
    #testIndexes = [n for n in choice[int(len(choice) * 3/4):]]
    #trainIndexes = [n for n in range(len(Outputs)) if Features[n][0] < nQubits[-1]]
    #testIndexes = [n for n in range(len(Outputs)) if Features[n][0] == nQubits[-1]]

    allIdealKeys = [trainKeys, calKeys, testKeys]
        

if computeDistances:
    # create measure keys
    measureKeys = []
    for distance in distances:
        for features in featureMaps:
            for method in methods:
                key = distance+'_'+features+'_'+method
                measureKeys.append(key)

    for idata in range(len(dataNames)):
        idealKeys = allIdealKeys[idata]
        M, shotKeys = {}, {}
        for mkey in measureKeys:
            print(mkey)
            distance, features, method = mkey.split(sep = '_')
            f = distanceFunctions[distance]
            M[mkey] = []
            shotKeys[mkey] = []
            for key in idealKeys:
                ibmkey = 'ibmq'+key[5:]
                s = [datasets[key], datasets[ibmkey]]
                fakekey = 'fake'+key[5:]
                print(mkey, ':', key, ibmkey, fakekey)
                shotKeys[mkey].append([key, ibmkey, fakekey])
                dim = -1 + 3 * (features != 'none')
                X1, X2 =reduceDimensions(s[0], s[1], dim, features)
                M[mkey].append(f(X1, X2, method, None))

        with open(outdir+'allmeasures_'+dataNames[idata]+'.npy', 'wb') as f:
            pickle.dump(M, f)
        with open(outdir+'shotKeys_'+dataNames[idata]+'.npy', 'wb') as f:
            pickle.dump(shotKeys, f)

###################################################################
useRF = 0
if run:
    with open(outdir+'datasets.npy', 'rb') as f:
        datasets = pickle.load(f)
    S = []
    for useRF in [0, 1, 2]:
        low = {}
        high = {}
        Scores = {}
        good = {}
        RFs = {}
        normalizations = {}
        Q = {}
        
        allMeasures = []
        allShotKeys = []
        for dataname in dataNames:
            with open(outdir+'allmeasures_'+dataname+'.npy', 'rb') as f:
                measures = (pickle.load(f))
            with open(outdir+'shotKeys_'+dataname+'.npy', 'rb') as f:
                shotkeys = (pickle.load(f))
            allMeasures.append(measures)
            allShotKeys.append(shotkeys)

        if useRF == 2:
            temp = allMeasures[0]
            allMeasures[0] = allMeasures[1]
            allMeasures[1] = temp
            temp = allShotKeys[0]
            allShotKeys[0] = allShotKeys[1]
            allShotKeys[1] = temp

        for idataname in range(len(dataNames)):
            measures = allMeasures[idataname]
            dataname = dataNames[idataname]
            shotkeys = allShotKeys[idataname]


            for key, val in measures.items():
                print(key, ':', numpy.max(measures[key]))
                if dataname == 'train':
                    normalizations[key] = numpy.median(measures[key])
                # retrive ibm shots 
                shots = []
                for datakeys in shotkeys[key]:
                    shots.append(datasets[datakeys[1]])
                # normalize measures
                y = numpy.array(measures[key])#/normalizations[key]
                # create attributes
                cut = 2
                means = [numpy.mean(x[:, :cut], axis=0).tolist() for x in shots]
                covs  = [numpy.cov(x[:, :cut].T).reshape(cut**2).tolist() for x in shots]
                X = [numpy.array(means[i] + covs[i]) for i in range(len(covs))]
                #X = [numpy.array(means[i]) for i in range(len(means))]
                
                if dataname == 'train' and useRF == 1:
                    # train model (training set only)
                    RFs[key] = sklearn.ensemble.RandomForestRegressor() 
                    RFs[key].fit(X, y)
                    haty = RFs[key].predict(X) 
                    print('train ER', numpy.mean((haty - y)**2))#/(numpy.mean(y)**2))
                
                if dataname == 'calibration':
                    # CP calibration (calibration set only)
                    if useRF == 1: 
                        haty = RFs[key].predict(X) 
                        print('cal ER', numpy.mean((haty - y)**2))#/(numpy.mean(y)**2))
                        a = y - haty #/abs(haty)
                    else:
                        a = y
                    na = int(numpy.ceil((1 - alpha) * (len(a) + 1)))
                    Q[key] = numpy.sort(a)[na-1] 
                    print('q, max:', Q[key], numpy.max(a))
                
                if dataname == 'test':
                    # CP evaluation (test set only)
                    if useRF == 1:
                        haty = RFs[key].predict(X) 
                        print('test ER', numpy.mean((haty - y)**2))#/(numpy.mean(y)**2))
                        a = y - haty#/abs(haty)
                    else:
                        haty = 0 * numpy.ones(len(y))
                        a = y
                    low[key], high[key] = [haty  - Q[key], 
                                           haty + Q[key]]
                    
                    good[key] = (a <=Q[key])

                    #size = numpy.mean(2 * Q[key])
                    #size = numpy.mean(high[key]-low[key])#/abs(numpy.mean(measures[key]))
                    size = numpy.mean(high[key])#/abs(numpy.mean(measures[key]))
                    val = numpy.mean(1 * good[key])
                    Scores[key] = [size, val]

        for key, v in Scores.items():
                print(key, ':', v)
        S.append(Scores)
    
    modes = ['mondrian', 'shift + mondrian', 'exchange']
    Scores = S[0]
    ScoresRF = S[1]
    for key, v in Scores.items():
        for i in range(len(modes)):
            mode = modes[i]
            print(mode)
            print(key, 'val:', S[i][key][1])
            print(key, 'size:', S[i][key][0])
"""


