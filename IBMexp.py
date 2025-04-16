import numpy
import pickle

import matplotlib.pyplot as plt
seed = 12345
numpy.random.seed(seed)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

from functions import *

loadData = 1
computeDistances = 1
run = 0

#nQubits = [3, 5, 7, 9, 11, 13, 15]
nQubits = [5, 7, 9, 11, 13, 15]
tasks = [0, 1, 3, 5, 10, 11, 12]
goodDevices = [1, 2, 3, 4, 5] 
#goodDevices = [1, 2, 3]#, 4, 5] 
indexes = {
        '0':'W-State',
        '1': 'Portfolio Optimization with VQE',
        '2': 'Deutsch-Jozsa', 
        '3':'Graph State',
        '4':'GHZ State',
        '5':'Variational Quantum Eigensolver (VQE)',
        '6': 'Quantum Walk (no ancilla)',
        '7': 'Quantum Walk (v-chain)',
        '8': 'Grovers (v-chain)',
        '9': 'Grovers (no anchilla)',
        '10': 'Random',
        '11': 'Deep Random',
        '12': 'Martina'
        }

circuits = {
        'Deutsch-Jozsa': 'dj_indep_qiskit_',
        'Graph State': 'graphstate_indep_qiskit_',
        'GHZ State': 'ghz_indep_qiskit_',
        'Quantum Fourier Transformation (QFT)': 'qft_indep_qiskit_',
        'W-State':  'wstate_indep_qiskit_', 
        'Variational Quantum Eigensolver (VQE)':'vqe_indep_qiskit_', #3-16
        'Quantum Walk (no ancilla)': 'qwalk-noancilla_indep_qiskit_', #3-15
        'Quantum Walk (v-chain)': 'qwalk-v-chain_indep_qiskit_', #3-5-7-...-27
        'Portfolio Optimization with VQE':'portfoliovqe_indep_qiskit_',#3-18
        'Grovers (v-chain)': 'grover-v-chain_indep_qiskit_', #3-5-..-19
        'Grovers (no anchilla)': 'grover-noancilla_indep_qiskit_', #2-3-...12
        'Random': 'random_',
        'Deep Random': 'deep_random_',
        'Martina': 'martina_'
        }

devices = [
            #'fake_brisbane',  # type: ignore
            'fake_cusco',  # type: ignore
            'fake_kawasaki',  # type: ignore
            'fake_kyiv',  # type: ignore
            'fake_kyoto',  # type: ignore
            'fake_osaka',  # type: ignore
            #'fake_quebec',  # type: ignore
            #'fake_sherbrooke',  # type: ignore
            #'fake_washingtonV2'  # type: ignore
            ]
tasks = [
         'W-State',
         'Portfolio Optimization with VQE',
         'Deutsch-Jozsa', 
         'Graph State',
         'GHZ State',
         'Variational Quantum Eigensolver (VQE)',
         'Random',
         'Deep Random',
         'Martina'
    ]

distances = ['KL', 'BC', 'TV', 'hotelling', 'FID']
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
#pwd='/home/nicolo/Documents/Work/Research/CP/SampleBasedCP/'
#pwd = ' C:\\Users\\ugqm002\\OneDrive - Royal Holloway University of London\\research\\conformalPredictors\\Quantum/CPfromSamples\\'
pwd = ''
nShots = 1000
alpha = 0.1
##################################3
# Load dataset from Output
if loadData:    
    datasets = {}
    for device in devices:
        for task in tasks:
            circuit = circuits[task]
            for nQ in nQubits:
                #key = device+'-'+circuit+'-'+str(nQ)
                #print('device:', device, 
                #      ', qubits:', nQ, 
                #      ', task:', circuit)
                expName = circuit + str(nQ)
                directory = 'Data/Output/IBM/'+expName
                for prefix in ['ideal_', 'ibm_']:
                    key = prefix+device+'_'+circuit+str(nQ)
                    datasets[key] = numpy.genfromtxt(
                            directory+'/' +key+'.out', 
                            dtype='int', delimiter=1)
    #key = 'ideal_fake_kyoto_'+circuits['W-State']+'7'
    #print('ideal:', datasets[key][:10])
    #key = 'ibm_fake_kyoto_'+circuits['W-State']+'7'
    #print('IBM:', datasets[key][:10])

    with open('Outputs/IBM/datasets.npy', 'wb') as f:
        pickle.dump(datasets, f)




if computeDistances:
    with open('Outputs/IBM/datasets.npy', 'rb') as f:
        datasets = pickle.load(f)
    
    allDeviceScores = []
    for device in devices:
        # select ideal exp and device
        keys = []
        for key, val in datasets.items():
            k = key.split(sep='_')
            if k[0] == 'ideal' and k[2] != device.split(sep='_')[1]:#k[2] != device.split(sep='_')[1]:
                keys.append(key)
        
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
                ibmkey = 'ibm'+key[5:]
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
                if int(dkey.split(sep='_')[-1]) == nQubits[-1]:
                    data[imode][2][key].append(val)
                    shots[imode][2][key].append(dkey)
            #print(shots[imode][2])
        # cal = all, nqubits[-2], nquabits[-2] 
        print('smax', nQubits[-2])
        for imode in range(len(modes)):
            mode  = modes[imode]
            for comb, val in M.items():
                key, dkey = comb.split(sep='&')
                if mode == 'all':
                    if int(dkey.split(sep='_')[-1]) < nQubits[-1]:
                        data[imode][1][key].append(val)
                        shots[imode][1][key].append(dkey)
                    if int(dkey.split(sep='_')[-1]) < nQubits[-1]:
                        data[imode][0][key].append(val)
                        shots[imode][0][key].append(dkey)
                
                if mode in ['mondrian', 'shift + mondrian']:
                    if int(dkey.split(sep='_')[-1]) in [nQubits[-1], nQubits[-2]]:
                        #nQubits[-2]:#smax - 1:
                        data[imode][1][key].append(val) #calibration
                        shots[imode][1][key].append(dkey) #calibration
                    if int(dkey.split(sep='_')[-1]) < nQubits[-2]:
                        data[imode][0][key].append(val) #train
                        shots[imode][0][key].append(dkey) #train
                
                if mode in ['shift']:
                    if int(dkey.split(sep='_')[-1]) < nQubits[-1]:
                        r = numpy.random.rand() > 0.5
                        if r:
                            data[imode][1][key].append(val) # calibration
                            shots[imode][1][key].append(dkey) # calibration
                        else:
                            data[imode][0][key].append(val)
                            shots[imode][0][key].append(dkey)
               
        with open('Outputs/IBM/data.npy', 'wb') as f:
            pickle.dump(data, f)
        with open('Outputs/IBM/shots.npy', 'wb') as f:
            pickle.dump(shots, f)
        
        with open('Outputs/IBM/datasets.npy', 'rb') as f:
            datasets = pickle.load(f)
        
        with open('Outputs/IBM/data.npy', 'rb') as f:
            data = (pickle.load(f))
        with open('Outputs/IBM/shots.npy', 'rb') as f:
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
                    cut = 4
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


    
