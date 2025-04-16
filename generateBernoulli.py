import numpy
import subprocess
import os
import glob
import pickle
import matplotlib.pyplot as plt

seed = 1234

from functions import *

generate = 0
computeDistances = 1
producePlot = 1
plotWeights = 1
distanceFunctions = {
        'KL' : KL, 
        'BC' : BC, 
        'TV' : TV}
#, 'hotelling' : hotelling, 'FID' : FID }
distances = ['BC', 'KL', 'TV']#, 'hotelling', 'FID']
methods = ['ratio']#, 'hist']

###################################3
# bernoulli 
machines = ['log', 'rand', 'cos']#, 'linear']
perturbations = ['plog', 'prand', 'pcos']#, 'plinear']
nQubits = [10, 20, 40, 80]
nExp = 5
noise = 0.1
nShots = 1000

if plotWeights:
    with open('Outputs/Bernoulli/allweights0.npy', 'rb') as f:
        allW = pickle.load(f)
    for key, val in allW.items():
        plt.plot(val[0], 'r')
        plt.plot(val[1], 'b')
    plt.show()

if generate:
    allbc = {}
    for iExp in range(nExp):
        print(seed * iExp)
        numpy.random.seed(seed * iExp)
        allshots = {}
        allweights = {}
        for machine in machines:
            for perturbation in perturbations:
                for nq in nQubits: 
                    w = createW(machine, nq)
                    hw = perturbW(w, perturbation, noise)
                    s = [sampler(v ,nShots) for v in [w, hw]]
                    key = machine+'_'+perturbation+'_'+str(nq)
                    allshots[key] = s
                    allweights[key] = [w, hw]
                    if iExp == 0:
                        allbc[key] = [trueBC(w, hw)]
                    else:
                        allbc[key].append(trueBC(w, hw))
                    print(key)
        with open('Outputs/Bernoulli/allshots'+str(iExp)+'.npy', 'wb') as f:
            pickle.dump(allshots, f)
        with open('Outputs/Bernoulli/allweights'+str(iExp)+'.npy', 'wb') as f:
            pickle.dump(allweights, f)
    with open('Outputs/Bernoulli/allbc'+str(iExp)+'.npy', 'wb') as f:
        pickle.dump(allweights, f)
    for key, val in allbc.items():
        x, y = numpy.mean(val), numpy.std(val)
        s = str(numpy.round(x, 3))+'('+str(numpy.round(x, 3))+')'
        print(key, ':', s)
        
##############################################################

if computeDistances:
    for iExp in range(nExp):
        print(seed * iExp)
        numpy.random.seed(seed * iExp)

        with open('Outputs/Bernoulli/allshots'+str(iExp)+'.npy', 'rb') as f:
            allshots = pickle.load(f)
        with open('Outputs/Bernoulli/allweights'+str(iExp)+'.npy', 'rb') as f:
            allweights = pickle.load(f)

        M = {}
        for distance in distances:
            for method in methods:
                if method in ['ratio', 'hist'] :
                    if distance in ['BC', 'KL', 'TV']:
                        key = distance+'_'+method
                        M[key] = []
                else:
                    key = distance+'_'+method
                    M[key] = []
        M['ref'] = []

        shotKeys = []
        for key, value in allshots.items():
            print(key)
            D = allshots[key]
            W = allweights[key]
            D = [[str(x) for x in s] for s in D]
            s = [[[int(x[i]) for i in range(len(x))]for x in d] for d in D]
            
            shotKeys.append(key)
            for mkey, mvalue in M.items():
                if mkey == 'ref':
                    M['ref'].append(trueBC(W[0], W[1]))
                    print('true BC:', M['ref'][-1])
                else:
                    distance, method = mkey.split(sep = '_')
                    f = distanceFunctions[distance]
                    #X1, X2 =reduceDimensions(s[0], s[1], -1, None)
                    M[mkey].append(f(s[0], s[1], method, W))
                

        with open('Outputs/Bernoulli/allmeasures'+str(iExp)+'.npy', 'wb') as f:
            pickle.dump(M, f)
        with open('Outputs/Bernoulli/shotKeys'+str(iExp)+'.npy', 'wb') as f:
            pickle.dump(shotKeys, f)

if producePlot:
    import random
    
    with open('Outputs/Bernoulli/allmeasures0.npy', 'rb') as f:
        M = pickle.load(f)
    # select scores
    number_of_colors = len(M)
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
             for i in range(number_of_colors)]
    scores = {}
    colors = {}
    t = 0
    for key, val in M.items():
            scores[key] = []
            colors[key] = color[t]
            t = t + 1

    for iExp in range(nExp):
        with open('Outputs/Bernoulli/allmeasures'+str(iExp)+'.npy', 'rb') as f:
            M = pickle.load(f)
        X = numpy.array(M['ref'])
        X = X/sum(X)
        order = numpy.argsort(X)
        for key, val in scores.items():
            Y = numpy.array(M[key])
            Y = Y/sum(Y)
            scores[key].append(numpy.corrcoef(X, Y)[0, 1])
            if key != 'ref':
                if iExp > (nExp - 2):
                    mean, std = numpy.mean(scores[key]), numpy.std(scores[key])
                    x, y = mean, std
                    s = str(numpy.round(x, 3))+'('+str(numpy.round(y, 3))+')'
                    k = key.split(sep='_')
                    label = ''.join([k[0], '-', k[1]])+': '+s
                    print(key, label)
                    plt.scatter(X[order], Y[order], color=colors[key], label=label)
                else:
                    plt.scatter(X[order], Y[order], color=colors[key])
    
    plt.plot([min(X), max(X)], [min(X), max(X)], 'k--', alpha=0.5)
    plt.legend()
    plt.xlabel('true BC')
    plt.ylabel('estimated distance')
    filename='Outputs/Bernoulli/bernoulli.pdf'
    plt.savefig(filename)
    plt.show()
