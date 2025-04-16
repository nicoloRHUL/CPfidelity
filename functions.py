import numpy

import scipy 
from scipy.optimize import minimize
from scipy.linalg import sqrtm

import sklearn
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression

eps = 1e-4
reg = 1e-2
pwd = ''

########################
def minMax(x):
    return (numpy.array(x) - min(x))/(max(x) - min(x))

def relu(x):
    return x * (x > 0) + 0 * (x <= 0)

def dist(X, D):
    return numpy.diag((X - D) @ (X - D).T)

def covariance(D1, D2 = None):
    if D2 == None: D2 = D1
    return numpy.mean([numpy.outer(D1[i], D2[i]) for i in range(len(D1))], 
            axis = 0)

def trace(A):
    return numpy.sum(numpy.diag(A))

######################################
def bernoulliDensity(Y, theta):
    p = []
    for y in Y:
        log = 0
        for i in range(len(y)):
            log = log + y[i] * theta[i] + (1 - y[i]) * (1 - theta[i])
        p.append(log)
    return numpy.exp(numpy.array(p))

###############################################
# Generate random circuit
def generateRandomCircuit(nQbits):
    circ = random_circuit(nQbits, nQbits, measure=True)
    return circ

def generateDeepRandomCircuit(nQbits):
    circ = random_circuit(nQbits, 3 * nQbits, measure=True)
    return circ

def walker(step):
    circ = QuantumCircuit(4, 4)
    for i in range(step):
        if i%3 == 0:
            circ.h(0)
            circ.h(1)
            circ.cx(0,2)

            circ.barrier(0,1,2,3)
        elif i%3 ==1:
            circ.cx(1,3)
            circ.x(0)

            circ.barrier(0,1,2,3)
        else:
            circ.x(1)
            circ.ccx(0,1,2)

            circ.barrier(0,1,2,3)
    circ.measure(2, 0)
    circ.measure(3, 1)
    circ.measure(1, 2)
    circ.measure(2, 3)
    return circ

def loadCircuit(nQbits, task):
    if task == 10:
        filename='random_'+ str(nQbits)
        circ = generateRandomCircuit(nQbits)
    if task == 11:
        filename='deep_random_'+ str(nQbits)
        circ = generateDeepRandomCircuit(nQbits)
    if task == 12:
        filename='martina_'+ str(nQbits)
        circ = walker(nQbits)
    if task < 10:
        filename = circuits[indexes[str(task)]] + str(nQbits)
        print('loading ' + filename)
        circ = qiskit.qasm2.load(pwdQasm + filename + '.qasm')
    
    directory = pwd+filename
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('make dir')
    return circ

#############################################
def cut(w):
    return w * (w < 1) + 1 * (w >= 1)

def perturbW(w, f, noise = 1/10):
    w0 = w
    c = numpy.random.rand()
    if f == 'plog':
        w = [x + numpy.log(eps + x) for x in w]
    if f == 'plinear':
        w = [1 + c * x for x in w]
    if f == 'pcos':
        w = [numpy.cos(x * 2 * 3.14) for x in w]
    if f == 'prand':
        w = [x + x * numpy.random.randn() for x in w]
    w = numpy.array([abs(x) for x in w]) + reg
    w = cut(w/max(w))
    w = (1 - noise) * w0 + noise * w
    return w

def createW(g, nQubits):
    c = numpy.random.rand()
    if g == 'log':
        w = [numpy.log(1 + i * c)/nQubits for i in range(nQubits)]
    if g == 'rand':
        w = [numpy.random.rand() * i/nQubits for i in range(nQubits)]
    if g == 'cos':
        w = [numpy.cos(1/(eps + i * c) * 3.14)**2 for i in range(nQubits)]
    if g == 'linear':
        w = [c + i/nQubits for i in range(nQubits)]
    w = numpy.array([abs(x) for x in w]) + reg
    w = cut(w/max(w))
    return w

def generateProbabilities(nQ, icircuit, imachine, noise):    
    filename = circuits[indexes[str(icircuit)]] + str(nQ)
    directory = expDir + filename
    wIdeal = fakeCircuit(circuits[indexes[str(icircuit)]], nQ)
    wNoise = perturb(backends[imachine], wIdeal, noise) 
    circ = [wIdeal, wNoise]
    if not os.path.exists(directory):
        os.makedirs(directory)
    return circ


####################################################
def ratioEstimation(X1, X2):
    X = numpy.concatenate([X1, X2], axis = 0)
    y = [1 for x in X1] + [0 for x in X2]
    clf = LogisticRegression(random_state=0).fit(X, y)
    q = clf.predict_proba(X1)
    r = q[:, 0]/(eps + q[:, 1])
    return r
####################################

def BC(X1, X2, method = 'ratio',  thetas = None):
    r2 = ratioEstimation(X1, X2) 
    return - numpy.log(numpy.mean(numpy.sqrt(r2)))

def KL(X1, X2, method = 'ratio',  thetas = None):
    r2 = ratioEstimation(X1, X2) 
    return - numpy.mean(numpy.log(r2 + eps))

def TV(X1, X2,  method = 'ratio',  thetas = None):
    r2 = ratioEstimation(X1, X2) 
    return numpy.mean(abs(1 - r2))
   
def trueBC(theta1, theta2):
    log = 0
    for i in range(len(theta1)):
        t1, t2 = theta1[i], theta2[i]
        if (t1 < 0) + (t2 < 0) or (t1 > 1) + (t2 > 1):
            print('problem', theta1, theta2)
        log = log + numpy.log(eps + numpy.sqrt(t1 * t2) + numpy.sqrt((1 - t1)*(1 - t2)))
    return -log

def FID(X1, X2, method = None, thetas = None):
    if len(X1[0]) > 10:
        phi1 = sklearn.decomposition.PCA(n_components = 10)#, tol = 0.01) 
        phi2 = sklearn.decomposition.PCA(n_components = 10)#, tol = 0.01) 
        n, d = X1.shape
        X1 = phi1.fit_transform(X1 + eps * numpy.random.randn(n, d))
        X2 = phi1.transform(X2)

    m1, m2 = numpy.mean(X1, axis = 0), numpy.mean(X2, axis = 0)
    s1, s2 = covariance(X1 - m1), covariance(X2 - m2)
    tr1, tr2, tr12 = trace(s1), trace(s2), numpy.sqrt(trace(s1 @ s2))
    FID = (m1 - m2).T @(m1 - m2) + tr1 + tr2 - 2 * tr12
    return FID

def hotelling(X1, X2, method = None, thetas = None):
    if len(X1[0]) > 10:
        phi1 = sklearn.decomposition.PCA(n_components = 10)#, tol = 0.01) 
        phi2 = sklearn.decomposition.PCA(n_components = 10)#, tol = 0.01) 
        n, d = X1.shape
        X1 = phi1.fit_transform(X1 + eps * numpy.random.randn(n, d))
        X2 = phi1.transform(X2)

    m1, m2 = numpy.mean(X1, axis = 0), numpy.mean(X2, axis = 0)
    s1, s2 = covariance(X1 - m1), covariance(X2 - m2)
    n1, n2 = len(s1), len(s2)
    sPooled = (n1 * s1 + n2 * s2)/(n1 + n2)
    one = numpy.eye(len(sPooled))
    Z = (m1 - m2).T @ numpy.linalg.pinv(sPooled + reg * one) @ (m1 - m2)
    return Z


