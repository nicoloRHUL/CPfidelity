import numpy
import subprocess
import os
import glob
import pickle

import qiskit 
import matplotlib.pyplot as plt

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

pwd=''
####################################

#martina et al circuit
#####################################
# Generate random circuit
def generateRandomCircuit(nQbits):
    circ = random_circuit(nQbits, nQbits, measure=True)
    return circ


def walkersimple(step): # martina's walker simple
    circ = QuantumCircuit(4, 4)
    for i in range(step):
        if i%2 == 0:
            circ.h(0)
            circ.cx(0,2)
            circ.barrier(0,1,2,3)
        else:
            circ.h(1)
            circ.cx(1,3)
            circ.barrier(0,1,2,3)

    for q in range(len(circ.qubits)):
        circ.measure(q, q)

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
    return circ

basePath = 'Data/walker'
machines = ['athens', 'casablanca', 'lima', 'quito', 'santiago', '5_yorktown']
prefixes = ['ibmq_', 'fake_', 'ideal_']

backends = {
        'athens': FakeAthensV2(),
        'casablanca': FakeCasablancaV2(),
        'lima': FakeLimaV2(),
        'quito': FakeQuitoV2(),
        'santiago': FakeSantiagoV2(),
        '5_yorktown': FakeYorktownV2() 
        }
#outdir = 'C:/Users/ugqm002/Documents/Temp/'
outdir = 'Output/Martina/Steps/'

nShots = 1000
nSteps = 9
nExecutions = 10
for im in range(len(machines)):
    
    machine, fakeMachine, idealMachine = [prefixes[i]+machines[im] for i in [0, 1, 2]]
    sim_IBM = AerSimulator.from_backend(backends[machines[im]])
    sim_ideal = AerSimulator()
    
    executions = []
    nE = 0
    for filename in sorted(glob.glob(os.path.join(basePath, '{}-'.format(machine)+('[0-2]'*6)+'.p'))):
        if nE < nExecutions:
            if nE == 0: 
                print('true device:', machine, ' Data loaded from', filename)
                print('fake device:', fakeMachine, 'Data simulated with', sim_IBM.name)
                print('ideal device:', idealMachine, 'Data simulated with', sim_ideal.name)
            nE = nE + 1
            
            results = pickle.load(open(filename, 'rb'))
            if results['results'][0]['success'] == False:
                print('filename failed')

            steps = []
            for iStep in range(nSteps): 
                print('(execution, step):', nE, iStep)
                # load hardware data
                currExecution = []
                for t in range(nShots): #range(len(results['results'][n]['data']['memory'])):
                    execution = ['0 0', '0 1', '1 0', '1 1'][int(results['results'][iStep]['data']['memory'][t], 0)]
                    currExecution.append(execution)
                fileOut = outdir+machine+'_step_'+str(iStep)+'_'+'execution_' +str(nE) + '.out'
                numpy.savetxt(fileOut, currExecution, fmt='%s')
    
                
                circ = walker(iStep + 1)
                
                # create fake data
                ibmCirc = transpile(circ, sim_IBM)
                result = sim_IBM.run(ibmCirc, shots=nShots, memory=True).result()
                memory = result.get_memory(ibmCirc)
                memory = [str(x[2])+' '+str(x[3]) for x in memory]
                fileOut = outdir+fakeMachine+'_step_'+str(iStep)+'_'+'execution_' +str(nE) + '.out'
                numpy.savetxt(fileOut, memory, fmt='%s')

                # create idea data
                idealCirc = transpile(circ, sim_ideal)
                result = sim_ideal.run(idealCirc, shots=nShots, memory=True).result()
                memory = result.get_memory(idealCirc)
                memory = [str(x[2])+' '+str(x[3]) for x in memory]
                fileOut = outdir+idealMachine+'_step_'+str(iStep)+'_'+'execution_' +str(nE) + '.out'
                numpy.savetxt(fileOut, memory, fmt='%s')


"""
circ = transpile(circ, simulator)
# Run and get counts
result = simulator.run(circ, shots=10, memory=True).result()
memory = result.get_memory(circ)
print(memory)


# Execute and get counts
result = sim_ideal.run(transpile(circ, sim_ideal)).result()
counts = result.get_counts(0)
plot_histogram(counts, title='Ideal counts for 3-qubit GHZ state')


simulator = AerSimulator()
print(simulator.available_devices())
print(simulator.available_methods())


circ = walker(4)
circ.draw(output='mpl')
plt.show()

"""
