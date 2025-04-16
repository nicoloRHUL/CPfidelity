import numpy
import subprocess
import os

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
backends = [
            FakeBrisbane(),  # type: ignore
            FakeCusco(),  # type: ignore
            FakeKawasaki(),  # type: ignore
            FakeKyiv(),  # type: ignore
            FakeKyoto(),  # type: ignore
            FakeOsaka(),  # type: ignore
            FakeQuebec(),  # type: ignore
            FakeSherbrooke(),  # type: ignore
            FakeWashingtonV2()  # type: ignore
        ]

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

pwd='C:/Users/ugqm002/OneDrive - Royal Holloway University of London/research/conformalPredictors/Quantum/CPfromSamples/Output/IBM/' 
pwdQasm='C:/Users/ugqm002/OneDrive - Royal Holloway University of London/research/conformalPredictors/Quantum/CPfromSamples/Data/Dataset/' 
#############################i#######

#martina et al circuit
#####################################
# Generate random circuit

nShots = 1000
nQbits = [3, 5, 7, 9, 11, 13, 15]
tasks = [0, 1, 2, 3, 4, 5, 10, 11, 12]#range(11)
devices = [1, 2, 3, 4, 5] #max 9
for device in devices:
    deviceName = backends[device]
    for t in tasks:
        for n in nQbits:
            circ = loadCircuit(n, t)
            
            device_backend = backends[device]
            print('device:', device_backend.name, 'qubits:', n, 'task:', circuits[indexes[str(t)]])
            filename = circuits[indexes[str(t)]] + str(n)

            sim_ideal = AerSimulator()#method='extended_stabilizer')
            iCirc = transpile(circ, sim_ideal)
            result_ideal = sim_ideal.run(iCirc, shots=nShots, memory=True).result()
            #counts_ideal = result_ideal.get_counts(iCirc)
            memory_ideal = result_ideal.get_memory(iCirc)
            directory = pwd+ filename
            outputFile = directory + '/ideal_' + device_backend.name + '_' + filename + '.out'
            numpy.savetxt(outputFile, memory_ideal, fmt='%s')#, delimiter=',')

            # Fake-IBM noisy simulator
            sim_IBM = AerSimulator.from_backend(device_backend)
            vCirc = transpile(circ, sim_IBM)
            result_noise = sim_IBM.run(vCirc, shots=nShots, memory=True).result()
            #counts_noise = result_noise.get_counts(vCirc)
            memory_IBM = result_noise.get_memory(vCirc)
            directory = pwd + filename
            outputFile = directory + '/ibm_' + device_backend.name + '_' + filename + '.out'
            numpy.savetxt(outputFile, memory_IBM, fmt='%s')#, delimiter=',')




