from pyquil.paulis import PauliTerm, PauliSum, exponential_map
from pyquil.api import WavefunctionSimulator

# More pyquil
from pyquil import Program
from pyquil.gates import RX, RZ, CNOT

# Numpy and Scipy
import numpy as np
from numpy import random
from scipy.optimize import minimize

# Globals
sim = WavefunctionSimulator(random_seed=1337)

def solve_vqe(hamiltonian: PauliSum) -> float:
    '''
    Can assume that the PauliSum hamiltonian is defined on at most 5 qubits.
    '''
    # Hyperparameters
    NUM_LAYERS = 5 # Number of ansatz layers

    # Returns a list of the qubit indices which the PauliSum acts on
    qubits = hamiltonian.get_qubits()
    num_qubits = len(qubits)
    
    # Initialize theta
    theta = init_theta(num_qubits, NUM_LAYERS)
    min_result = minimize(expectation, np.asarray(theta.flat), args=(num_qubits, NUM_LAYERS, hamiltonian), method='Nelder-Mead')

    return min_result.fun

def construct_ansatz(theta, num_qubits, num_layers):
    '''
    Constructs the ansatz based on the example provided in the new project spec.
    '''
    theta = np.reshape(theta, (num_qubits, num_layers))

    program = Program()

    for l in range(num_layers):
        for q in range(num_qubits): # Assumes qubits are ordered

            program += RX(theta[q, l], q)
            program += RZ(theta[q, l], q)

            if (q + 1) < num_qubits:
                program += CNOT(q, q+1)

    return program

def init_theta(num_qubits, num_layers, seed=269):
    '''
    Initialize parameters to be optimized.

    Theta is shape (num_qubits x num_layers)
    '''

    # Set the seed for consistent results.
    random.seed(seed)

    # Initialize the parameters to small, random values.
    return random.normal(loc=np.pi, scale=1.0, size=(num_qubits, num_layers))

def expectation(theta, num_qubits, num_layers, H_t):
    '''
    Callable function for minimization.
    '''
    return sim.expectation(construct_ansatz(theta, num_qubits, num_layers), H_t)
