from pyquil.paulis import PauliTerm, PauliSum, exponential_map
from pyquil.api import WavefunctionSimulator

# More pyquil
from pyquil import Program
from pyquil.gates import RX, RZ, CNOT

# Numpy and Scipy
import numpy as np
from numpy import random
from scipy.optimize import minimize

# Debugging
from IPython.core.debugger import set_trace

# Progress bar *****DELETE BEFORE SUBMISSION*****
from tqdm import tqdm

# Globals
sim = WavefunctionSimulator(random_seed=1337)

def solve_vqe(hamiltonian: PauliSum) -> float:
    '''
    Can assume that the PauliSum hamiltonian is defined on at most 5 qubits.

    Need to choose annealing schedule.
    '''
    # Hyperparameters
    # TOL = 1e-8      # Convergence tolerance
    MAX_STEPS = 1e2 # Max number of steps
    NUM_LAYERS = 5 # Number of ansatz layers
    SCHEDULE   = 'linear' # Annealing schedule type

    # Returns a list of the qubit indices which the PauliSum acts on
    qubits = hamiltonian.get_qubits()
    num_qubits = len(qubits)

    # Initialize the driver hamiltonian as a PauliSum of Pauli-X terms
    # pauli_sum_seq = [PauliTerm('X', q) for q in qubits]
    # hamiltonian_drive = PauliSum(pauli_sum_seq)

    # Get annealing schedule
    # annealing_schedule = get_annealing_schedule(SCHEDULE, MAX_STEPS)

    # Initialize theta
    theta = init_theta(num_qubits, NUM_LAYERS)

    min_result = minimize(expectation, np.asarray(theta.flat), args=(num_qubits, NUM_LAYERS, hamiltonian), method='Nelder-Mead')
    return min_result.x, min_result.fun

    # # Optimization loop
    # lambda_t_plus_1 = 0 # Placeholder value
    #
    # # lambda_arr just for debugging *****DELETE BEFORE SUBMITTING*****
    # lambda_arr = np.zeros(int(MAX_STEPS))
    #
    # for i in tqdm(range(int(MAX_STEPS))):
    #
    #     # Update
    #     lambda_t = np.copy(lambda_t_plus_1)
    #     # Step towards problem hamiltonian following annealing schedule
    #     H_t = float(annealing_schedule[i]) * hamiltonian_drive + (1 - float(annealing_schedule[i])) * hamiltonian
    #     # Minimize expectation
    #     min_result = minimize(expectation, np.asarray(theta.flat), args=(num_qubits, NUM_LAYERS, H_t), method='Nelder-Mead')
    #     # Save result
    #     lambda_t_plus_1 = min_result.fun
    #     # Update theta
    #     theta = min_result.x
    #
    #     # Add value to lambda array
    #     lambda_arr[i] = lambda_t_plus_1
    #
    # return lambda_t_plus_1, np.arange(int(MAX_STEPS)), lambda_arr

def get_annealing_schedule(schedule, MAX_STEPS):
    '''
    Generates an array for the weights of the annealing schedule.

    Array returned should be used for the weights of the driver hamiltonian.
    '''
    if schedule == 'linear':
        # First try: linear schedule
        m = -1/MAX_STEPS
        return np.asarray(m * np.linspace(0, MAX_STEPS, MAX_STEPS) + 1)
    else:
        return None

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
    return random.normal(scale=1.0, size=(num_qubits, num_layers))

def expectation(theta, num_qubits, num_layers, H_t):
    '''
    Callable function for minimization.
    '''
    return sim.expectation(construct_ansatz(theta, num_qubits, num_layers), H_t)
