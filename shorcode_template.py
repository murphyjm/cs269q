from typing import List
import numpy as np

from pyquil import Program
from pyquil.gates import MEASURE, I
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection

##
############# YOU MUST COMMENT OUT THESE TWO LINES FOR IT TO WORK WITH THE AUTOGRADER
# import subprocess
# subprocess.Popen("/src/qvm/qvm -S > qvm.log 2>&1", shell=True)


# Do not change this SEED value you or your autograder score will be incorrect.
qvm = QVMConnection(random_seed=1337)


def bit_flip_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob) * np.asarray([[0, 1], [1, 0]])
    return [noisy_I, noisy_X]


def phase_flip_channel(prob: float):
    pass


def depolarizing_channel(prob: float):
    pass


def bit_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):

    ### Do your encoding step here
    code_register = None  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = None  # the Program that does the encoding

    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)


    ### Do your decoding and correction steps here

    return pq, code_register


def phase_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    ### Do your encoding step here
    code_register = None  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = None  # the Program that does the encoding

    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)

    ### Do your decoding and correction steps here

    return pq, code_register


def shor(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    # Note that in order for this code to work properly, you must build your Shor code using the phase code and
    # bit code methods above
    pass


def run_code(error_code, noise, trials=10):
    """ Takes in an error_code function (e.g. bit_code, phase_code or shor) and runs this code on the QVM"""
    pq, code_register = error_code(QubitPlaceholder(), noise=noise)
    ro = pq.declare('ro', 'BIT', len(code_register))
    pq += [MEASURE(qq, rr) for qq, rr in zip(code_register, ro)]

    return qvm.run(address_qubits(pq), trials=trials)


def simulate_code(kraus_operators, trials, error_code) -> int:
    """
    :param kraus_operators: The set of Kraus operators to apply as the noise model on the identity gate
    :param trials: The number of times to simulate the program
    :param error_code: The error code {bit_code, phase_code or shor} to use
    :return: The number of times the code did not correct back to the logical zero state for "trials" attempts
    """
    # Apply the error_code to some qubits and return back a Program pq
    pq = None

    # THIS CODE APPLIES THE NOISE FOR YOU
    kraus_ops = kraus_operators
    noise_data = Program()
    for qq in range(3):
        noise_data.define_noisy_gate("I", [qq], kraus_ops)
    pq = noise_data + pq

    # Run the simulation trials times using the QVM and check how many times it did not work
    # return that as the score. E.g. if it always corrected back to the 0 state then it should return 0.
    score = None
    return score
