from typing import List
import numpy as np

from pyquil import Program
from pyquil.gates import MEASURE, H, I, CNOT, X
from pyquil.quil import address_qubits
from pyquil.quilatom import QubitPlaceholder
from pyquil.api import QVMConnection

##
############# YOU MUST COMMENT OUT THESE TWO LINES FOR IT TO WORK WITH THE AUTOGRADER
import subprocess
subprocess.Popen("/src/qvm/qvm -S > qvm.log 2>&1", shell=True)


# Do not change this SEED value you or your autograder score will be incorrect.
qvm = QVMConnection(random_seed=1337)


def bit_flip_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob) * np.asarray([[0, 1], [1, 0]])
    return [noisy_I, noisy_X]


def phase_flip_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_Z = np.sqrt(prob) * np.asarray([[1, 0], [0, -1]])
    return [noisy_I, noisy_Z]


def depolarizing_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob/3) * np.asarray([[0, 1], [1, 0]])
    noisy_Z = np.sqrt(prob/3) * np.asarray([[1, 0], [0, -1]])
    noisy_Y = np.sqrt(prob/3) * np.asarray([[0, 0-1.0j], [0+1.0j, 0]])
    return [noisy_I, noisy_X, noisy_Y, noisy_Z]


def bit_code(qubit: QubitPlaceholder, noise=None, roVal = 1) -> (Program, List[QubitPlaceholder]):
    
    ### Do your encoding step here
    q1 = QubitPlaceholder()
    q2 = QubitPlaceholder()
    a0 = QubitPlaceholder()
    a1 = QubitPlaceholder()
    code_register = [qubit, q1, q2]  # the List[QubitPlaceholder] of the qubits you have encoded into
    
    pq = Program(CNOT(code_register[0], code_register[1]), CNOT(code_register[0], code_register[2]))
    # the Program that does the encoding
    
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)


    ### Do your decoding and correction steps here
    ro = pq.declare('flip' + str(roVal), 'BIT', 2)

    pq += Program(CNOT(code_register[0], a0), CNOT(code_register[1], a0))
    pq += MEASURE(a0, ro[0])

    pq += Program(CNOT(code_register[0], a1), CNOT(code_register[2], a1))
    pq += MEASURE(a1, ro[1])

    pq.if_then(ro[0], Program().if_then(ro[1], X(code_register[0]), X(code_register[1])), Program().if_then(ro[1], X(code_register[2])))
    

    return pq, code_register




def phase_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    ### Do your encoding step here
    q1 = QubitPlaceholder()
    q2 = QubitPlaceholder()
    a0 = QubitPlaceholder()
    a1 = QubitPlaceholder()
    code_register = [qubit, q1, q2]  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = Program(CNOT(code_register[0], code_register[1]), CNOT(code_register[0], code_register[2]))
    pq += (H(q) for q in code_register[:3])
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)

    ### Do your decoding and correction steps here
    pq += (H(q) for q in code_register[:3])

    ro = pq.declare('phase', 'BIT', 2)

    pq += CNOT(code_register[0], a0)
    pq += CNOT(code_register[1], a0)
    pq += MEASURE(a0, ro[0])

    pq += CNOT(code_register[0], a1)
    pq += CNOT(code_register[2], a1)
    pq += MEASURE(a1, ro[1])

    pq.if_then(ro[0], Program().if_then(ro[1], X(code_register[0]), X(code_register[1])), Program().if_then(ro[1], X(code_register[2])))

    return pq, code_register


def shor(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    
    #run phase code first
    pq, phase_register = phase_code(qubit, noise = noise)
    code_register = []
    roValue = 1
    for reg in phase_register:
        p, bit_register = bit_code(reg, noise = noise, roVal = roValue)
        pq += p
        code_register += bit_register
        roValue += 1
    return pq, code_register


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
    pq, code_register = error_code(QubitPlaceholder())
    ro = pq.declare('ro', 'BIT', len(code_register))
    pq += [MEASURE(qq, rr) for qq, rr in zip(code_register, ro)]
    
    # THIS CODE APPLIES THE NOISE FOR YOU
    kraus_ops = kraus_operators
    noise_data = Program()
    for qq in range(3):
        noise_data.define_noisy_gate("I", [qq], kraus_ops)
    pq = noise_data + pq

    # Run the simulation trials times using the QVM and check how many times it did not work
    results = qvm.run(address_qubits(pq), trials = trials)
    score = 0
    for i in results:
        count = np.sum(i)
        #if count >= len(code_register)/2
        if count == len(code_register):
            score += 1
    return int(score)

    
