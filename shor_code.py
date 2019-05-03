from typing import List
import numpy as np

from pyquil import Program
from pyquil.gates import MEASURE, H, I, CNOT
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
    pass


def depolarizing_channel(prob: float):
    pass


def bit_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    
    ### Do your encoding step here
    q0 = QubitPlaceholder()
    q1 = QubitPlaceholder()
    a0 = QubitPlaceholder()
    a1 = QubitPlaceholder()
    code_register = [qubit, q0, q1, a0, a1]  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = Program(CNOT(code_register[0], code_register[1]), CNOT(code_register[0], code_register[2]))
    # the Program that does the encoding
    
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)


    ### Do your decoding and correction steps here
    ro = pq.declare('ro1', 'BIT', 2)

    pq += Program(CNOT(code_register[0], code_register[3]), CNOT(code_register[1], code_register[3]))
    pq = pq + MEASURE(code_register[3], ro[0])

    pq += Program(CNOT(code_register[1], code_register[4]), CNOT(code_register[2], code_register[4]))
    pq = pq + MEASURE(code_register[4], ro[1])
    
    if ro[0] == 1 and ro[1] == 1:
        pq += X(code_register[1])

    elif ro[0] == 1 and ro[1] == 0:
        pq += X(code_register[0])
    
    elif ro[0] == 0 and ro[1] == 1:
        pq += X(code_register[2])

    return pq, code_register



def phase_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    ### Do your encoding step here
    q0 = QubitPlaceholder()
    q1 = QubitPlaceholder()
    a0 = QubitPlaceholder()
    a1 = QubitPlaceholder()
    code_register = [qubit, q0, q1, a0, a1]  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = Program(CNOT(code_register[0], code_register[1]), CNOT(code_register[0], code_register[2]))
    pq += (H(q) for q in code_register[:3])
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)

    ### Do your decoding and correction steps here
    pq += (H(q) for q in code_register[:3])
    ro1 = pq.declare('ro', 'BIT', 2)
    pq += Program(CNOT(code_register[0], code_register[3]), CNOT(code_register[1], code_register[3]))
    pq = pq + MEASURE(code_register[3], ro1[0])

    pq += Program(CNOT(code_register[1], code_register[4]), CNOT(code_register[2], code_register[4]))
    pq = pq + MEASURE(code_register[4], ro1[1])

    if ro1[0] == 1 and ro1[1] == 1:
        pq += X(code_register[1])
    
    elif ro1[0] == 1 and ro1[1] == 0:
        pq += X(code_register[0])
    
    elif ro1[0] == 0 and ro1[1] == 1:
        pq += X(code_register[2])

    return pq, code_register


def shor(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    
    q0 = QubitPlaceholder()
    q1 = QubitPlaceholder()
    q00 = QubitPlaceholder()
    q01 = QubitPlaceholder()
    q10 = QubitPlaceholder()
    q11 = QubitPlaceholder()
    q20 = QubitPlaceholder()
    q21 = QubitPlaceholder()
    a00 = QubitPlaceholder()
    a01 = QubitPlaceholder()
    a10 = QubitPlaceholder()
    a11 = QubitPlaceholder()
    a20 = QubitPlaceholder()
    a21 = QubitPlaceholder()
    b0 = QubitPlaceholder()
    b1 = QubitPlaceholder()
    
    code_register = [qubit, q0, q1, q00, q01, q10, q11, q20, q21, a00, a01, a10, a11, a20, a21, b0, b1]  # the List[QubitPlaceholder] of the qubits you have encoded into
    #encode qubit q0 q1
    pq = Program(CNOT(code_register[0], code_register[1]), CNOT(code_register[0], code_register[2]))
    
    pq += (H(q) for q in code_register[:3])
    
    #encode qubit q00 q01
    pq = Program(CNOT(code_register[0], code_register[3]), CNOT(code_register[0], code_register[4]))
    
    #encode q0 q10 q11
    pq = Program(CNOT(code_register[1], code_register[5]), CNOT(code_register[1], code_register[6]))
    
    #encode q1 q20 q21
    pq = Program(CNOT(code_register[2], code_register[7]), CNOT(code_register[2], code_register[8]))
    
    ro = pq.declare('ro', 'BIT', 8)
    
    
    '''
    FIND BIT ERRORS
    '''
    #parity [qubit, q00] a00
    pq += Program(CNOT(code_register[0], code_register[9]), CNOT(code_register[3], code_register[9]))
    pq = pq + MEASURE(code_register[9], ro[0])
    
    #parity [q00, q01] a01
    pq += Program(CNOT(code_register[3], code_register[10]), CNOT(code_register[4], code_register[10]))
    pq = pq + MEASURE(code_register[10], ro[1])
    
    if ro[0] == 1 and ro[1] == 1:
        pq += X(code_register[3])
    
    elif ro[0] == 1 and ro[1] == 0:
        pq += X(code_register[0])
    
    elif ro[0] == 0 and ro[1] == 1:
        pq += X(code_register[4])

    #parity [q0, q10] a10
    pq += Program(CNOT(code_register[1], code_register[11]), CNOT(code_register[5], code_register[11]))
    pq = pq + MEASURE(code_register[11], ro[2])

    #parity [q10, q11] a11
    pq += Program(CNOT(code_register[5], code_register[12]), CNOT(code_register[6], code_register[12]))
    pq = pq + MEASURE(code_register[12], ro[3])

    if ro[2] == 1 and ro[3] == 1:
        pq += X(code_register[5])
    
    elif ro[2] == 1 and ro[3] == 0:
        pq += X(code_register[1])
    
    elif ro[2] == 0 and ro[3] == 1:
        pq += X(code_register[6])

    #parity [q1, q21] a20
    pq += Program(CNOT(code_register[2], code_register[13]), CNOT(code_register[7], code_register[13]))
    pq = pq + MEASURE(code_register[13], ro[4])

    #parity [q21, q22] a21
    pq += Program(CNOT(code_register[7], code_register[14]), CNOT(code_register[8], code_register[14]))
    pq = pq + MEASURE(code_register[14], ro[5])
    
    if ro[4] == 1 and ro[5] == 1:
        pq += X(code_register[7])
    
    elif ro[4] == 1 and ro[5] == 0:
        pq += X(code_register[2])

    elif ro[4] == 0 and ro[5] == 1:
        pq += X(code_register[8])
    
    '''
    FIND BIT ERRORS
    '''
    pq += (H(q) for q in code_register[:9])
    
    #parity [qubit, q00, q01, q0, q10, q11] b0
    pq += Program(CNOT(code_register[0], code_register[15]), CNOT(code_register[3], code_register[15]), CNOT(code_register[4], code_register[15]), CNOT(code_register[1], code_register[15]), CNOT(code_register[5], code_register[15]), CNOT(code_register[6], code_register[15]))
    pq = pq + MEASURE(code_register[15], ro[6])
    
    #parity [q0, q10, q11, q1, q20, q21] b1
    pq += Program(CNOT(code_register[1], code_register[16]), CNOT(code_register[5], code_register[16]), CNOT(code_register[6], code_register[16]), CNOT(code_register[2], code_register[16]), CNOT(code_register[7], code_register[16]), CNOT(code_register[8], code_register[16]))
    pq = pq + MEASURE(code_register[16], ro[7])
    
    if ro[6] == 1 and ro[7] == 1:
        pq += X(code_register[1])
    
    elif ro[6] == 1 and ro[7] == 0:
        pq += X(code_register[0])
    
    elif ro[6] == 0 and ro[7] == 1:
        pq += X(code_register[2])
    
    return pq, code_register


def run_code(error_code, noise, trials=10):
    """ Takes in an error_code function (e.g. bit_code, phase_code or shor) and runs this code on the QVM"""
    pq, code_register = error_code(QubitPlaceholder(), noise=noise)
    ro = pq.declare('ro2', 'BIT', len(code_register))
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
