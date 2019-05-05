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
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_Z = np.sqrt(prob) * np.asarray([[1, 0], [0, -1]])
    return [noisy_I, noisy_Z]


def depolarizing_channel(prob: float):
    noisy_I = np.sqrt(1-prob) * np.asarray([[1, 0], [0, 1]])
    noisy_X = np.sqrt(prob/3) * np.asarray([[0, 1], [1, 0]])
    noisy_Z = np.sqrt(prob/3) * np.asarray([[1, 0], [0, -1]])
    noisy_Y = np.sqrt(prob/3) * np.asarray([[0, -1], [1, 0]])
    return [noisy_I, noisy_X, noisy_Y, noisy_Z]


def bit_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    
    ### Do your encoding step here
    q0 = QubitPlaceholder()
    q1 = QubitPlaceholder()
    a0 = QubitPlaceholder()
    a1 = QubitPlaceholder()
    code_register = [qubit, q0, q1]  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = Program(CNOT(qubit, q0), CNOT(qubit, q1))
    # the Program that does the encoding
    
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)


    ### Do your decoding and correction steps here
    flip = pq.declare('flip', 'BIT', 2)

    pq += Program(CNOT(qubit, a0), CNOT(q0, a0))
    pq = pq + MEASURE(a0, flip[0])

    pq += Program(CNOT(q0, a1), CNOT(q1, a1))
    pq = pq + MEASURE(a1, flip[1])

    if flip[0] == 1 and flip[1] == 1:
        pq += X(q0)

    elif flip[0] == 1 and flip[1] == 0:
        pq += X(qubit)
    
    elif flip[0] == 0 and flip[1] == 1:
        pq += X(q1)
    code_register = [qubit, q0, q1]  # the List[QubitPlaceholder] of the qubits you have encoded into
    return pq, code_register
    '''
    pq = Program(CNOT(code_register[0], code_register[1]), CNOT(qubit, code_register[2]))
    # the Program that does the encoding
    
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)


    ### Do your decoding and correction steps here
    flip = pq.declare('flip', 'BIT', 2)

    pq += Program(CNOT(code_register[0], a0), CNOT(code_register[1], a0))
    pq = pq + MEASURE(a0, flip[0])

    pq += Program(CNOT(code_register[1], a1), CNOT(code_register[2], a1))
    pq = pq + MEASURE(a1, flip[1])
    
    if flip[0] == 1 and flip[1] == 1:
        pq += X(code_register[1])

    elif flip[0] == 1 and flip[1] == 0:
        pq += X(code_register[0])
    
    elif flip[0] == 0 and flip[1] == 1:
        pq += X(code_register[2])

    return pq, code_register
    '''



def phase_code(qubit: QubitPlaceholder, noise=None) -> (Program, List[QubitPlaceholder]):
    ### Do your encoding step here
    q0 = QubitPlaceholder()
    q1 = QubitPlaceholder()
    a0 = QubitPlaceholder()
    a1 = QubitPlaceholder()
    code_register = [qubit, q0, q1]  # the List[QubitPlaceholder] of the qubits you have encoded into
    pq = Program(CNOT(code_register[0], code_register[1]), CNOT(code_register[0], code_register[2]))
    pq += (H(q) for q in code_register[:3])
    # DON'T CHANGE THIS CODE BLOCK. It applies the errors for simulations
    if noise is None:
        pq += [I(qq) for qq in code_register]
    else:
        pq += noise(code_register)

    ### Do your decoding and correction steps here
    pq += (H(q) for q in code_register[:3])
    phase = pq.declare('phase', 'BIT', 2)
    pq += Program(CNOT(code_register[0], a0), CNOT(code_register[1], a0))
    pq = pq + MEASURE(a0, phase[0])

    pq += Program(CNOT(code_register[1], a1), CNOT(code_register[2], a1))
    pq = pq + MEASURE(a1, phase[1])

    if phase[0] == 1 and phase[1] == 1:
        pq += X(code_register[1])
    
    elif phase[0] == 1 and phase[1] == 0:
        pq += X(code_register[0])
    
    elif phase[0] == 0 and phase[1] == 1:
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
    
    code_register = [qubit, q0, q1, q00, q01, q10, q11, q20, q21]  # the List[QubitPlaceholder] of the qubits you have encoded into
    #encode qubit q0 q1
    pq = Program(CNOT(code_register[0], code_register[1]), CNOT(code_register[0], code_register[2]))
    
    pq += (H(q) for q in code_register[:3])
    
    #encode qubit q00 q01
    pq = Program(CNOT(code_register[0], code_register[3]), CNOT(code_register[0], code_register[4]))
    
    #encode q0 q10 q11
    pq = Program(CNOT(code_register[1], code_register[5]), CNOT(code_register[1], code_register[6]))
    
    #encode q1 q20 q21
    pq = Program(CNOT(code_register[2], code_register[7]), CNOT(code_register[2], code_register[8]))
    
    roS = pq.declare('roS', 'BIT', 8)
    
    
    '''
    FIND BIT ERRORS
    '''
    #parity [qubit, q00] a00
    pq += Program(CNOT(code_register[0], a00), CNOT(code_register[3], a00))
    pq = pq + MEASURE(a00, roS[0])
    
    #parity [q00, q01] a01
    pq += Program(CNOT(code_register[3], a01), CNOT(code_register[4], a10))
    pq = pq + MEASURE(a01, roS[1])
    
    if roS[0] == 1 and roS[1] == 1:
        pq += X(code_register[3])
    
    elif roS[0] == 1 and roS[1] == 0:
        pq += X(code_register[0])
    
    elif roS[0] == 0 and roS[1] == 1:
        pq += X(code_register[4])

    #parity [q0, q10] a10
    pq += Program(CNOT(code_register[1], a10), CNOT(code_register[5], a10))
    pq = pq + MEASURE(a10, roS[2])

    #parity [q10, q11] a11
    pq += Program(CNOT(code_register[5], a11), CNOT(code_register[6], a11))
    pq = pq + MEASURE(a11, roS[3])

    if roS[2] == 1 and roS[3] == 1:
        pq += X(code_register[5])
    
    elif roS[2] == 1 and roS[3] == 0:
        pq += X(code_register[1])
    
    elif roS[2] == 0 and roS[3] == 1:
        pq += X(code_register[6])

    #parity [q1, q21] a20
    pq += Program(CNOT(code_register[2], a20), CNOT(code_register[7], a20))
    pq = pq + MEASURE(a20, roS[4])

    #parity [q21, q22] a21
    pq += Program(CNOT(code_register[7], a21), CNOT(code_register[8], a21)
    pq = pq + MEASURE(a21, roS[5])
    
    if roS[4] == 1 and roS[5] == 1:
        pq += X(code_register[7])
    
    elif roS[4] == 1 and roS[5] == 0:
        pq += X(code_register[2])

    elif roS[4] == 0 and roS[5] == 1:
        pq += X(code_register[8])
    
    '''
    FIND BIT ERRORS
    '''
    pq += (H(q) for q in code_register[:9])
    
    #parity [qubit, q00, q01, q0, q10, q11] b0
    pq += Program(CNOT(code_register[0], b0), CNOT(code_register[3], b0), CNOT(code_register[4], b0), CNOT(code_register[1], b0), CNOT(code_register[5], b0), CNOT(code_register[6], b0))
    pq = pq + MEASURE(b0, roS[6])
    
    #parity [q0, q10, q11, q1, q20, q21] b1
    pq += Program(CNOT(code_register[1], b1), CNOT(code_register[5], b1), CNOT(code_register[6], b1), CNOT(code_register[2], b1), CNOT(code_register[7], b1), CNOT(code_register[8], b1))
    pq = pq + MEASURE(b1, roS[7])
    
    if roS[6] == 1 and roS[7] == 1:
        pq += X(code_register[1])
    
    elif roS[6] == 1 and roS[7] == 0:
        pq += X(code_register[0])
    
    elif roS[6] == 0 and roS[7] == 1:
        pq += X(code_register[2])
    
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
