import numpy as np
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Estimator
from qiskit.quantum_info import PauliList, SparsePauliOp
from qiskit_algorithms.minimum_eigensolvers import QAOA

# Define a 3-node graph for Max-Cut
graph = nx.Graph()
graph.add_edges_from([(0, 1), (1, 2), (2, 0)])

# Create cost Hamiltonian
pauli_list = []
coeffs = []
for edge in graph.edges():
    z_term = ['I'] * len(graph.nodes)
    z_term[edge[0]] = 'Z'
    z_term[edge[1]] = 'Z'
    pauli_list.append("".join(z_term))
    coeffs.append(-1)

hamiltonian = SparsePauliOp(PauliList(pauli_list), coeffs)

# Create QAOA instance
estimator = Estimator()
p = 1  # Number of QAOA layers
qaoa = QAOA(estimator, optimizer=COBYLA(), reps=p)

# Solve Max-Cut
result = qaoa.compute_minimum_eigenvalue(hamiltonian)
print("Optimal parameters:", result.optimal_parameters)
print("Optimal value:", result.optimal_value)
print("Optimal eigenstate:", result.eigenstate)
