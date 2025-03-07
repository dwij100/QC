import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph
import numpy as np
import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

# Graph setup
n = 5
graph = rx.PyGraph()
graph.add_nodes_from(range(n))
edge_list = [(0, 1, 1.0), (0, 2, 1.0), (0, 4, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 4, 1.0)]
graph.add_edges_from(edge_list)

graph_fig, graph_ax = plt.subplots()
draw_graph(graph, node_size=600, with_labels=True, ax=graph_ax)

# Pauli list conversion
def build_max_cut_paulis(graph: rx.PyGraph) -> list[tuple[str, float]]:
    pauli_list = []
    for edge in list(graph.edge_list()):
        paulis = ["I"] * len(graph)
        paulis[edge[0]], paulis[edge[1]] = "Z", "Z"
        weight = graph.get_edge_data(edge[0], edge[1])
        pauli_list.append(("".join(paulis)[::-1], weight))
    return pauli_list

max_cut_paulis = build_max_cut_paulis(graph)
cost_hamiltonian = SparsePauliOp.from_list(max_cut_paulis)

# Create QAOA circuit
ansatz = QAOAAnsatz(cost_operator=cost_hamiltonian, reps=4)
ansatz.measure_all()

circuit_fig, circuit_ax = plt.subplots(figsize=(10, 4))
ansatz.draw("mpl", ax=circuit_ax)

# Transpile the circuit to the basis gates supported by the simulator
simulator = AerSimulator()
transpiled_ansatz = transpile(ansatz, simulator)
transpiled_ansatz.draw("mpl")
simulator.set_options(shots=10000)

# Cost function for optimizer
objective_func_vals = []
def cost_func_estimator(params, ansatz, hamiltonian, simulator):
    param_circuit = ansatz.assign_parameters(params)
    transpiled_param_circuit = transpile(param_circuit, simulator)
    job = simulator.run(transpiled_param_circuit)
    result = job.result()
    cost = result.get_counts().get('0' * len(ansatz.qubits), 0) / simulator.options.shots
    objective_func_vals.append(cost)
    
    return cost

# Optimizer setup
init_params = [np.pi, np.pi/2, np.pi, np.pi/2,np.pi, np.pi/2,np.pi, np.pi/2]
result = minimize(
    cost_func_estimator,
    init_params,
    args=(ansatz, cost_hamiltonian, simulator),
    method="COBYLA",
    tol=1e-3,
)

# Optimization progress plot
opt_fig, opt_ax = plt.subplots(figsize=(12, 6))
opt_ax.plot(objective_func_vals, color="tab:blue")
opt_ax.set_xlabel("Iteration")
opt_ax.set_ylabel("Cost")
opt_ax.set_title("Optimization Progress")

# Assign optimized parameters to circuit
optimized_circuit = ansatz.assign_parameters(result.x)

#final_circuit_fig, final_circuit_ax = plt.subplots(figsize=(10, 4))
#optimized_circuit.draw("mpl", ax=final_circuit_ax)

# Print optimized parameter values (angles)
optimized_params = result.x
for i, angle in enumerate(optimized_params):
    print(f"Angle {i + 1}: {angle:.4f} radians")


# Run final sampling
job = simulator.run(transpile(optimized_circuit, simulator))
result = job.result()
counts = result.get_counts()
shots = sum(counts.values())
final_distribution = {key: val / shots for key, val in counts.items()}

# Extract most likely bitstring
most_likely = max(final_distribution, key=final_distribution.get)
most_likely_bitstring = [int(bit) for bit in most_likely[::-1]]

# Final probability distribution plot (Top 4 in purple)
sorted_items = sorted(final_distribution.items(), key=lambda x: x[1], reverse=True)
top_4 = {key for key, _ in sorted_items[:4]}  # Select top 4 bitstrings

dist_fig, dist_ax = plt.subplots(figsize=(11, 6))
colors = ["tab:purple" if key in top_4 else "tab:grey" for key in final_distribution.keys()]
dist_ax.bar(final_distribution.keys(), final_distribution.values(), color=colors)
dist_ax.set_xticklabels(final_distribution.keys(), rotation=45)
dist_ax.set_xlabel("Bitstrings (reversed)")
dist_ax.set_ylabel("Probability")
dist_ax.set_title("Result Distribution")

# Graph coloring plot
def plot_result(G, x, ax):
    colors = ["tab:grey" if i == 0 else "tab:purple" for i in x]
    pos = rx.spring_layout(G)
    rx.visualization.mpl_draw(G, node_color=colors, node_size=100, alpha=0.8, pos=pos, ax=ax)

graph_result_fig, graph_result_ax = plt.subplots()
plot_result(graph, most_likely_bitstring, graph_result_ax)

# Evaluate max-cut value
def evaluate_sample(x, graph):
    return sum(x[u] * (1 - x[v]) + x[v] * (1 - x[u]) for u, v in list(graph.edge_list()))

cut_value = evaluate_sample(most_likely_bitstring, graph)
print('The value of the cut is:', cut_value)

# Show all plots at the end
plt.show()
