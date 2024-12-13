import random
import math
import numpy as np
from typing import Tuple
from collections import defaultdict
import matplotlib.pyplot as plt


class Graph:
    """A class representing an undirected graph with vertices and edges."""

    def __init__(self, n_vertices: int):
        """Initialize graph with given number of vertices.

        Args:
            n_vertices: Number of vertices in the graph
        """
        self.n = n_vertices
        self.edges = set()  # Store edges as tuples (u,v) where u < v
        self.adj_list = defaultdict(set)  # Adjacency list representation

    def add_edge(self, u: int, v: int):
        """Add undirected edge between vertices u and v if it doesn't exist.

        Args:
            u, v: Vertex indices to connect
        """
        if (u, v) not in self.edges and (v, u) not in self.edges:
            self.edges.add((min(u, v), max(u, v)))  # Ensure ordered pairs
            self.adj_list[u].add(v)
            self.adj_list[v].add(u)

    def remove_edge(self, u: int, v: int):
        """Remove edge between vertices u and v if it exists.

        Args:
            u, v: Vertex indices to disconnect
        """
        edge = (min(u, v), max(u, v))
        if edge in self.edges:
            self.edges.remove(edge)
            self.adj_list[u].remove(v)
            self.adj_list[v].remove(u)

    def get_edge_density(self) -> float:
        """Calculate the current edge density of the graph.

        Returns:
            Float between 0 and 1 representing edge density
        """
        max_edges = (self.n * (self.n - 1)) / 2
        return len(self.edges) / max_edges if max_edges > 0 else 0

    @classmethod
    def copy(cls, graph: "Graph") -> "Graph":
        """Create a deep copy of the given graph.

        Args:
            graph: Graph instance to copy

        Returns:
            New Graph instance with same structure
        """
        new_graph = cls(graph.n)
        new_graph.edges = set(graph.edges)
        new_graph.adj_list = defaultdict(
            set, {k: set(v) for k, v in graph.adj_list.items()}
        )
        return new_graph


class ObliviousAlgorithm:
    """Implementation of a privacy-preserving edge density estimation algorithm
    that is oblivious to adversarial updates."""

    def __init__(self, graph: Graph, epsilon: float):
        """Initialize algorithm with given graph and privacy parameter.

        Args:
            graph: Initial graph structure
            epsilon: Privacy parameter
        """
        self.graph = Graph.copy(graph)
        self.epsilon = epsilon
        self.random_seed = random.randint(0, 2**32 - 1)
        random.seed(self.random_seed)
        # Calculate sampling probability based on graph size
        self.sample_prob = min(1.0, 5 * math.log(self.graph.n) / self.graph.n)
        self.initialize()

    def initialize(self):
        """Initialize edge sampling based on sampling probability."""
        self.sampled_edges = set()
        for edge in self.graph.edges:
            if random.random() < self.sample_prob:
                self.sampled_edges.add(edge)

    def handle_update(self, edge: tuple, is_addition: bool) -> float:
        """Process an edge update and return new density estimate.

        Args:
            edge: Tuple of vertices defining the edge
            is_addition: True if adding edge, False if removing

        Returns:
            Estimated edge density
        """
        if is_addition:
            self.graph.add_edge(*edge)
            if random.random() < self.sample_prob:
                self.sampled_edges.add(edge)
        else:
            self.graph.remove_edge(*edge)
            if edge in self.sampled_edges:
                self.sampled_edges.remove(edge)
        return self.query()

    def query(self) -> float:
        """Calculate current edge density estimate based on sampled edges.

        Returns:
            Estimated edge density
        """
        if not self.sample_prob:
            return 0
        estimated_edges = len(self.sampled_edges) / self.sample_prob
        max_possible_edges = (self.graph.n * (self.graph.n - 1)) / 2
        return estimated_edges / max_possible_edges if max_possible_edges > 0 else 0


class AdaptiveAlgorithm:
    """Implementation of a privacy-preserving edge density estimation algorithm
    that is resistant to adaptive adversaries."""

    def __init__(self, graph: Graph, epsilon: float, T: int):
        """Initialize algorithm with multiple instances of ObliviousAlgorithm.

        Args:
            graph: Initial graph structure
            epsilon: Privacy parameter
            T: Number of updates to process before reset
        """
        self.graph = graph
        self.epsilon = epsilon
        self.T = T
        self.num_copies = int(2 * math.sqrt(T))
        # Create multiple instances of ObliviousAlgorithm
        self.algorithms = [
            ObliviousAlgorithm(graph, epsilon) for _ in range(self.num_copies)
        ]
        self.privacy_budget = epsilon / (2 * math.sqrt(math.log(T)))
        self.updates_processed = 0

    def add_noise(self, value: float) -> float:
        """Add Laplace noise to maintain differential privacy.

        Args:
            value: Original value

        Returns:
            Noisy value between 0 and 1
        """
        scale = 0.005 / (self.privacy_budget * math.sqrt(self.T))
        noise = np.random.laplace(0, scale)
        return max(0, min(1, value + noise))

    def pMedian(self, values: list[float], epsilon: float, beta: float) -> float:
        """Compute differentially private median using binary search.

        Args:
            values: List of values
            epsilon: Privacy parameter
            beta: Failure probability

        Returns:
            Private median estimate
        """
        values.sort()
        n = len(values)
        Gamma = (1 / epsilon) * math.log(len(values) / beta)

        def noisy_count(threshold):
            count = sum(1 for v in values if v <= threshold)
            noise = np.random.laplace(0, 1 / epsilon)
            return count + noise

        low, high = 0, n - 1
        while low < high:
            mid = (low + high) // 2
            mid_val = values[mid]

            if noisy_count(mid_val) < n / 2:
                low = mid + 1
            else:
                high = mid

        return values[low]

    def handle_update(self, edge: tuple, is_addition: bool) -> float:
        """Process update across all algorithm instances and return median estimate.

        Args:
            edge: Tuple of vertices defining the edge
            is_addition: True if adding edge, False if removing

        Returns:
            Private median of density estimates
        """
        values = []
        for algo in self.algorithms:
            value = algo.handle_update(edge, is_addition)
            noisy_value = self.add_noise(value)
            values.append(noisy_value)

        self.updates_processed += 1
        if self.updates_processed >= self.T:
            self.reset()

        return self.pMedian(values, self.epsilon, beta=0.01)

    def reset(self):
        """Reset all algorithm instances and update counter."""
        for algo in self.algorithms:
            algo.initialize()
        self.updates_processed = 0


class Adversary:
    """Simulates an adversary that generates graph updates."""

    def __init__(self, n_vertices: int, is_adaptive: bool):
        """Initialize adversary with graph size and strategy type.

        Args:
            n_vertices: Number of vertices in graph
            is_adaptive: Whether to use adaptive strategy
        """
        self.n = n_vertices
        self.is_adaptive = is_adaptive
        self.prev_response = None
        self.edge_history = set()
        self.current_edges = set()

    def get_update(
        self, algorithm_response: float = None
    ) -> Tuple[Tuple[int, int], bool]:
        """Generate next edge update based on strategy.

        Args:
            algorithm_response: Previous algorithm estimate (for adaptive strategy)

        Returns:
            Tuple of ((u,v), is_addition) representing edge update
        """
        if algorithm_response is not None:
            self.prev_response = algorithm_response

        if self.is_adaptive and self.prev_response is not None:
            # Adaptive strategy: try to maximize difference from estimate
            if self.prev_response > self.true_density() and self.current_edges:
                edge = random.choice(list(self.current_edges))
                self.current_edges.remove(edge)
                return (edge, False)

        # Add new random edge that hasn't been used
        possible_edges = (
            set((i, j) for i in range(self.n) for j in range(i + 1, self.n))
            - self.edge_history
        )
        if not possible_edges:
            if self.current_edges:
                edge = random.choice(list(self.current_edges))
                self.current_edges.remove(edge)
                return (edge, False)
            return ((0, 1), True)

        edge = random.choice(list(possible_edges))
        self.edge_history.add(edge)
        self.current_edges.add(edge)
        return (edge, True)

    def true_density(self) -> float:
        """Calculate actual edge density of current graph.

        Returns:
            Current edge density
        """
        max_edges = (self.n * (self.n - 1)) / 2
        return len(self.current_edges) / max_edges if max_edges > 0 else 0


def compare_algorithms(n_vertices: int, n_updates: int):
    """Compare performance of oblivious and adaptive algorithms against both types of adversaries.

    Args:
        n_vertices: Number of vertices in graph
        n_updates: Number of updates to process
    """
    graph = Graph(n_vertices)
    epsilon = 0.1

    # Test against oblivious adversary
    print("Testing against oblivious adversary...")
    oblivious_algo = ObliviousAlgorithm(graph, epsilon)
    adaptive_algo = AdaptiveAlgorithm(graph, epsilon, n_updates)
    adversary = Adversary(n_vertices, is_adaptive=False)

    oblivious_results = []
    adaptive_results = []
    true_density = []

    for _ in range(n_updates):
        edge, is_addition = adversary.get_update()
        oblivious_value = oblivious_algo.handle_update(edge, is_addition)
        adaptive_value = adaptive_algo.handle_update(edge, is_addition)
        true_density.append(adversary.true_density())
        oblivious_results.append(oblivious_value)
        adaptive_results.append(adaptive_value)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(oblivious_results, label="Oblivious Algorithm", alpha=0.7)
    plt.plot(adaptive_results, label="Adaptive Algorithm", alpha=0.7)
    plt.plot(true_density, "k--", label="True Density", alpha=0.5)
    plt.title("Performance Against Oblivious Adversary")
    plt.xlabel("Update Number")
    plt.ylabel("Edge Density Estimation")
    plt.legend()
    plt.grid(True)

    # Test against adaptive adversary
    print("Testing against adaptive adversary...")
    graph = Graph(n_vertices)
    oblivious_algo = ObliviousAlgorithm(graph, epsilon)
    adaptive_algo = AdaptiveAlgorithm(graph, epsilon, n_updates)
    adversary = Adversary(n_vertices, is_adaptive=True)

    oblivious_results = []
    adaptive_results = []
    true_density = []

    for _ in range(n_updates):
        edge, is_addition = adversary.get_update(
            adaptive_results[-1] if adaptive_results else None
        )
        oblivious_value = oblivious_algo.handle_update(edge, is_addition)
        adaptive_value = adaptive_algo.handle_update(edge, is_addition)
        true_density.append(adversary.true_density())
        oblivious_results.append(oblivious_value)
        adaptive_results.append(adaptive_value)

    plt.subplot(1, 2, 2)
    plt.plot(oblivious_results, label="Oblivious Algorithm", alpha=0.7)
    plt.plot(adaptive_results, label="Adaptive Algorithm", alpha=0.7)
    plt.plot(true_density, "k--", label="True Density", alpha=0.5)
    plt.title("Performance Against Adaptive Adversary")
    plt.xlabel("Update Number")
    plt.ylabel("Edge Density Estimation")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    compare_algorithms(n_vertices=30, n_updates=1000)
