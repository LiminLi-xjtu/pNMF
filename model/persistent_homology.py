"""
This module includes components inspired by the implementation of

Hozumi, Y., and Wei, G.-W.
Analyzing single cell RNA sequencing with topological nonnegative matrix factorization,
Journal of Computational and Applied Mathematics, 2024.

Original repository:
https://github.com/hozumiyu/TopologicalNMF-scRNAseq
"""


import numpy as np
from typing import Generator


class UnionFind:
    """
    Union-Find (Disjoint Set Union, DSU) data structure with path compression.
    Used to efficiently track connected components in a graph.
    """

    def __init__(self, n_vertices: int):
        """
        Initialize Union-Find with each vertex in its own component.

        Parameters
        ----------
        n_vertices : int
            Number of vertices in the graph.
        """
        self._parent = np.arange(n_vertices, dtype=int)

    def find(self, u: int) -> int:
        """
        Find the root of vertex `u` with path compression.

        Parameters
        ----------
        u : int
            Vertex index.

        Returns
        -------
        int
            Root of the vertex.
        """
        root = u
        while self._parent[root] != root:
            root = int(self._parent[root])
        while self._parent[u] != u:
            parent = int(self._parent[u])
            self._parent[u] = root
            u = parent
        return root

    def merge(self, u: int, v: int) -> None:
        """
        Merge the components containing vertices `u` and `v`.

        Parameters
        ----------
        u : int
            Vertex index.
        v : int
            Vertex index.
        """
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            self._parent[root_u] = root_v

    def roots(self) -> Generator[int, None, None]:
        """
        Yield the roots of all components.

        Yields
        ------
        int
            Root of a component.
        """
        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class Persistent_Homology_Calculation:
    """
    Compute 0-dimensional persistent homology pairs from a distance or
    edge-weight matrix. Each returned pair contains the endpoints of a
    Kruskal-accepted tree edge. Its weight is a finite H0 death value.
    """

    def __call__(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute persistence pairs from the upper-triangular entries of the matrix.

        Parameters
        ----------
        matrix : np.ndarray, shape (N, N)
            Symmetric matrix representing pairwise distances or edge weights.

        Returns
        -------
        np.ndarray, shape (num_pairs, 2)
            Vertex indices of the Kruskal-accepted edges that merge components.
        """
        n_vertices = matrix.shape[0]
        uf = UnionFind(n_vertices)

        # Get upper-triangular indices to avoid duplicates
        triu_indices = np.triu_indices(n_vertices, k=1)
        edge_weights = matrix[triu_indices]

        # Sort edges by weight in non-decreasing order
        sorted_edge_indices = np.argsort(edge_weights, kind='stable')

        persistence_pairs = []

        # Process edges in order of increasing weight
        for idx in sorted_edge_indices:
            u = triu_indices[0][idx]
            v = triu_indices[1][idx]

            root_u = uf.find(u)
            root_v = uf.find(v)

            # Skip if already in the same component
            if root_u == root_v:
                continue

            # Merge components (always merge younger component into older)
            if root_u > root_v:
                uf.merge(v, u)
            else:
                uf.merge(u, v)

            # Store the persistence pair (sorted)
            persistence_pairs.append((min(u, v), max(u, v)))

            # A connected graph on n vertices has exactly n - 1 successful
            # merges. Remaining edges cannot add persistence pairs.
            if len(persistence_pairs) == n_vertices - 1:
                break

        return np.array(persistence_pairs, dtype=int)
