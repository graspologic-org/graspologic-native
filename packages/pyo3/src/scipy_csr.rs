// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Zero-copy CSR graph view over scipy.sparse.csr_matrix arrays.
//!
//! scipy stores CSR as:
//! - `indptr`: int64[] of length n+1 (row offsets)
//! - `indices`: int32[] (column indices for each non-zero)
//! - `data`: float64[] (values for each non-zero)
//!
//! We borrow these arrays and implement `NetworkView` directly on them.
//! The only computed data are `node_weights` (row sums) and totals.

use network_partitions::network::network_view::{Neighbor, NetworkView};

/// Zero-copy graph view over scipy CSR arrays.
///
/// Borrows `indptr`, `indices`, and `data` from numpy arrays.
/// Precomputes `node_weights` (O(nnz)) on construction — this is the only allocation.
pub struct ScipyCsrView<'a> {
    /// Row offsets as i64 (scipy int64 indptr). Length = n_nodes + 1.
    indptr: &'a [i64],
    /// Column indices as i32 (scipy int32 indices). Length = nnz.
    indices: &'a [i32],
    /// Edge weights as f64 (scipy float64 data). Length = nnz.
    data: &'a [f64],
    /// Number of nodes.
    n_nodes: usize,
    /// Precomputed node weights.
    node_weights: Vec<f64>,
    /// Total edge weight (sum of non-self-loop weights / 2 for undirected).
    total_edge_weight: f64,
    /// Total self-link edge weight.
    total_self_links_weight: f64,
    /// Number of undirected non-self-loop edges.
    num_edges: usize,
}

impl<'a> ScipyCsrView<'a> {
    /// Create a graph view from scipy CSR components.
    ///
    /// # Arguments
    /// - `use_modularity`: if true, node weights are set to the sum of incident
    ///   non-self-loop edge weights (modularity convention). If false, node weights
    ///   are all 1.0 (CPM convention).
    ///
    /// # Requirements
    /// - `indptr` length must be `n_nodes + 1`
    /// - `indptr[0]` must be 0
    /// - `indptr` values must be non-negative and non-decreasing
    /// - All values in `indices` must be in `[0, n_nodes)`
    /// - `indices` and `data` must have the same length (`indptr[n_nodes]` elements)
    /// - All weights must be finite and non-negative
    /// - The matrix should be symmetric (both directions stored) for undirected graphs
    pub fn new(
        indptr: &'a [i64],
        indices: &'a [i32],
        data: &'a [f64],
        n_nodes: usize,
        use_modularity: bool,
    ) -> Result<Self, String> {
        if indptr.len() != n_nodes + 1 {
            return Err(format!(
                "indptr length {} does not match n_nodes + 1 = {}",
                indptr.len(),
                n_nodes + 1
            ));
        }

        if indptr[0] != 0 {
            return Err(format!("indptr[0] must be 0, got {}", indptr[0]));
        }

        // Validate indptr is monotonically non-decreasing and non-negative
        for i in 0..indptr.len() - 1 {
            if indptr[i] < 0 || indptr[i] > indptr[i + 1] {
                return Err(format!(
                    "indptr is not monotonically non-decreasing at index {}: {} > {}",
                    i,
                    indptr[i],
                    indptr[i + 1]
                ));
            }
        }

        let nnz = indptr[n_nodes] as usize;
        if indices.len() != nnz || data.len() != nnz {
            return Err(format!(
                "indices length {} or data length {} does not match nnz = {}",
                indices.len(),
                data.len(),
                nnz
            ));
        }

        // Validate indices are non-negative and within [0, n_nodes)
        for (i, &idx) in indices.iter().enumerate() {
            if idx < 0 || (idx as usize) >= n_nodes {
                return Err(format!(
                    "indices[{i}] = {idx} is out of bounds (n_nodes = {n_nodes})"
                ));
            }
        }

        // Compute node weights and totals — O(nnz)
        // For modularity: node_weight = sum of incident non-self-loop edge weights
        // For CPM: node_weight = 1.0
        let mut node_weights = vec![0.0_f64; n_nodes];
        let mut total = 0.0_f64;
        let mut total_self_links = 0.0_f64;
        let mut num_diag_entries = 0usize;

        for node in 0..n_nodes {
            let start = indptr[node] as usize;
            let end = indptr[node + 1] as usize;
            let mut row_sum = 0.0_f64;
            for pos in start..end {
                let weight = data[pos];
                if !weight.is_finite() || weight < 0.0 {
                    return Err(format!(
                        "data[{pos}] = {weight} is not a valid weight (must be finite and non-negative)"
                    ));
                }
                if indices[pos] as usize == node {
                    total_self_links += weight;
                    num_diag_entries += 1;
                } else {
                    row_sum += weight;
                }
            }
            node_weights[node] = if use_modularity { row_sum } else { 1.0 };
            total += row_sum;
        }

        // Each undirected non-self edge is stored twice; halve for the true total.
        let total_edge_weight = total / 2.0;
        let nnz = indptr[n_nodes] as usize;
        let num_edges = (nnz - num_diag_entries) / 2;

        Ok(Self {
            indptr,
            indices,
            data,
            n_nodes,
            node_weights,
            total_edge_weight,
            total_self_links_weight: total_self_links,
            num_edges,
        })
    }

    #[inline]
    fn row_range(
        &self,
        node: usize,
    ) -> (usize, usize) {
        let start = self.indptr[node] as usize;
        let end = self.indptr[node + 1] as usize;
        (start, end)
    }
}

/// Iterator over neighbors of a node in a scipy CSR view.
/// Skips self-loop entries (where neighbor_id == source node).
pub struct ScipyCsrNeighborIterator<'a> {
    indices: &'a [i32],
    data: &'a [f64],
    node_weights: &'a [f64],
    pos: usize,
    end: usize,
    source_node: usize,
}

impl Iterator for ScipyCsrNeighborIterator<'_> {
    type Item = Neighbor;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.end {
            let id = self.indices[self.pos] as usize;
            let edge_weight = self.data[self.pos];
            self.pos += 1;
            if id == self.source_node {
                continue; // skip self-loops
            }
            let node_weight = self.node_weights[id];
            return Some(Neighbor {
                id,
                edge_weight,
                node_weight,
            });
        }
        None
    }
}

impl<'a> NetworkView for ScipyCsrView<'a> {
    type Neighbors<'b>
        = ScipyCsrNeighborIterator<'b>
    where
        Self: 'b;

    #[inline]
    fn num_nodes(&self) -> usize {
        self.n_nodes
    }

    #[inline]
    fn node_weight(
        &self,
        node_id: usize,
    ) -> f64 {
        self.node_weights[node_id]
    }

    #[inline]
    fn neighbors_for(
        &self,
        node_id: usize,
    ) -> Self::Neighbors<'_> {
        let (start, end) = self.row_range(node_id);
        ScipyCsrNeighborIterator {
            indices: self.indices,
            data: self.data,
            node_weights: &self.node_weights,
            pos: start,
            end,
            source_node: node_id,
        }
    }

    fn total_node_weight(&self) -> f64 {
        self.node_weights.iter().sum()
    }

    #[inline]
    fn total_edge_weight(&self) -> f64 {
        self.total_edge_weight
    }

    #[inline]
    fn total_self_links_edge_weight(&self) -> f64 {
        self.total_self_links_weight
    }

    fn num_edges(&self) -> usize {
        self.num_edges
    }

    fn node_weights(&self) -> Vec<f64> {
        self.node_weights.clone()
    }
}

// ScipyCsrView contains only immutable borrows (&[i64], &[i32], &[f64]) and an owned Vec<f64>,
// all of which are Send + Sync automatically. No manual unsafe impl needed.
