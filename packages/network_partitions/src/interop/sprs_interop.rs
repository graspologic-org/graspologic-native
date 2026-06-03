// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! `NetworkView` implementation for `sprs::CsMat<f64>` (CSR sparse matrix).
//!
//! This allows running the Leiden algorithm directly on a sparse adjacency matrix
//! from the `sprs` crate without copying into a `CompactNetwork`.
//!
//! # Matrix conventions
//!
//! - The matrix must be square (N×N) representing an undirected graph's adjacency.
//! - It must be symmetric: `A[i][j] == A[j][i]`.
//! - Non-zero entries represent edge weights (must be finite and non-negative).
//! - Diagonal entries represent self-loop weights.
//! - The matrix must be in CSR (Compressed Sparse Row) format.
//!
//! # Node weights
//!
//! Node weights default to 1.0, which is appropriate for CPM mode. For modularity
//! mode, callers must supply node weights equal to the weighted degree (sum of
//! incident non-self-loop edge weights) via [`SprsNetworkView::with_node_weights`].

use sprs::CsMatI;

use crate::network::network_view::{Neighbor, NetworkView};

/// A zero-copy view over a `sprs::CsMatI<f64, usize>` (CSR) implementing `NetworkView`.
///
/// Node weights default to 1.0 unless explicitly provided.
pub struct SprsNetworkView<'a> {
    matrix: &'a CsMatI<f64, usize>,
    node_weights: Option<&'a [f64]>,
    total_node_weight: f64,
    total_edge_weight: f64,
    total_self_links_edge_weight: f64,
}

/// Error type for `SprsNetworkView` construction.
#[derive(Debug, Clone)]
pub enum SprsValidationError {
    /// The matrix is not square.
    NotSquare { rows: usize, cols: usize },
    /// The matrix is not in CSR storage format.
    NotCsr,
    /// The node weights vector has the wrong length.
    NodeWeightsLengthMismatch { expected: usize, actual: usize },
    /// An edge weight is not finite or is negative.
    InvalidWeight { row: usize, col: usize, value: f64 },
}

impl std::fmt::Display for SprsValidationError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::NotSquare { rows, cols } => {
                write!(f, "Matrix is not square: {rows}x{cols}")
            }
            Self::NotCsr => write!(f, "Matrix must be in CSR (row-major) storage format"),
            Self::NodeWeightsLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Node weights length mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InvalidWeight { row, col, value } => {
                write!(
                    f,
                    "Edge weight at ({row}, {col}) is invalid: {value} (must be finite and non-negative)"
                )
            }
        }
    }
}

impl std::error::Error for SprsValidationError {}

impl<'a> SprsNetworkView<'a> {
    /// Create a new view over the given CSR matrix with uniform node weights of 1.0.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix is not square
    /// - The matrix is not in CSR format
    pub fn new(matrix: &'a CsMatI<f64, usize>) -> Result<Self, SprsValidationError> {
        Self::with_node_weights(matrix, None)
    }

    /// Create a new view with explicit node weights.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The matrix is not square
    /// - The matrix is not in CSR format
    /// - The node weights length doesn't match the matrix dimensions
    pub fn with_node_weights(
        matrix: &'a CsMatI<f64, usize>,
        node_weights: Option<&'a [f64]>,
    ) -> Result<Self, SprsValidationError> {
        let (rows, cols) = matrix.shape();
        if rows != cols {
            return Err(SprsValidationError::NotSquare { rows, cols });
        }
        if !matrix.is_csr() {
            return Err(SprsValidationError::NotCsr);
        }
        if let Some(nw) = node_weights
            && nw.len() != rows
        {
            return Err(SprsValidationError::NodeWeightsLengthMismatch {
                expected: rows,
                actual: nw.len(),
            });
        }

        let n = rows;
        let total_node_weight = match node_weights {
            Some(nw) => nw.iter().sum(),
            None => n as f64,
        };

        let mut total_edge_weight = 0.0;
        let mut total_self_links_edge_weight = 0.0;

        // Iterate over upper triangle + diagonal to count each edge once
        for (row_idx, row_vec) in matrix.outer_iterator().enumerate() {
            for (col_idx, &val) in row_vec.iter() {
                if !val.is_finite() || val < 0.0 {
                    return Err(SprsValidationError::InvalidWeight {
                        row: row_idx,
                        col: col_idx,
                        value: val,
                    });
                }
                if col_idx == row_idx {
                    total_self_links_edge_weight += val;
                } else if col_idx > row_idx {
                    total_edge_weight += val;
                }
            }
        }

        Ok(Self {
            matrix,
            node_weights,
            total_node_weight,
            total_edge_weight,
            total_self_links_edge_weight,
        })
    }
}

/// Iterator over neighbors of a node in an sprs CSR matrix.
/// Skips self-loop entries (where neighbor_id == source node).
pub struct SprsNeighborIterator<'a> {
    indices: &'a [usize],
    data: &'a [f64],
    pos: usize,
    end: usize,
    node_weights: Option<&'a [f64]>,
    source_node: usize,
}

impl<'a> Iterator for SprsNeighborIterator<'a> {
    type Item = Neighbor;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.end {
            let col_idx = self.indices[self.pos];
            let edge_weight = self.data[self.pos];
            self.pos += 1;
            if col_idx == self.source_node {
                continue; // skip self-loops
            }
            let node_weight = match self.node_weights {
                Some(nw) => nw[col_idx],
                None => 1.0,
            };
            return Some(Neighbor {
                id: col_idx,
                edge_weight,
                node_weight,
            });
        }
        None
    }
}

impl<'a> NetworkView for SprsNetworkView<'a> {
    type Neighbors<'b>
        = SprsNeighborIterator<'b>
    where
        Self: 'b;

    fn num_nodes(&self) -> usize {
        self.matrix.rows()
    }

    fn node_weight(
        &self,
        node_id: usize,
    ) -> f64 {
        match self.node_weights {
            Some(nw) => nw[node_id],
            None => 1.0,
        }
    }

    fn neighbors_for(
        &self,
        node_id: usize,
    ) -> Self::Neighbors<'_> {
        let indptr = self.matrix.indptr();
        let start = indptr.outer_inds_sz(node_id).start;
        let end = indptr.outer_inds_sz(node_id).end;
        SprsNeighborIterator {
            indices: self.matrix.indices(),
            data: self.matrix.data(),
            pos: start,
            end,
            node_weights: self.node_weights,
            source_node: node_id,
        }
    }

    fn total_node_weight(&self) -> f64 {
        self.total_node_weight
    }

    fn total_edge_weight(&self) -> f64 {
        self.total_edge_weight
    }

    fn total_self_links_edge_weight(&self) -> f64 {
        self.total_self_links_edge_weight
    }

    fn num_edges(&self) -> usize {
        let nnz = self.matrix.nnz();
        let n = self.matrix.rows();
        let indices = self.matrix.indices();
        let indptr = self.matrix.indptr();
        // Count ALL diagonal entries (handles duplicate (i,i) non-zeros)
        let diag_nnz: usize = (0..n)
            .map(|i| {
                let range = indptr.outer_inds_sz(i);
                indices[range].iter().filter(|&&col| col == i).count()
            })
            .sum();
        // Exclude self-loops: only count non-diagonal entries, halved for undirected
        (nnz - diag_nnz) / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leiden::leiden_view;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use sprs::TriMatI;

    /// Build a symmetric CSR matrix from a list of (row, col, weight) edges.
    /// Automatically symmetrizes (adds both directions).
    fn symmetric_csr(
        n: usize,
        edges: &[(usize, usize, f64)],
    ) -> CsMatI<f64, usize> {
        let mut tri = TriMatI::new((n, n));
        for &(r, c, w) in edges {
            tri.add_triplet(r, c, w);
            if r != c {
                tri.add_triplet(c, r, w);
            }
        }
        tri.to_csr()
    }

    #[test]
    fn test_sprs_view_basic_properties() {
        let mat = symmetric_csr(3, &[(0, 1, 0.5), (1, 2, 1.5), (0, 2, 2.0)]);
        let view = SprsNetworkView::new(&mat).unwrap();

        assert_eq!(view.num_nodes(), 3);
        assert_eq!(view.total_node_weight(), 3.0); // uniform weights
        assert_eq!(view.total_edge_weight(), 4.0); // 0.5 + 1.5 + 2.0
        assert_eq!(view.total_self_links_edge_weight(), 0.0);
    }

    #[test]
    fn test_sprs_view_custom_node_weights() {
        let mat = symmetric_csr(3, &[(0, 1, 1.0), (1, 2, 1.0)]);
        let nw = [2.0, 3.0, 4.0];
        let view = SprsNetworkView::with_node_weights(&mat, Some(&nw)).unwrap();

        assert_eq!(view.node_weight(0), 2.0);
        assert_eq!(view.node_weight(1), 3.0);
        assert_eq!(view.node_weight(2), 4.0);
        assert_eq!(view.total_node_weight(), 9.0);
    }

    #[test]
    fn test_sprs_view_neighbors() {
        let mat = symmetric_csr(3, &[(0, 1, 2.0), (0, 2, 3.0)]);
        let view = SprsNetworkView::new(&mat).unwrap();

        let neighbors: Vec<Neighbor> = view.neighbors_for(0).collect();
        assert_eq!(neighbors.len(), 2);
        let ids: Vec<usize> = neighbors.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_sprs_view_self_loops() {
        let mat = symmetric_csr(3, &[(0, 1, 1.0), (0, 0, 0.5)]);
        let view = SprsNetworkView::new(&mat).unwrap();

        assert_eq!(view.total_self_links_edge_weight(), 0.5);
        assert_eq!(view.total_edge_weight(), 1.0);
    }

    #[test]
    fn test_sprs_validation_not_square() {
        let tri = TriMatI::<f64, usize>::new((3, 4));
        let mat = tri.to_csr();
        let result = SprsNetworkView::new(&mat);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(err, SprsValidationError::NotSquare { .. }));
    }

    #[test]
    fn test_sprs_validation_node_weights_mismatch() {
        let mat = symmetric_csr(3, &[(0, 1, 1.0)]);
        let nw = [1.0, 2.0]; // wrong length
        let result = SprsNetworkView::with_node_weights(&mat, Some(&nw));
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(matches!(
            err,
            SprsValidationError::NodeWeightsLengthMismatch { .. }
        ));
    }

    #[test]
    fn test_sprs_view_to_compact_network() {
        let mat = symmetric_csr(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
        let view = SprsNetworkView::new(&mat).unwrap();
        let compact = view.to_compact_network();

        assert_eq!(compact.num_nodes(), 3);
    }

    #[test]
    fn test_sprs_leiden_triangle() {
        let mat = symmetric_csr(3, &[(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0)]);
        let view = SprsNetworkView::new(&mat).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);

        let (_improved, clustering) =
            leiden_view(&view, None, Some(2), Some(0.5), None, &mut rng, false, None).unwrap();

        // improved flag depends on resolution and graph structure
        let communities: std::collections::HashSet<usize> = (0..clustering.num_nodes())
            .map(|i| clustering.cluster_at(i).unwrap())
            .collect();
        assert_eq!(communities.len(), 1);
    }

    #[test]
    fn test_sprs_leiden_two_cliques() {
        // Two cliques of 4 nodes each, connected by a weak bridge
        let mut edges = Vec::new();
        // Clique 1: nodes 0-3
        for i in 0..4usize {
            for j in (i + 1)..4 {
                edges.push((i, j, 1.0));
            }
        }
        // Clique 2: nodes 4-7
        for i in 4..8usize {
            for j in (i + 1)..8 {
                edges.push((i, j, 1.0));
            }
        }
        // Weak bridge
        edges.push((3, 4, 0.01));

        let mat = symmetric_csr(8, &edges);
        let view = SprsNetworkView::new(&mat).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);

        let (_improved, clustering) =
            leiden_view(&view, None, Some(2), Some(0.5), None, &mut rng, false, None).unwrap();

        // improved flag depends on resolution and graph structure
        let communities: std::collections::HashSet<usize> = (0..clustering.num_nodes())
            .map(|i| clustering.cluster_at(i).unwrap())
            .collect();
        assert_eq!(communities.len(), 2);
        // Verify within-clique membership
        assert_eq!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(1).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(2).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(3).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(4).unwrap(),
            clustering.cluster_at(5).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(4).unwrap(),
            clustering.cluster_at(6).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(4).unwrap(),
            clustering.cluster_at(7).unwrap()
        );
        assert_ne!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(4).unwrap()
        );
    }

    #[test]
    fn test_sprs_leiden_weighted_communities() {
        // Three-node graph: strong edge between 0-1, weak edge to 2
        let mat = symmetric_csr(3, &[(0, 1, 10.0), (1, 2, 0.01), (0, 2, 0.01)]);
        let view = SprsNetworkView::new(&mat).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);

        let (_, clustering) =
            leiden_view(&view, None, Some(2), Some(0.5), None, &mut rng, false, None).unwrap();

        // Nodes 0 and 1 should be in the same community
        assert_eq!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(1).unwrap()
        );
    }

    #[test]
    fn test_sprs_leiden_disconnected_components() {
        // Two disconnected pairs
        let mat = symmetric_csr(4, &[(0, 1, 1.0), (2, 3, 1.0)]);
        let view = SprsNetworkView::new(&mat).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);

        let (_, clustering) =
            leiden_view(&view, None, Some(2), Some(0.5), None, &mut rng, false, None).unwrap();

        // Each pair should form its own community
        assert_eq!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(1).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(2).unwrap(),
            clustering.cluster_at(3).unwrap()
        );
        assert_ne!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(2).unwrap()
        );
    }

    #[test]
    fn test_sprs_leiden_karate_club() {
        // Zachary's karate club via sprs
        let edges: &[(usize, usize, f64)] = &[
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0),
            (0, 4, 1.0),
            (0, 5, 1.0),
            (0, 6, 1.0),
            (0, 7, 1.0),
            (0, 8, 1.0),
            (0, 10, 1.0),
            (0, 11, 1.0),
            (0, 12, 1.0),
            (0, 13, 1.0),
            (0, 17, 1.0),
            (0, 19, 1.0),
            (0, 21, 1.0),
            (0, 31, 1.0),
            (1, 2, 1.0),
            (1, 3, 1.0),
            (1, 7, 1.0),
            (1, 13, 1.0),
            (1, 17, 1.0),
            (1, 19, 1.0),
            (1, 21, 1.0),
            (1, 30, 1.0),
            (2, 3, 1.0),
            (2, 7, 1.0),
            (2, 8, 1.0),
            (2, 9, 1.0),
            (2, 13, 1.0),
            (2, 27, 1.0),
            (2, 28, 1.0),
            (2, 32, 1.0),
            (3, 7, 1.0),
            (3, 12, 1.0),
            (3, 13, 1.0),
            (4, 6, 1.0),
            (4, 10, 1.0),
            (5, 6, 1.0),
            (5, 10, 1.0),
            (5, 16, 1.0),
            (6, 16, 1.0),
            (8, 30, 1.0),
            (8, 32, 1.0),
            (8, 33, 1.0),
            (9, 33, 1.0),
            (13, 33, 1.0),
            (14, 32, 1.0),
            (14, 33, 1.0),
            (15, 32, 1.0),
            (15, 33, 1.0),
            (18, 32, 1.0),
            (18, 33, 1.0),
            (19, 33, 1.0),
            (20, 32, 1.0),
            (20, 33, 1.0),
            (22, 32, 1.0),
            (22, 33, 1.0),
            (23, 25, 1.0),
            (23, 27, 1.0),
            (23, 29, 1.0),
            (23, 32, 1.0),
            (23, 33, 1.0),
            (24, 25, 1.0),
            (24, 27, 1.0),
            (24, 31, 1.0),
            (25, 31, 1.0),
            (26, 29, 1.0),
            (26, 33, 1.0),
            (27, 33, 1.0),
            (28, 31, 1.0),
            (28, 33, 1.0),
            (29, 32, 1.0),
            (29, 33, 1.0),
            (30, 32, 1.0),
            (30, 33, 1.0),
            (31, 32, 1.0),
            (31, 33, 1.0),
            (32, 33, 1.0),
        ];

        let mat = symmetric_csr(34, edges);
        let view = SprsNetworkView::new(&mat).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);

        let (_improved, clustering) = leiden_view(
            &view,
            None,
            Some(2),
            Some(0.05),
            None,
            &mut rng,
            false,
            None,
        )
        .unwrap();

        // improved flag depends on resolution and graph structure
        assert_eq!(clustering.num_nodes(), 34);
        let communities: std::collections::HashSet<usize> = (0..clustering.num_nodes())
            .map(|i| clustering.cluster_at(i).unwrap())
            .collect();
        assert!(communities.len() >= 2);
        assert!(communities.len() <= 34);
    }

    #[test]
    fn test_sprs_leiden_single_node() {
        let mat = symmetric_csr(1, &[]);
        let view = SprsNetworkView::new(&mat).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);

        let (_, clustering) = leiden_view(
            &view,
            None,
            Some(2),
            Some(0.05),
            None,
            &mut rng,
            false,
            None,
        )
        .unwrap();

        assert_eq!(clustering.num_nodes(), 1);
        assert_eq!(clustering.cluster_at(0).unwrap(), 0);
    }

    #[test]
    fn test_sprs_view_num_edges() {
        let mat = symmetric_csr(4, &[(0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (0, 0, 0.5)]);
        let view = SprsNetworkView::new(&mat).unwrap();

        // 3 undirected non-self-loop edges (self-loops excluded from num_edges)
        assert_eq!(view.num_edges(), 3);
    }
}
