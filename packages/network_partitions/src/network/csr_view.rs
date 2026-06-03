// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! A CSR (Compressed Sparse Row) network view that implements `NetworkView`.
//!
//! This allows the Leiden algorithm to operate directly on CSR data without
//! requiring conversion to `CompactNetwork` for the initial local moving phase.
//! Recursive aggregation will still materialize a `CompactNetwork` internally.

use super::network_view::{Neighbor, NetworkView};

/// Validation errors for CSR network construction.
#[derive(Debug, Clone, PartialEq)]
pub enum CsrValidationError {
    /// `indptr` must have length `num_nodes + 1`.
    InvalidIndptrLength { expected: usize, actual: usize },
    /// `indptr[0]` must be 0.
    IndptrDoesNotStartAtZero { value: usize },
    /// `indices` and `data` must have the same length.
    IndicesDataLengthMismatch { indices_len: usize, data_len: usize },
    /// `indptr` must be monotonically non-decreasing.
    IndptrNotMonotonic { position: usize },
    /// The last entry of `indptr` must equal `indices.len()`.
    IndptrFinalMismatch { indptr_last: usize, nnz: usize },
    /// A neighbor index is out of bounds.
    IndexOutOfBounds {
        row: usize,
        neighbor_id: usize,
        num_nodes: usize,
    },
    /// A weight is not finite or is negative.
    InvalidWeight { row: usize, position: usize },
    /// Node weights length doesn't match num_nodes.
    NodeWeightsLengthMismatch { expected: usize, actual: usize },
}

impl std::fmt::Display for CsrValidationError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::InvalidIndptrLength { expected, actual } => {
                write!(f, "indptr length {actual} != num_nodes + 1 ({expected})")
            }
            Self::IndptrDoesNotStartAtZero { value } => {
                write!(f, "indptr[0] must be 0, got {value}")
            }
            Self::IndicesDataLengthMismatch {
                indices_len,
                data_len,
            } => {
                write!(f, "indices length {indices_len} != data length {data_len}")
            }
            Self::IndptrNotMonotonic { position } => {
                write!(f, "indptr not monotonic at position {position}")
            }
            Self::IndptrFinalMismatch { indptr_last, nnz } => {
                write!(f, "indptr last value {indptr_last} != nnz {nnz}")
            }
            Self::IndexOutOfBounds {
                row,
                neighbor_id,
                num_nodes,
            } => {
                write!(
                    f,
                    "row {row}: neighbor id {neighbor_id} >= num_nodes {num_nodes}"
                )
            }
            Self::InvalidWeight { row, position } => {
                write!(
                    f,
                    "row {row}: non-finite or negative weight at position {position}"
                )
            }
            Self::NodeWeightsLengthMismatch { expected, actual } => {
                write!(f, "node_weights length {actual} != num_nodes {expected}")
            }
        }
    }
}

impl std::error::Error for CsrValidationError {}

/// A CSR network view that borrows its data from external storage.
///
/// This is the primary zero-copy path for Python (scipy CSR) and Rust (sprs) interop.
/// The data is validated at construction time.
///
/// # Layout
///
/// - `indptr[i]..indptr[i+1]` gives the range of neighbors for node `i` in `indices`/`data`
/// - `indices[j]` is the neighbor node ID for edge `j`
/// - `data[j]` is the edge weight for edge `j`
/// - `node_weights[i]` is the weight of node `i`
///
/// # Invariants (enforced at construction)
///
/// - The graph is assumed undirected (symmetric adjacency); symmetry is NOT validated
/// - All indices are in bounds
/// - All weights are finite and non-negative
/// - `indptr` is monotonically non-decreasing and starts at 0
pub struct CsrNetworkView<'a> {
    indptr: &'a [usize],
    indices: &'a [usize],
    data: &'a [f64],
    node_weights: &'a [f64],
    // Precomputed totals (computed during validation)
    cached_total_node_weight: f64,
    cached_total_edge_weight: f64,
    cached_total_self_links_weight: f64,
    cached_num_edges: usize,
}

impl<'a> CsrNetworkView<'a> {
    /// Create a new CSR network view with validation.
    ///
    /// `node_weights` should have length `num_nodes`. For modularity mode,
    /// node weights should equal the sum of incident edge weights for each node.
    ///
    /// Returns an error if the data violates any structural invariants.
    pub fn new(
        indptr: &'a [usize],
        indices: &'a [usize],
        data: &'a [f64],
        node_weights: &'a [f64],
    ) -> Result<Self, CsrValidationError> {
        if indptr.is_empty() {
            return Err(CsrValidationError::InvalidIndptrLength {
                expected: 1,
                actual: 0,
            });
        }

        if indptr[0] != 0 {
            return Err(CsrValidationError::IndptrDoesNotStartAtZero { value: indptr[0] });
        }

        let num_nodes = indptr.len() - 1;

        if indices.len() != data.len() {
            return Err(CsrValidationError::IndicesDataLengthMismatch {
                indices_len: indices.len(),
                data_len: data.len(),
            });
        }

        if node_weights.len() != num_nodes {
            return Err(CsrValidationError::NodeWeightsLengthMismatch {
                expected: num_nodes,
                actual: node_weights.len(),
            });
        }

        // Validate indptr monotonicity
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i - 1] {
                return Err(CsrValidationError::IndptrNotMonotonic { position: i });
            }
        }

        // Validate last indptr matches nnz
        if indptr[num_nodes] != indices.len() {
            return Err(CsrValidationError::IndptrFinalMismatch {
                indptr_last: indptr[num_nodes],
                nnz: indices.len(),
            });
        }

        // Validate indices and weights, compute totals
        let mut total_edge_weight: f64 = 0.0;
        let mut total_self_links_weight: f64 = 0.0;
        let mut num_diag_entries: usize = 0;

        for row in 0..num_nodes {
            let start = indptr[row];
            let end = indptr[row + 1];
            for pos in start..end {
                let neighbor_id = indices[pos];
                if neighbor_id >= num_nodes {
                    return Err(CsrValidationError::IndexOutOfBounds {
                        row,
                        neighbor_id,
                        num_nodes,
                    });
                }
                let weight = data[pos];
                if !weight.is_finite() || weight < 0.0 {
                    return Err(CsrValidationError::InvalidWeight { row, position: pos });
                }
                total_edge_weight += weight;
                if neighbor_id == row {
                    total_self_links_weight += weight;
                    num_diag_entries += 1;
                }
            }
        }

        // Each undirected non-self edge is stored twice, so halve for the true total.
        // Self-loop weights are excluded from total_edge_weight (tracked separately).
        total_edge_weight = (total_edge_weight - total_self_links_weight) / 2.0;

        let cached_total_node_weight: f64 = node_weights.iter().sum();
        let cached_num_edges = (indices.len() - num_diag_entries) / 2;

        Ok(CsrNetworkView {
            indptr,
            indices,
            data,
            node_weights,
            cached_total_node_weight,
            cached_total_edge_weight: total_edge_weight,
            cached_total_self_links_weight: total_self_links_weight,
            cached_num_edges,
        })
    }

    /// The number of non-zero entries (directed edge count, i.e. 2x undirected edges).
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }
}

/// Iterator over neighbors of a node in a CSR view.
/// Skips self-loop entries (where neighbor_id == source node).
pub struct CsrNeighborIterator<'a> {
    indices: &'a [usize],
    data: &'a [f64],
    node_weights: &'a [f64],
    pos: usize,
    end: usize,
    source_node: usize,
}

impl Iterator for CsrNeighborIterator<'_> {
    type Item = Neighbor;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < self.end {
            let id = self.indices[self.pos];
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

impl<'a> NetworkView for CsrNetworkView<'a> {
    type Neighbors<'b>
        = CsrNeighborIterator<'b>
    where
        Self: 'b;

    fn num_nodes(&self) -> usize {
        self.indptr.len() - 1
    }

    fn node_weight(
        &self,
        node_id: usize,
    ) -> f64 {
        self.node_weights[node_id]
    }

    fn neighbors_for(
        &self,
        node_id: usize,
    ) -> Self::Neighbors<'_> {
        let start = self.indptr[node_id];
        let end = self.indptr[node_id + 1];
        CsrNeighborIterator {
            indices: self.indices,
            data: self.data,
            node_weights: self.node_weights,
            pos: start,
            end,
            source_node: node_id,
        }
    }

    fn total_node_weight(&self) -> f64 {
        self.cached_total_node_weight
    }

    fn total_edge_weight(&self) -> f64 {
        self.cached_total_edge_weight
    }

    fn total_self_links_edge_weight(&self) -> f64 {
        self.cached_total_self_links_weight
    }

    fn num_edges(&self) -> usize {
        self.cached_num_edges
    }

    fn node_weights(&self) -> Vec<f64> {
        self.node_weights.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Simple triangle: A-B-C-A with weight 1.0
    // Adjacency (symmetric):
    //   0: [1, 2]
    //   1: [0, 2]
    //   2: [0, 1]
    fn triangle_csr() -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<f64>) {
        let indptr = vec![0, 2, 4, 6];
        let indices = vec![1, 2, 0, 2, 0, 1];
        let data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        // For modularity: node weight = sum of incident edge weights
        let node_weights = vec![2.0, 2.0, 2.0];
        (indptr, indices, data, node_weights)
    }

    #[test]
    fn test_basic_construction() {
        let (indptr, indices, data, node_weights) = triangle_csr();
        let view = CsrNetworkView::new(&indptr, &indices, &data, &node_weights).unwrap();
        assert_eq!(view.num_nodes(), 3);
        assert_eq!(view.num_edges(), 3);
        assert_eq!(view.total_node_weight(), 6.0);
        assert_eq!(view.total_edge_weight(), 3.0);
        assert_eq!(view.total_self_links_edge_weight(), 0.0);
    }

    #[test]
    fn test_neighbors() {
        let (indptr, indices, data, node_weights) = triangle_csr();
        let view = CsrNetworkView::new(&indptr, &indices, &data, &node_weights).unwrap();

        let neighbors: Vec<Neighbor> = view.neighbors_for(0).collect();
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].id, 1);
        assert_eq!(neighbors[0].edge_weight, 1.0);
        assert_eq!(neighbors[1].id, 2);
    }

    #[test]
    fn test_self_loops() {
        // Node 0 has a self-loop
        let indptr = vec![0, 3, 5, 7];
        let indices = vec![0, 1, 2, 0, 2, 0, 1]; // node 0 -> self, 1, 2
        let data = vec![0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let node_weights = vec![2.5, 2.0, 2.0];

        let view = CsrNetworkView::new(&indptr, &indices, &data, &node_weights).unwrap();
        assert_eq!(view.total_self_links_edge_weight(), 0.5);
    }

    #[test]
    fn test_validation_indptr_monotonic() {
        let indptr = vec![0, 3, 2, 4]; // not monotonic
        let indices = vec![1, 2, 0, 2];
        let data = vec![1.0; 4];
        let node_weights = vec![1.0, 1.0, 1.0];

        let result = CsrNetworkView::new(&indptr, &indices, &data, &node_weights);
        assert!(matches!(
            result,
            Err(CsrValidationError::IndptrNotMonotonic { position: 2 })
        ));
    }

    #[test]
    fn test_validation_index_out_of_bounds() {
        let indptr = vec![0, 2, 4, 6];
        let indices = vec![1, 5, 0, 2, 0, 1]; // 5 is out of bounds for 3 nodes
        let data = vec![1.0; 6];
        let node_weights = vec![2.0, 2.0, 2.0];

        let result = CsrNetworkView::new(&indptr, &indices, &data, &node_weights);
        assert!(matches!(
            result,
            Err(CsrValidationError::IndexOutOfBounds {
                row: 0,
                neighbor_id: 5,
                num_nodes: 3
            })
        ));
    }

    #[test]
    fn test_validation_negative_weight() {
        let indptr = vec![0, 2, 4, 6];
        let indices = vec![1, 2, 0, 2, 0, 1];
        let data = vec![1.0, -1.0, 1.0, 1.0, 1.0, 1.0]; // negative weight
        let node_weights = vec![2.0, 2.0, 2.0];

        let result = CsrNetworkView::new(&indptr, &indices, &data, &node_weights);
        assert!(matches!(
            result,
            Err(CsrValidationError::InvalidWeight { row: 0, .. })
        ));
    }

    #[test]
    fn test_to_compact_network() {
        let (indptr, indices, data, node_weights) = triangle_csr();
        let view = CsrNetworkView::new(&indptr, &indices, &data, &node_weights).unwrap();

        let compact = view.to_compact_network();
        assert_eq!(NetworkView::num_nodes(&compact), 3);
        assert_eq!(compact.num_edges(), 3);
    }

    #[test]
    fn test_leiden_with_csr_view() {
        use crate::leiden::leiden;
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        // Two triangles connected by a weak link:
        // 0-1-2-0 (weight 10) and 3-4-5-3 (weight 10), 0-3 (weight 1)
        // Node 0: neighbors [1(10), 2(10), 3(1)]
        // Node 1: neighbors [0(10), 2(10)]
        // Node 2: neighbors [0(10), 1(10)]
        // Node 3: neighbors [0(1), 4(10), 5(10)]
        // Node 4: neighbors [3(10), 5(10)]
        // Node 5: neighbors [3(10), 4(10)]
        let indptr = vec![0, 3, 5, 7, 10, 12, 14];
        let indices = vec![
            1, 2, 3, // node 0
            0, 2, // node 1
            0, 1, // node 2
            0, 4, 5, // node 3
            3, 5, // node 4
            3, 4, // node 5
        ];
        let data = vec![
            10.0, 10.0, 1.0, // node 0
            10.0, 10.0, // node 1
            10.0, 10.0, // node 2
            1.0, 10.0, 10.0, // node 3
            10.0, 10.0, // node 4
            10.0, 10.0, // node 5
        ];
        // Modularity mode: node weight = sum of incident edges
        let node_weights = vec![21.0, 20.0, 20.0, 21.0, 20.0, 20.0];

        let view = CsrNetworkView::new(&indptr, &indices, &data, &node_weights).unwrap();
        let compact = view.to_compact_network();

        let mut rng = SmallRng::seed_from_u64(42);
        let (improved, clustering) =
            leiden(&compact, None, Some(1), None, None, &mut rng, true, None).unwrap();

        assert!(improved);
        // Should find 2 communities
        assert_eq!(clustering.next_cluster_id(), 2);
        // Nodes 0,1,2 should be in one cluster, 3,4,5 in another
        let c0 = clustering.cluster_at(0).unwrap();
        assert_eq!(clustering.cluster_at(1).unwrap(), c0);
        assert_eq!(clustering.cluster_at(2).unwrap(), c0);
        let c3 = clustering.cluster_at(3).unwrap();
        assert_eq!(clustering.cluster_at(4).unwrap(), c3);
        assert_eq!(clustering.cluster_at(5).unwrap(), c3);
        assert_ne!(c0, c3);
    }
}
