// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/// A neighbor entry yielded by `NetworkView::neighbors_for`.
#[derive(Debug, Clone, Copy)]
pub struct Neighbor {
    pub id: usize,
    pub edge_weight: f64,
    pub node_weight: f64,
}

/// A read-only view of a network suitable for the Leiden algorithm's local moving phase.
///
/// This trait abstracts over different network representations (CSR, petgraph, etc.)
/// so the algorithm can operate generically without requiring a specific storage format.
///
/// # Invariants
///
/// Implementors must guarantee:
/// - The network is undirected: if node A lists node B as a neighbor with weight W,
///   then node B must list node A as a neighbor with weight W.
/// - All neighbor IDs returned by `neighbors_for` are `< num_nodes()`.
/// - `neighbors_for` does NOT yield self-loop entries (diagonal entries). Self-loop
///   weights are tracked separately via `total_self_links_edge_weight()`.
/// - Edge weights are finite and non-negative.
/// - Node IDs are dense integers in `0..num_nodes()`.
/// - `total_edge_weight()` equals half the sum of all non-self-loop edge weights
///   (since each undirected edge is stored twice in the adjacency representation).
///   Self-loop weights are excluded from this total.
/// - `total_self_links_edge_weight()` is the sum of all diagonal/self-loop weights.
/// - `total_node_weight()` equals the sum of all node weights.
/// - Node weights (for modularity) should equal the sum of incident non-self-loop
///   edge weights for that node.
pub trait NetworkView {
    /// Iterator type returned by `neighbors_for`.
    type Neighbors<'a>: Iterator<Item = Neighbor> + 'a
    where
        Self: 'a;

    /// The number of nodes in the network.
    fn num_nodes(&self) -> usize;

    /// The weight of a specific node.
    fn node_weight(
        &self,
        node_id: usize,
    ) -> f64;

    /// An iterator over the neighbors of a given node.
    fn neighbors_for(
        &self,
        node_id: usize,
    ) -> Self::Neighbors<'_>;

    /// The total weight across all nodes.
    fn total_node_weight(&self) -> f64;

    /// The total edge weight of the network (each undirected edge counted once).
    fn total_edge_weight(&self) -> f64;

    /// The total weight of self-loop edges.
    fn total_self_links_edge_weight(&self) -> f64;

    /// The number of edges in the network (each undirected edge counted once).
    fn num_edges(&self) -> usize {
        // Default: count neighbor entries and divide by 2 (undirected)
        (0..self.num_nodes())
            .map(|i| self.neighbors_for(i).count())
            .sum::<usize>()
            / 2
    }

    /// The sum of incident edge weights for each node.
    ///
    /// Returns a vector of length `num_nodes()` where entry `i` is the sum of
    /// edge weights for all edges incident to node `i`.
    fn total_edge_weight_per_node(&self) -> Vec<f64> {
        (0..self.num_nodes())
            .map(|i| self.neighbors_for(i).map(|n| n.edge_weight).sum())
            .collect()
    }

    /// All node weights as a vector.
    ///
    /// Returns a vector of length `num_nodes()` where entry `i` is `node_weight(i)`.
    fn node_weights(&self) -> Vec<f64> {
        (0..self.num_nodes()).map(|i| self.node_weight(i)).collect()
    }

    /// Materialize this view into a `CompactNetwork`.
    ///
    /// This is used when the algorithm needs to perform recursive aggregation
    /// (subnetworks, induced networks) which requires the full `CompactNetwork` API.
    /// The default implementation builds a `CompactNetwork` from the view's data.
    fn to_compact_network(&self) -> super::compact_network::CompactNetwork {
        use super::compact_network::{CompactNeighbor, CompactNode};

        let mut nodes: Vec<CompactNode> = Vec::with_capacity(self.num_nodes());
        let mut neighbors: Vec<CompactNeighbor> = Vec::new();

        for node_id in 0..self.num_nodes() {
            let weight = self.node_weight(node_id);
            let connection_start = neighbors.len();
            nodes.push((weight, connection_start));

            for neighbor in self.neighbors_for(node_id) {
                neighbors.push((neighbor.id, neighbor.edge_weight));
            }
        }

        super::compact_network::CompactNetwork::from(
            nodes,
            neighbors,
            self.total_self_links_edge_weight(),
        )
    }
}
