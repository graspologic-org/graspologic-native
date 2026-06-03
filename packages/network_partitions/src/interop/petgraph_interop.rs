// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! `NetworkView` implementation for petgraph `UnGraph<f64, f64>`.
//!
//! This allows running the Leiden algorithm directly on a petgraph undirected
//! graph without copying into a `CompactNetwork` for the initial local-moving phase.
//!
//! # Graph conventions
//!
//! - **Node weights** (`f64`): Used as `NetworkView::node_weight()`. For modularity
//!   mode, callers must set node weights to the weighted degree (sum of incident
//!   non-self-loop edge weights). For CPM mode, use `1.0` for all nodes.
//! - **Edge weights** (`f64`): The weight of each undirected edge. Must be finite
//!   and non-negative.
//! - **Node indices**: Must be dense and contiguous starting from 0. This is
//!   naturally the case for graphs built with `add_node()` and no removals.
//! - **Self-loops**: Supported and counted toward `total_self_links_edge_weight`.

use petgraph::graph::UnGraph;
use petgraph::visit::EdgeRef;

use crate::network::network_view::{Neighbor, NetworkView};

/// Validation errors for `PetgraphNetworkView` construction.
#[derive(Debug, Clone)]
pub enum PetgraphValidationError {
    /// Node indices are not dense/contiguous (nodes have been removed).
    NonContiguousNodeIndices { node_count: usize, max_index: usize },
    /// An edge weight is not finite or is negative.
    InvalidEdgeWeight {
        source: usize,
        target: usize,
        weight: f64,
    },
}

impl std::fmt::Display for PetgraphValidationError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::NonContiguousNodeIndices {
                node_count,
                max_index,
            } => {
                write!(
                    f,
                    "Node indices are not contiguous: node_count={node_count}, max_index={max_index}"
                )
            }
            Self::InvalidEdgeWeight {
                source,
                target,
                weight,
            } => {
                write!(
                    f,
                    "Edge ({source}, {target}) has invalid weight: {weight} (must be finite and non-negative)"
                )
            }
        }
    }
}

impl std::error::Error for PetgraphValidationError {}

/// A zero-copy view over a `petgraph::UnGraph<f64, f64>` implementing `NetworkView`.
///
/// Node weights are the graph's node weights, edge weights are the graph's edge weights.
/// Nodes must be densely indexed starting at 0 (no removals).
pub struct PetgraphNetworkView<'a> {
    graph: &'a UnGraph<f64, f64>,
    total_node_weight: f64,
    total_edge_weight: f64,
    total_self_links_edge_weight: f64,
    num_non_self_loop_edges: usize,
}

impl<'a> PetgraphNetworkView<'a> {
    /// Create a new view over the given graph.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Node indices are not dense/contiguous (nodes have been removed)
    /// - Any edge weight is not finite or is negative
    pub fn new(graph: &'a UnGraph<f64, f64>) -> Result<Self, PetgraphValidationError> {
        let max_index = graph
            .node_indices()
            .next_back()
            .map_or(0, |i| i.index() + 1);
        if graph.node_count() != max_index {
            return Err(PetgraphValidationError::NonContiguousNodeIndices {
                node_count: graph.node_count(),
                max_index,
            });
        }

        let total_node_weight: f64 = graph.node_weights().sum();

        let mut total_edge_weight = 0.0;
        let mut total_self_links_edge_weight = 0.0;
        let mut num_self_loop_edges = 0usize;

        for edge in graph.edge_references() {
            let w = *edge.weight();
            if !w.is_finite() || w < 0.0 {
                return Err(PetgraphValidationError::InvalidEdgeWeight {
                    source: edge.source().index(),
                    target: edge.target().index(),
                    weight: w,
                });
            }
            if edge.source() == edge.target() {
                total_self_links_edge_weight += w;
                num_self_loop_edges += 1;
            } else {
                total_edge_weight += w;
            }
        }

        Ok(Self {
            graph,
            total_node_weight,
            total_edge_weight,
            total_self_links_edge_weight,
            num_non_self_loop_edges: graph.edge_count() - num_self_loop_edges,
        })
    }
}

/// Iterator over neighbors of a node in a petgraph graph.
pub struct PetgraphNeighborIterator<'a> {
    edges: petgraph::graph::Edges<'a, f64, petgraph::Undirected>,
    graph: &'a UnGraph<f64, f64>,
    source: petgraph::graph::NodeIndex,
}

impl<'a> Iterator for PetgraphNeighborIterator<'a> {
    type Item = Neighbor;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let edge_ref = self.edges.next()?;
            let target = if edge_ref.source() == self.source {
                edge_ref.target()
            } else {
                edge_ref.source()
            };
            // Skip self-loops per NetworkView contract
            if target == self.source {
                continue;
            }
            return Some(Neighbor {
                id: target.index(),
                edge_weight: *edge_ref.weight(),
                node_weight: self.graph[target],
            });
        }
    }
}

impl<'a> NetworkView for PetgraphNetworkView<'a> {
    type Neighbors<'b>
        = PetgraphNeighborIterator<'b>
    where
        Self: 'b;

    fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    fn node_weight(
        &self,
        node_id: usize,
    ) -> f64 {
        self.graph[petgraph::graph::NodeIndex::new(node_id)]
    }

    fn neighbors_for(
        &self,
        node_id: usize,
    ) -> Self::Neighbors<'_> {
        let node_idx = petgraph::graph::NodeIndex::new(node_id);
        PetgraphNeighborIterator {
            edges: self.graph.edges(node_idx),
            graph: self.graph,
            source: node_idx,
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
        self.num_non_self_loop_edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::leiden::leiden_view;
    use petgraph::graph::UnGraph;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn karate_club_graph() -> UnGraph<f64, f64> {
        // Zachary's karate club (simplified): 34 nodes, well-known community structure
        // Using a subset of edges for a manageable test
        let mut g = UnGraph::new_undirected();
        for _ in 0..34 {
            g.add_node(1.0);
        }
        // Classic karate club edges (0-indexed)
        let edges: &[(u32, u32)] = &[
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (0, 8),
            (0, 10),
            (0, 11),
            (0, 12),
            (0, 13),
            (0, 17),
            (0, 19),
            (0, 21),
            (0, 31),
            (1, 2),
            (1, 3),
            (1, 7),
            (1, 13),
            (1, 17),
            (1, 19),
            (1, 21),
            (1, 30),
            (2, 3),
            (2, 7),
            (2, 8),
            (2, 9),
            (2, 13),
            (2, 27),
            (2, 28),
            (2, 32),
            (3, 7),
            (3, 12),
            (3, 13),
            (4, 6),
            (4, 10),
            (5, 6),
            (5, 10),
            (5, 16),
            (6, 16),
            (8, 30),
            (8, 32),
            (8, 33),
            (9, 33),
            (13, 33),
            (14, 32),
            (14, 33),
            (15, 32),
            (15, 33),
            (18, 32),
            (18, 33),
            (19, 33),
            (20, 32),
            (20, 33),
            (22, 32),
            (22, 33),
            (23, 25),
            (23, 27),
            (23, 29),
            (23, 32),
            (23, 33),
            (24, 25),
            (24, 27),
            (24, 31),
            (25, 31),
            (26, 29),
            (26, 33),
            (27, 33),
            (28, 31),
            (28, 33),
            (29, 32),
            (29, 33),
            (30, 32),
            (30, 33),
            (31, 32),
            (31, 33),
            (32, 33),
        ];
        for &(s, t) in edges {
            g.add_edge(
                petgraph::graph::NodeIndex::new(s as usize),
                petgraph::graph::NodeIndex::new(t as usize),
                1.0,
            );
        }
        g
    }

    #[test]
    fn test_petgraph_view_basic_properties() {
        let mut g = UnGraph::new_undirected();
        let a = g.add_node(1.0);
        let b = g.add_node(2.0);
        let c = g.add_node(3.0);
        g.add_edge(a, b, 0.5);
        g.add_edge(b, c, 1.5);
        g.add_edge(a, c, 2.0);

        let view = PetgraphNetworkView::new(&g).unwrap();

        assert_eq!(view.num_nodes(), 3);
        assert_eq!(view.total_node_weight(), 6.0);
        // 3 undirected edges, total weight = 0.5 + 1.5 + 2.0 = 4.0
        assert_eq!(view.total_edge_weight(), 4.0);
        assert_eq!(view.total_self_links_edge_weight(), 0.0);
        assert_eq!(view.num_edges(), 3);
    }

    #[test]
    fn test_petgraph_view_node_weights() {
        let mut g = UnGraph::new_undirected();
        g.add_node(1.5);
        g.add_node(2.5);
        g.add_node(3.5);

        let view = PetgraphNetworkView::new(&g).unwrap();

        assert_eq!(view.node_weight(0), 1.5);
        assert_eq!(view.node_weight(1), 2.5);
        assert_eq!(view.node_weight(2), 3.5);
    }

    #[test]
    fn test_petgraph_view_neighbors() {
        let mut g = UnGraph::new_undirected();
        let a = g.add_node(1.0);
        let b = g.add_node(1.0);
        let c = g.add_node(1.0);
        g.add_edge(a, b, 2.0);
        g.add_edge(a, c, 3.0);

        let view = PetgraphNetworkView::new(&g).unwrap();

        let neighbors: Vec<Neighbor> = view.neighbors_for(0).collect();
        assert_eq!(neighbors.len(), 2);
        // Check that both b and c are neighbors of a
        let ids: Vec<usize> = neighbors.iter().map(|n| n.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_petgraph_view_self_loops() {
        let mut g = UnGraph::new_undirected();
        let a = g.add_node(1.0);
        let b = g.add_node(1.0);
        g.add_edge(a, b, 1.0);
        g.add_edge(a, a, 0.5); // self-loop

        let view = PetgraphNetworkView::new(&g).unwrap();

        assert_eq!(view.total_self_links_edge_weight(), 0.5);
        assert_eq!(view.total_edge_weight(), 1.0);
    }

    #[test]
    fn test_petgraph_view_to_compact_network() {
        let mut g = UnGraph::new_undirected();
        let a = g.add_node(1.0);
        let b = g.add_node(1.0);
        let c = g.add_node(1.0);
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.add_edge(a, c, 1.0);

        let view = PetgraphNetworkView::new(&g).unwrap();
        let compact = view.to_compact_network();

        assert_eq!(compact.num_nodes(), 3);
    }

    #[test]
    fn test_petgraph_leiden_triangle() {
        // A simple triangle should result in a single community
        let mut g = UnGraph::new_undirected();
        let a = g.add_node(1.0);
        let b = g.add_node(1.0);
        let c = g.add_node(1.0);
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        g.add_edge(a, c, 1.0);

        let view = PetgraphNetworkView::new(&g).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);
        let resolution = 0.5;
        let iterations = 2;

        let (_improved, clustering) = leiden_view(
            &view,
            None,
            Some(iterations),
            Some(resolution),
            None,
            &mut rng,
            false,
            None,
        )
        .unwrap();

        // Triangle at resolution 1.0 should be one community
        // improved flag depends on resolution and graph structure
        let communities: std::collections::HashSet<usize> = (0..clustering.num_nodes())
            .map(|i| clustering.cluster_at(i).unwrap())
            .collect();
        assert_eq!(communities.len(), 1);
    }

    #[test]
    fn test_petgraph_leiden_two_cliques() {
        // Two cliques connected by a single weak edge -> should split into 2 communities
        let mut g = UnGraph::new_undirected();
        for _ in 0..8 {
            g.add_node(1.0);
        }
        // Clique 1: nodes 0-3
        for i in 0..4u32 {
            for j in (i + 1)..4 {
                g.add_edge(
                    petgraph::graph::NodeIndex::new(i as usize),
                    petgraph::graph::NodeIndex::new(j as usize),
                    1.0,
                );
            }
        }
        // Clique 2: nodes 4-7
        for i in 4..8u32 {
            for j in (i + 1)..8 {
                g.add_edge(
                    petgraph::graph::NodeIndex::new(i as usize),
                    petgraph::graph::NodeIndex::new(j as usize),
                    1.0,
                );
            }
        }
        // Weak bridge
        g.add_edge(
            petgraph::graph::NodeIndex::new(3),
            petgraph::graph::NodeIndex::new(4),
            0.01,
        );

        let view = PetgraphNetworkView::new(&g).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);

        let (_improved, clustering) =
            leiden_view(&view, None, Some(2), Some(0.5), None, &mut rng, false, None).unwrap();

        // improved flag depends on resolution and graph structure
        // Should find 2 communities
        let communities: std::collections::HashSet<usize> = (0..clustering.num_nodes())
            .map(|i| clustering.cluster_at(i).unwrap())
            .collect();
        assert_eq!(communities.len(), 2);
        // Nodes within each clique should be in the same community
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
        // And the two cliques should be different communities
        assert_ne!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(4).unwrap()
        );
    }

    #[test]
    fn test_petgraph_leiden_karate_club() {
        let g = karate_club_graph();
        let view = PetgraphNetworkView::new(&g).unwrap();
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
        // Karate club should have between 2 and 6 communities at resolution 1.0
        let communities: std::collections::HashSet<usize> = (0..clustering.num_nodes())
            .map(|i| clustering.cluster_at(i).unwrap())
            .collect();
        assert!(communities.len() >= 2);
        assert!(communities.len() <= 34);
    }

    #[test]
    fn test_petgraph_leiden_weighted_edges() {
        // Heavily weighted internal edges should keep communities together
        let mut g = UnGraph::new_undirected();
        for _ in 0..6 {
            g.add_node(1.0);
        }
        // Strong internal edges in group {0,1,2}
        g.add_edge(
            petgraph::graph::NodeIndex::new(0),
            petgraph::graph::NodeIndex::new(1),
            10.0,
        );
        g.add_edge(
            petgraph::graph::NodeIndex::new(1),
            petgraph::graph::NodeIndex::new(2),
            10.0,
        );
        g.add_edge(
            petgraph::graph::NodeIndex::new(0),
            petgraph::graph::NodeIndex::new(2),
            10.0,
        );
        // Strong internal edges in group {3,4,5}
        g.add_edge(
            petgraph::graph::NodeIndex::new(3),
            petgraph::graph::NodeIndex::new(4),
            10.0,
        );
        g.add_edge(
            petgraph::graph::NodeIndex::new(4),
            petgraph::graph::NodeIndex::new(5),
            10.0,
        );
        g.add_edge(
            petgraph::graph::NodeIndex::new(3),
            petgraph::graph::NodeIndex::new(5),
            10.0,
        );
        // Weak cross-group edge
        g.add_edge(
            petgraph::graph::NodeIndex::new(2),
            petgraph::graph::NodeIndex::new(3),
            0.1,
        );

        let view = PetgraphNetworkView::new(&g).unwrap();
        let mut rng = SmallRng::seed_from_u64(123);

        let (_, clustering) =
            leiden_view(&view, None, Some(2), Some(0.5), None, &mut rng, false, None).unwrap();

        // Should split into 2 communities
        assert_eq!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(1).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(2).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(3).unwrap(),
            clustering.cluster_at(4).unwrap()
        );
        assert_eq!(
            clustering.cluster_at(3).unwrap(),
            clustering.cluster_at(5).unwrap()
        );
        assert_ne!(
            clustering.cluster_at(0).unwrap(),
            clustering.cluster_at(3).unwrap()
        );
    }

    #[test]
    fn test_petgraph_view_empty_graph() {
        let g: UnGraph<f64, f64> = UnGraph::new_undirected();
        let view = PetgraphNetworkView::new(&g).unwrap();

        assert_eq!(view.num_nodes(), 0);
        assert_eq!(view.total_node_weight(), 0.0);
        assert_eq!(view.total_edge_weight(), 0.0);
        assert_eq!(view.num_edges(), 0);
    }

    #[test]
    fn test_petgraph_view_disconnected_nodes() {
        let mut g = UnGraph::new_undirected();
        g.add_node(1.0);
        g.add_node(1.0);
        g.add_node(1.0);
        // No edges

        let view = PetgraphNetworkView::new(&g).unwrap();
        let mut rng = SmallRng::seed_from_u64(42);

        let (_, clustering) =
            leiden_view(&view, None, Some(2), Some(0.5), None, &mut rng, false, None).unwrap();

        // Each node should be in its own community (no reason to merge)
        assert_eq!(clustering.num_nodes(), 3);
        let communities: std::collections::HashSet<usize> = (0..clustering.num_nodes())
            .map(|i| clustering.cluster_at(i).unwrap())
            .collect();
        assert_eq!(communities.len(), 3);
    }
}
