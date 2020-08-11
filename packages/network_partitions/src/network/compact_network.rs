// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use super::identifier;
/// A CompactNetwork is a tightly packed neighbor-optimized datastructure that defines a
/// *undirected* Network's edges.
///
/// The structure is optimized for:
///  - Fast lookups of all the neighbors (and the edges' weights) for a given Node
///  - Iteration through all the Nodes in the Network
///  - Iteration through all the neighbors (and edge weights) for each Node
///
/// To that end, the structure is primarily defined as two related collections:
///  - A vector whose indices are the NodeIds (usize) and whose values are the first EdgeId in the
///    second collection
///  - A second vector whose indices are the EdgeIds (usize) and whose values are a tuple of
///    (NodeId, weight (f64)).
///  The second vector's entries make two guarantees: that all of the neighbors for a given node
///  will be continuous, and that the neighbors will be sorted in ascending order as per NodeId.
///
/// This layout will hopefully minimize memory paging with cache misses by locating all of the
/// edges for a given node within the fewest pages possible vs. a conceptually simpler pointer based
/// approach.
///
/// This also means that the neighbors for a given NodeId `x` can be described roughly as:
///  - `second_vec[first_vec[x]..first_vec[x+1]]` (except when x = first_vec.len() - 1)
///
/// These Vectors also allow us to determine the number of nodes (first_vec.len()) and the
/// number of edges (second_vec.len() / 2) very quickly.
///
/// Desired behaviors:
///  - Iterate through CompactNetwork by a tuple of (NodeId, NeighborIterator)
///  - NeighborIterator must yield a tuple of:
///    - EdgeId (index of the edge in the array)
///    - Neighbor's NodeId
///    - Edge weight (f64)
///
/// e.g. a Network defined by the edges:
///   0,1,3.0
///   0,2,2.1
///   1,0,3.0
///   1,2,5.0
///   2,0,2.1
///   2,1,5.0
///
/// And the following code:
/// for (node_id, neighbor_iter) in &compact_network {
///   println!("Node ID: {}", node_id);
///   println!("Edges: ");
///   for (edge_id, (neighbor_node_id, edge_weight)) in neighbor_iter {
///     println!("\t{}: {} @ index {}", neighbor_node_id, edge_weight, edge_id);
///   }
/// }
///
/// Would print:
///   Node ID: 0
///   Edges:
///       1: 3.0 @ index 0
///       2: 2.1 @ index 1
///   Node ID: 1
///   Edges:
///       0: 3.0 @ index 2
///       2: 5.0 @ index 3
///   Node ID: 2
///   Edges:
///       0: 2.1 @ index 4
///       1: 5.0 @ index 5
///
/// The EdgeId is the direct index into the edges array for fast lookups of a specific edge,
/// which is primarily useful to the subnetwork generation functions
use crate::clustering::{ClusterItem, Clustering};
use std::collections::HashMap;
use std::ops::Range;

// Simple types that can be exposed
pub type CompactNodeId = usize;
pub type ClusterId = usize;
pub type ConnectionId = usize; // note: Maybe we shouldn't expose this?  It's not super pertinent outside of this module

// the combination of these tuples plus the vector index associated with them makes the public struct
pub type CompactNode = (f64, ConnectionId);
pub type CompactNeighbor = (CompactNodeId, f64);

#[derive(Debug)]
pub struct CompactNodeItem<'a> {
    pub id: CompactNodeId,
    pub weight: f64,
    pub neighbors: NeighborIterator<'a>,
}

#[derive(Debug)]
pub struct CompactNeighborItem {
    pub id: ConnectionId,
    pub neighbor_id: CompactNodeId,
    pub edge_weight: f64,
}

#[derive(Debug, PartialEq)]
pub struct CompactNetwork {
    nodes: Vec<CompactNode>,
    neighbors: Vec<CompactNeighbor>,
    total_self_links_edge_weight: f64,
}

#[derive(Debug, PartialEq)]
pub struct CompactSubnetwork {
    compact_network: CompactNetwork,
    node_id_map: Vec<CompactNodeId>, // the subnetwork will get its own node IDs, and this vec will allow us to map back to the original
}

#[derive(Debug, PartialEq)]
pub struct CompactSubnetworkItem {
    id: ClusterId,
    subnetwork: CompactSubnetwork,
}

impl CompactNetwork {
    pub fn from(
        nodes: Vec<CompactNode>,
        neighbors: Vec<CompactNeighbor>,
        total_self_links_edge_weight: f64,
    ) -> CompactNetwork {
        return CompactNetwork {
            nodes,
            neighbors,
            total_self_links_edge_weight,
        };
    }

    fn neighbor_range(
        &self,
        node_id: CompactNodeId,
    ) -> Range<CompactNodeId> {
        let (_, neighbor_start) = self.nodes[node_id];
        let end_range: ConnectionId = if node_id < self.nodes.len() - 1 {
            self.nodes[node_id + 1].1
        } else {
            self.neighbors.len()
        };
        return neighbor_start..end_range;
    }

    pub fn node(
        &self,
        id: CompactNodeId,
    ) -> CompactNodeItem {
        let weight: &f64 = &self.nodes[id].0;
        let neighbor_range: Range<ConnectionId> = self.neighbor_range(id);
        let neighbor_start: ConnectionId = neighbor_range.start;
        let neighbors = NeighborIterator {
            compact_network: self,
            neighbor_range,
            current_neighbor: neighbor_start,
        };
        return CompactNodeItem {
            id,
            weight: *weight,
            neighbors,
        };
    }

    pub fn subnetworks_iter<'a, 'b>(
        &'a self,
        clustering: &'b Clustering,
        subnetwork_minimum_size: Option<u32>,
    ) -> SubnetworkIterator<'a, 'b> {
        let working_map: HashMap<CompactNodeId, CompactNodeId> =
            HashMap::with_capacity(clustering.next_cluster_id());
        let num_nodes_per_cluster: Vec<usize> = clustering.num_nodes_per_cluster();
        let mut nodes_by_cluster: Vec<(ClusterId, Vec<CompactNodeId>)> =
            Vec::with_capacity(clustering.next_cluster_id());
        let mut largest_cluster: usize = 0;
        for cluster_id in 0..clustering.next_cluster_id() {
            let cluster_count: usize = num_nodes_per_cluster[cluster_id];
            largest_cluster = largest_cluster.max(cluster_count);
            nodes_by_cluster.push((cluster_id, Vec::with_capacity(cluster_count)));
        }

        for ClusterItem { node_id, cluster } in clustering {
            nodes_by_cluster[cluster].1.push(node_id);
        }

        return SubnetworkIterator {
            compact_supernetwork: self,
            clustering,
            clustered_nodes: nodes_by_cluster,
            current_clustered_nodes_index: 0,
            original_node_to_new_node_map: working_map,
            subnetwork_minimum_size,
        };
    }
}

impl<'a> IntoIterator for &'a CompactNetwork {
    type Item = CompactNodeItem<'a>;
    type IntoIter = NodeIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        return NodeIterator {
            compact_network: &self,
            current_node: 0,
        };
    }
}

pub struct NodeIterator<'a> {
    compact_network: &'a CompactNetwork,
    current_node: CompactNodeId,
}

impl<'a> Iterator for NodeIterator<'a> {
    type Item = CompactNodeItem<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        return if self.current_node == self.compact_network.nodes.len() {
            None
        } else {
            let item = self.compact_network.node(self.current_node);
            self.current_node += 1;
            Some(item)
        };
    }
}

#[derive(Debug)]
pub struct NeighborIterator<'a> {
    compact_network: &'a CompactNetwork,
    neighbor_range: Range<ConnectionId>,
    current_neighbor: ConnectionId,
}

impl<'a> Iterator for NeighborIterator<'a> {
    type Item = CompactNeighborItem;

    fn next(&mut self) -> Option<Self::Item> {
        return if self.neighbor_range.contains(&self.current_neighbor) {
            let (neighbor_id, edge_weight) = self.compact_network.neighbors[self.current_neighbor];
            let item = CompactNeighborItem {
                id: self.current_neighbor,
                neighbor_id,
                edge_weight,
            };
            self.current_neighbor += 1;
            Some(item)
        } else {
            None
        };
    }
}

pub struct SubnetworkIterator<'a, 'b> {
    compact_supernetwork: &'a CompactNetwork, // original data to project a new subnetwork from
    clustering: &'b Clustering,
    clustered_nodes: Vec<(ClusterId, Vec<CompactNodeId>)>,
    current_clustered_nodes_index: usize, // current clustered nodes to process.  Note that this index does not necessarily imply it is the ClusterId.
    original_node_to_new_node_map: HashMap<CompactNodeId, CompactNodeId>,
    subnetwork_minimum_size: Option<u32>,
}

impl<'a, 'b> Iterator for SubnetworkIterator<'a, 'b> {
    type Item = CompactSubnetworkItem;
    fn next(&mut self) -> Option<Self::Item> {
        self.original_node_to_new_node_map.clear(); // do the same in our lookup map
        let next_valid_position: Option<usize> = match self.subnetwork_minimum_size {
            None => {
                if self.current_clustered_nodes_index == self.clustered_nodes.len() {
                    None
                } else {
                    Some(self.current_clustered_nodes_index)
                }
            }
            Some(subnetwork_minimum_size) => {
                let mut found: bool = false;
                let mut possibly_valid: usize = self.current_clustered_nodes_index;
                while possibly_valid != self.clustered_nodes.len() && !found {
                    if self.clustered_nodes[possibly_valid].1.len()
                        >= subnetwork_minimum_size as usize
                    {
                        found = true;
                    } else {
                        possibly_valid += 1;
                    }
                }
                if found {
                    Some(self.current_clustered_nodes_index)
                } else {
                    None
                }
            }
        };
        return match next_valid_position {
            Some(current) => {
                self.current_clustered_nodes_index = current;
                let (cluster_id, nodes) = &self.clustered_nodes[current];

                let mut subnetwork_nodes: Vec<CompactNode> = vec![(0_f64, 0); nodes.len()];
                let mut subnetwork_neighbors: Vec<CompactNeighbor> = Vec::new();
                let mut subnetwork_id_map: Vec<CompactNodeId> = Vec::with_capacity(nodes.len());

                for subnetwork_node in nodes {
                    // set up our to/from maps/lookups
                    let current_new_node_id: usize = identifier::identify(
                        &mut self.original_node_to_new_node_map,
                        &mut subnetwork_id_map,
                        *subnetwork_node,
                    );

                    let mut node_weight: f64 = 0_f64;
                    // starting neighbor position needs to be captured now
                    let neighbor_start: usize = subnetwork_neighbors.len();

                    for CompactNeighborItem {
                        neighbor_id,
                        edge_weight,
                        ..
                    } in self.compact_supernetwork.node(*subnetwork_node).neighbors
                    {
                        if self.clustering[neighbor_id] == *cluster_id {
                            node_weight += edge_weight;
                            let new_neighbor_id: usize = identifier::identify(
                                &mut self.original_node_to_new_node_map,
                                &mut subnetwork_id_map,
                                neighbor_id,
                            );
                            subnetwork_neighbors.push((new_neighbor_id, edge_weight));
                        }
                    }
                    subnetwork_nodes[current_new_node_id] = (node_weight, neighbor_start);
                }
                let subnetwork: CompactSubnetwork = CompactSubnetwork {
                    compact_network: CompactNetwork::from(
                        subnetwork_nodes,
                        subnetwork_neighbors,
                        0_f64,
                    ),
                    node_id_map: subnetwork_id_map,
                };
                self.current_clustered_nodes_index += 1;
                Some(CompactSubnetworkItem {
                    id: *cluster_id,
                    subnetwork,
                })
            }
            None => {
                self.current_clustered_nodes_index = self.clustered_nodes.len();
                None
            }
        };
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /*
       edges
       0, 1, 2.0
       0, 3, 1.0
       0, 4, 1.0
       1, 0, 2.0
       1, 2, 6.0
       1, 4, 1.0
       1, 5, 4.0
       1, 6, 3.0
       2, 1, 6.0
       2, 6, 3.0
       3, 0, 1.0
       3, 7, 11.0
       4, 0, 1.0
       4, 1, 1.0
       5, 1, 4.0
       6, 1, 3.0
       6, 2, 3.0
       7, 3, 11.0
    */

    fn fresh_network(self_links: f64) -> CompactNetwork {
        let nodes: Vec<CompactNode> = vec![
            (4_f64, 0),
            (16_f64, 3),
            (9_f64, 8),
            (12_f64, 10),
            (2_f64, 12),
            (4_f64, 14),
            (6_f64, 15),
            (11_f64, 17),
        ];
        let neighbors: Vec<CompactNeighbor> = vec![
            (1, 2_f64), // 0
            (3, 1_f64),
            (4, 1_f64),
            (0, 2_f64), // 1
            (2, 6_f64),
            (4, 1_f64),
            (5, 4_f64),
            (6, 3_f64),
            (1, 6_f64), // 2
            (6, 3_f64),
            (0, 1_f64), // 3
            (7, 11_f64),
            (0, 1_f64), // 4
            (1, 1_f64),
            (1, 4_f64), // 5
            (1, 3_f64), // 6
            (2, 3_f64),
            (3, 11_f64), // 7
        ];
        return CompactNetwork::from(nodes, neighbors, self_links);
    }

    #[test]
    fn test_node_iterator() {
        let network: CompactNetwork = fresh_network(0_f64);
        // testing the neighbor iterator is in another test, so we're only validating the IDs and the node weights in this case.
        let expected_like: Vec<(CompactNodeId, f64)> = vec![
            (0, 4_f64),
            (1, 16_f64),
            (2, 9_f64),
            (3, 12_f64),
            (4, 2_f64),
            (5, 4_f64),
            (6, 6_f64),
            (7, 11_f64),
        ];
        let mut index: usize = 0;
        for CompactNodeItem { id, weight, .. } in &network {
            let (expected_id, expected_weight) = expected_like[index];
            assert_eq!(expected_id, id);
            assert_eq!(expected_weight, weight);
            index += 1;
        }
        assert_eq!(8, index);
    }

    #[test]
    fn test_neighbor_iterator() {
        let network: CompactNetwork = fresh_network(0_f64);
        let expected_neighbors: Vec<(CompactNodeId, f64)> =
            vec![(0, 2_f64), (2, 6_f64), (4, 1_f64), (5, 4_f64), (6, 3_f64)];
        let node = network.node(1);
        let actual: Vec<CompactNeighborItem> = node.neighbors.into_iter().collect();
        assert_eq!(5, actual.len());

        for (
            index,
            CompactNeighborItem {
                neighbor_id,
                edge_weight,
                ..
            },
        ) in actual.into_iter().enumerate()
        {
            let (expected_neighbor_id, expected_edge_weight) = expected_neighbors[index];
            assert_eq!(expected_neighbor_id, neighbor_id);
            assert_eq!(edge_weight, expected_edge_weight);
        }
    }

    #[test]
    fn test_subnetwork_iterator() {
        let network: CompactNetwork = fresh_network(1_f64);
        let node_to_cluster: Vec<usize> = vec![0, 1, 1, 0, 0, 1, 1, 0];

        let clustering: Clustering = Clustering::as_defined(node_to_cluster, 2);

        let subnetwork1: CompactNetwork = CompactNetwork {
            nodes: vec![(2_f64, 0), (12_f64, 2), (1_f64, 4), (11_f64, 5)],
            neighbors: vec![
                (1, 1_f64),
                (2, 1_f64),
                (0, 1_f64),
                (3, 11_f64),
                (0, 1_f64),
                (1, 11_f64),
            ],
            total_self_links_edge_weight: 0_f64,
        };
        let subnetwork2: CompactNetwork = CompactNetwork {
            nodes: vec![(13_f64, 0), (9_f64, 3), (4_f64, 5), (6_f64, 6)],
            neighbors: vec![
                (1, 6_f64),
                (2, 4_f64),
                (3, 3_f64),
                (0, 6_f64),
                (3, 3_f64),
                (0, 4_f64),
                (0, 3_f64),
                (1, 3_f64),
            ],
            total_self_links_edge_weight: 0_f64,
        };
        let expected: Vec<CompactSubnetworkItem> = vec![
            CompactSubnetworkItem {
                id: 0,
                subnetwork: CompactSubnetwork {
                    compact_network: subnetwork1,
                    node_id_map: vec![0, 3, 4, 7],
                },
            },
            CompactSubnetworkItem {
                id: 1,
                subnetwork: CompactSubnetwork {
                    compact_network: subnetwork2,
                    node_id_map: vec![1, 2, 5, 6],
                },
            },
        ];
        let actual: Vec<CompactSubnetworkItem> =
            network.subnetworks_iter(&clustering, None).collect();
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_subnetwork_filters() {
        let network: CompactNetwork = fresh_network(1_f64);
        let node_to_cluster: Vec<usize> = vec![0, 1, 1, 0, 0, 1, 1, 2];

        let clustering: Clustering = Clustering::as_defined(node_to_cluster, 3);

        let subnetwork1: CompactNetwork = CompactNetwork {
            nodes: vec![(2_f64, 0), (1_f64, 2), (1_f64, 3)],
            neighbors: vec![(1, 1_f64), (2, 1_f64), (0, 1_f64), (0, 1_f64)],
            total_self_links_edge_weight: 0_f64,
        };
        let subnetwork2: CompactNetwork = CompactNetwork {
            nodes: vec![(13_f64, 0), (9_f64, 3), (4_f64, 5), (6_f64, 6)],
            neighbors: vec![
                (1, 6_f64),
                (2, 4_f64),
                (3, 3_f64),
                (0, 6_f64),
                (3, 3_f64),
                (0, 4_f64),
                (0, 3_f64),
                (1, 3_f64),
            ],
            total_self_links_edge_weight: 0_f64,
        };
        let expected: Vec<CompactSubnetworkItem> = vec![
            CompactSubnetworkItem {
                id: 0,
                subnetwork: CompactSubnetwork {
                    compact_network: subnetwork1,
                    node_id_map: vec![0, 3, 4],
                },
            },
            CompactSubnetworkItem {
                id: 1,
                subnetwork: CompactSubnetwork {
                    compact_network: subnetwork2,
                    node_id_map: vec![1, 2, 5, 6],
                },
            },
        ];
        let actual: Vec<CompactSubnetworkItem> =
            network.subnetworks_iter(&clustering, Some(3)).collect();
        assert_eq!(expected, actual);
    }
}
