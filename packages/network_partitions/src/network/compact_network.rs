// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

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

use bumpalo::Bump;
use bumpalo::collections::Vec as BumpVec;

use crate::clustering::Clustering;
use std::ops::Range;
use std::collections::HashMap;

// Simple types that can be exposed
pub type CompactNodeId = usize;
pub type ClusterId = usize;
pub type ConnectionId = usize; // note: Maybe we shouldn't expose this?  It's not super pertinent outside of this module

// Internal tuples (the combination of these tuples plus the vector index associated with them makes the public struct)
type CompactNode = (f64, ConnectionId);
type CompactNeighbor = (CompactNodeId, f64);

#[derive(Debug)]
pub struct CompactNodeItem<'a> {
    id: CompactNodeId,
    weight: f64,
    neighbors: NeighborIterator<'a>
}

#[derive(Debug)]
pub struct CompactNeighborItem {
    id: ConnectionId,
    neighbor_id: CompactNodeId,
    edge_weight: f64,
}

#[derive(Debug)]
pub struct CompactNetwork<'a> {
    nodes: BumpVec<'a, CompactNode>,
    neighbors: BumpVec<'a, CompactNeighbor>,
}

#[derive(Debug)]
pub struct CompactSubnetwork<'a> {
    compact_network: CompactNetwork<'a>,
    node_id_map: BumpVec<'a, CompactNodeId>, // the subnetwork will get its own node IDs, and this vec will allow us to map back to the original
}

#[derive(Debug)]
pub struct CompactSubnetworkItem<'a> {
    id: ClusterId,
    subnetwork: CompactSubnetwork<'a>,
}

impl<'a> CompactNetwork<'a> {

    pub fn from(nodes: BumpVec<'a, CompactNode>, neighbors: BumpVec<'a, CompactNeighbor>) -> CompactNetwork<'a> {
        return CompactNetwork {
            nodes,
            neighbors
        };
    }

    fn neighbor_range(&self, node_id: CompactNodeId) -> Range<CompactNodeId> {
        let (_, neighbor_start) = self.nodes[node_id];
        let end_range: ConnectionId = if node_id < self.nodes.len() - 1 {
            self.nodes[node_id + 1].1
        } else {
            self.neighbors.len()
        };
        return neighbor_start..end_range;
    }

    fn node(&self, id: CompactNodeId) -> CompactNodeItem {
        let weight: &f64 = &self.nodes[id].0;
        let neighbor_range: Range<ConnectionId> = self.neighbor_range(id);
        let neighbor_start: ConnectionId = neighbor_range.start;
        let neighbors = NeighborIterator {
            compact_network: self,
            neighbor_range,
            current_neighbor: neighbor_start
        };
        return CompactNodeItem {
            id,
            weight: *weight,
            neighbors,
        };
    }

    pub fn subnetworks_iter<'b, 'c: 'a>(
        &self,
        clustering: &'b Clustering,
        subnetwork_minimum_size: Option<u32>
    ) -> SubnetworkIterator<'a, 'b> {
        let working_map: HashMap<CompactNodeId, CompactNodeId> = HashMap::with_capacity(clustering.next_cluster_id());
        let num_nodes_per_cluster: Vec<usize> = clustering.num_nodes_per_cluster();
        let subnetwork_bump_arena: Bump = Bump::new(); // something to try someday; pre-allocating based on projected sizes?  We should be able to determine maximum size based on self.
        let mut nodes_by_cluster: BumpVec<(ClusterId, BumpVec<CompactNodeId>)> = BumpVec::with_capacity_in(
            clustering.next_cluster_id(),
            &subnetwork_bump_arena
        );
        for cluster_id in 0..clustering.next_cluster_id() {
            let cluster_count = num_nodes_per_cluster[cluster_id];
            nodes_by_cluster.push((cluster_id, BumpVec::with_capacity_in(cluster_count, &subnetwork_bump_arena)));
        }

        for (node, cluster) in clustering.into_iter().enumerate() {
            nodes_by_cluster[cluster].1.push(node);
        }

        return SubnetworkIterator {
            compact_supernetwork: &self,
            clustering,
            clustered_nodes: nodes_by_cluster,
            current_clustered_nodes_index: 0,
            subnetwork_bump_arena,
            original_node_to_new_node_map: working_map
        };
    }
}

impl<'a> IntoIterator for &'a CompactNetwork<'a> {
    type Item = CompactNodeItem<'a>;
    type IntoIter = NodeIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        return NodeIterator {
            compact_network: &self,
            current_node: 0
        };
    }
}

pub struct NodeIterator<'a> {
    compact_network: &'a CompactNetwork<'a>,
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
    compact_network: &'a CompactNetwork<'a>,
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
                edge_weight
            };
            self.current_neighbor += 1;
            Some(item)
        } else {
            None
        }
    }
}

pub struct SubnetworkIterator<'a, 'b> {
    compact_supernetwork: &'a CompactNetwork<'a>, // original data to project a new subnetwork from
    clustering: &'b Clustering,
    clustered_nodes: BumpVec<'a, (ClusterId, BumpVec<'a, CompactNodeId>)>,
    current_clustered_nodes_index: usize, // current clustered nodes to process.  Note that this index does not necessarily imply it is the ClusterId.
    subnetwork_bump_arena: Bump, // this is a new bump arena, not to be confused with the supernetwork's bump arena
    original_node_to_new_node_map: HashMap<CompactNodeId, CompactNodeId>,
}

impl<'a, 'b> Iterator for SubnetworkIterator<'a, 'b> {
    type Item = CompactSubnetworkItem<'b>;
    fn next(&mut self) -> Option<Self::Item> {
        self.subnetwork_bump_arena.reset(); // clear the previous memory usage in the bump arena
        self.original_node_to_new_node_map.clear(); // do the same in our lookup map
        return if self.current_clustered_nodes_index == self.clustered_nodes.len() {
            None
        } else {
            let (cluster_id, nodes) = &self.clustered_nodes[self.current_clustered_nodes_index];

            // all 3 of these are in the bump arena and will be wiped out the next time we ask for a next() or when the bump arena itself is freed.
            let mut subnetwork_nodes: BumpVec<CompactNode> = bumpalo::vec![
                in &self.subnetwork_bump_arena;
                (0_f64, 0);
                nodes.len()
            ]; // in the bump arena, nodes.len() space AND make the values default to (0_f64, 0)
            let mut subnetwork_neighbors: BumpVec<CompactNeighbor> = BumpVec::new_in(&self.subnetwork_bump_arena); // we can't know this size in advance
            let mut subnetwork_node_map: BumpVec<CompactNodeId> = BumpVec::with_capacity_in(nodes.len(), &self.subnetwork_bump_arena); // we can know this size though

            let mut current_new_node_id: usize = 0;
            for subnetwork_node in nodes {
                // set up our to/from maps/lookups
                self.original_node_to_new_node_map.insert(*subnetwork_node, current_new_node_id); // note that we're not using the length of subnetwork_node_map, as it may/will grow faster than we iterate
                subnetwork_node_map.push(*subnetwork_node);

                // as we go we'll increment subnetwork node weight, but the starting neighbor position needs to be captured now
                let neighbor_start: usize = subnetwork_neighbors.len();
                let mut subnetwork_node_weight: f64 = 0_f64;

                for CompactNeighborItem { neighbor_id: neighbor_id, edge_weight: edge_weight, .. } in self.compact_supernetwork.node(*subnetwork_node).neighbors {
                    if self.clustering[neighbor_id] == *cluster_id {
                        subnetwork_node_weight += edge_weight;
                        let neighbor_id: usize = match self.original_node_to_new_node_map.get(&neighbor_id) {
                            Some(new_neighbor_id) => *new_neighbor_id,
                            None => {
                                let new_neighbor_id = subnetwork_node_map.len();
                                self.original_node_to_new_node_map.insert(neighbor_id, new_neighbor_id.clone());
                                subnetwork_node_map.push(neighbor_id);
                                new_neighbor_id
                            }
                        };
                        subnetwork_neighbors.push((neighbor_id, edge_weight));
                    }
                }

                subnetwork_nodes[current_new_node_id] = (subnetwork_node_weight, neighbor_start);
                current_new_node_id += 1;
            }

            let compact_network: CompactNetwork = CompactNetwork::from(subnetwork_nodes, subnetwork_neighbors);
            let subnetwork: CompactSubnetwork = CompactSubnetwork {
                compact_network,
                node_id_map: subnetwork_node_map
            };
            self.current_clustered_nodes_index += 1;
            Some(
                CompactSubnetworkItem {
                    id: *cluster_id,
                    subnetwork
                }
            )
        }

    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use bumpalo::collections::Vec as BumpVec;

    #[test]
    fn test_stuff() {
        let bump_arena: Bump = Bump::new();
        println!("Bump Arena Allocated Bytes: {:#?}", bump_arena.allocated_bytes());
        let neighbors: BumpVec<CompactNeighbor> = bumpalo::vec![
            in &bump_arena;
            (1, 3.0),
            (2, 2.1),
            (0, 3.0),
            (2, 5.0),
            (0, 2.1),
            (1, 5.0)
        ];
        let nodes: BumpVec<CompactNode> = bumpalo::vec![
            in &bump_arena;
            (1.0, 0),
            (2.0, 2),
            (1.0, 4),
        ];
        let compact_network: CompactNetwork = CompactNetwork {
            nodes,
            neighbors
        };
        println!("Bump Arena Allocated Bytes: {:#?}", bump_arena.allocated_bytes());

        for CompactNodeItem {id: node_id, weight: node_weight, neighbors: neighbor_iter} in &compact_network {
            println!("Node ID: {}", node_id);
            println!("Edges: ");
            for CompactNeighborItem { id: edge_id, neighbor_id: neighbor_node_id, edge_weight: edge_weight } in neighbor_iter {
                println!("\t{}: {} @ index {}", neighbor_node_id, edge_weight, edge_id);
            }
        }
    }
}