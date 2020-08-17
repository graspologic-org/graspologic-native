// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::io::{BufReader, Read};
use std::fs::File;
use std::collections::HashMap;

use crate::errors::NetworkError;

use super::compact_network::{CompactNeighbor, CompactNetwork, CompactNode, CompactNodeId};
use super::networks::NetworkDetails;
use super::{identifier, Edge};

pub struct LabeledNetwork {
    network_structure: CompactNetwork,
    labels_to_id: HashMap<String, usize>,
    id_to_labels: Vec<String>,
}

impl LabeledNetwork {
    /// Superficially this seems like an easy task. Get the edges, add them. But we don't *know*
    /// that the edges provided are already in sorted source order (e.g. all edges from A to <N>
    /// all appear sequentially in the list.
    /// So we must collect and guarantee that behavior with this function.
    pub fn from(
        edges: Vec<Edge>,
        use_modularity: bool,
    ) -> Self {
        let mut labels_to_id: HashMap<String, CompactNodeId> = HashMap::new();
        let mut id_to_labels: Vec<String> = Vec::new();
        let mut nodes: Vec<CompactNode> = Vec::new();
        let mut neighbors: Vec<CompactNeighbor> = Vec::with_capacity(edges.len());
        let mut node_to_neighbors: HashMap<usize, HashMap<usize, f64>> = HashMap::new();
        for (source, target, weight) in edges {
            // as we see nodes, we're going to give them an identity, either by reusing one already
            // given or by creating a new one in the order first seen in the edge list.
            let source_id: usize =
                identifier::identify(&mut labels_to_id, &mut id_to_labels, source);
            let target_id: usize =
                identifier::identify(&mut labels_to_id, &mut id_to_labels, target);
            node_to_neighbors
                .entry(source_id)
                .or_insert(HashMap::new())
                .entry(target_id)
                .or_insert(weight);
            node_to_neighbors
                .entry(target_id)
                .or_insert(HashMap::new())
                .entry(source_id)
                .or_insert(weight);
        }
        let mut total_self_links_edge_weight: f64 = 0_f64;
        for node_id in 0..id_to_labels.len() {
            let mut node_weight: f64 = 0_f64; // we are going to set the node_weight as the summation of edge weights regardless of whether we're using modularity or CPM, but if we are using CPM we won't bother to use it.
            let mut node_neighbors: Vec<(&usize, &f64)> = node_to_neighbors
                .get(&node_id)
                .unwrap()
                .into_iter()
                .collect();
            let neighbor_start: usize = neighbors.len();
            node_neighbors.sort_by(|a, b| a.0.cmp(b.0));
            for (neighbor_id, edge_weight) in node_neighbors {
                if *neighbor_id == node_id {
                    total_self_links_edge_weight += *edge_weight;
                } else {
                    node_weight += *edge_weight; // TODO: do we do this as well even if it's a self link?
                    neighbors.push((*neighbor_id, *edge_weight));
                }
            }
            let node_weight: f64 = if use_modularity { node_weight } else { 1_f64 };
            nodes.push((node_weight, neighbor_start));
        }

        let compact_network: CompactNetwork =
            CompactNetwork::from(nodes, neighbors, total_self_links_edge_weight);

        return LabeledNetwork {
            network_structure: compact_network,
            labels_to_id,
            id_to_labels,
        };
    }

    pub fn compact(&self) -> &CompactNetwork {
        return &self.network_structure;
    }

    pub fn compact_id_for(
        &self,
        id: &str,
    ) -> Option<usize> {
        return self.labels_to_id.get(id).cloned();
    }

    pub fn label_for(
        &self,
        compact_id: usize,
    ) -> &str {
        return self.id_to_labels[compact_id].as_str();
    }

    pub fn load_from(
        path: &str,
        separator: &str,
        source_index: usize,
        target_index: usize,
        weight_index: Option<usize>,
        skip_first_line: bool,
        use_modularity: bool
    ) -> Result<LabeledNetwork, NetworkError> {
        let minimum_required_length: usize = source_index
            .max(target_index)
            .max(weight_index.unwrap_or(target_index))
            + 1;
        let mut reader: BufReader<File> = BufReader::new(File::open(path)?);
        let mut contents = String::new();
        reader.read_to_string(&mut contents)?;
        let skip_lines: usize = if skip_first_line { 1 } else { 0 };
        let mut edges: Vec<Edge> = Vec::new();
        for line in contents.lines().skip(skip_lines) {
            if !line.is_empty() {
                let splits: Vec<&str> = line.split(separator).collect();
                if splits.len() < minimum_required_length {
                    return Err(NetworkError::EdgeFileFormatError);
                }
                let source: &str = splits[source_index];
                let target: &str = splits[target_index];
                let weight: f64 = match weight_index {
                    Some(weight_index) => splits[weight_index]
                        .parse::<f64>()
                        .map_err(|_err| NetworkError::EdgeFileFormatError)?,
                    None => 1_f64,
                };
                edges.push((source.into(), target.into(), weight));
            }
        }

        return Ok(LabeledNetwork::from(edges, use_modularity));
    }
}

impl NetworkDetails for LabeledNetwork {
    fn num_nodes(&self) -> usize {
        return self.network_structure.num_nodes();
    }

    fn num_edges(&self) -> usize {
        return self.network_structure.num_edges();
    }

    fn total_node_weight(&self) -> f64 {
        return self.network_structure.total_node_weight();
    }

    fn total_edge_weight(&self) -> f64 {
        return self.network_structure.total_edge_weight();
    }

    fn total_self_links_edge_weight(&self) -> f64 {
        return self.network_structure.total_self_links_edge_weight();
    }
}

#[cfg(test)]
pub mod tests {
    use super::super::compact_network::CompactNeighborItem;
    use super::*;
    use std::iter::FromIterator;

    /*
       same graph as the compact network graph, but our ids will be different based on order seen in edge list
       edges
       a, b, 2.0
       a, d, 1.0
       a, e, 1.0
       b, a, 2.0
       b, c, 6.0
       b, e, 1.0
       b, f, 4.0
       b, g, 3.0
       c, b, 6.0
       c, g, 3.0
       d, a, 1.0
       d, h, 11.0
       e, a, 1.0
       e, b, 1.0
       f, b, 4.0
       g, b, 3.0
       g, c, 3.0
       h, d, 11.0
    */

    fn edge_list() -> Vec<Edge> {
        let edges: Vec<Edge> = vec![
            ("a".into(), "b".into(), 2.0),
            ("a".into(), "d".into(), 1.0),
            ("a".into(), "e".into(), 1.0),
            ("b".into(), "a".into(), 2.0),
            ("b".into(), "c".into(), 6.0),
            ("b".into(), "e".into(), 1.0),
            ("b".into(), "f".into(), 4.0),
            ("b".into(), "g".into(), 3.0),
            ("c".into(), "b".into(), 6.0),
            ("c".into(), "g".into(), 3.0),
            ("d".into(), "a".into(), 1.0),
            ("d".into(), "h".into(), 11.0),
            ("e".into(), "a".into(), 1.0),
            ("e".into(), "b".into(), 1.0),
            ("f".into(), "b".into(), 4.0),
            ("g".into(), "b".into(), 3.0),
            ("g".into(), "c".into(), 3.0),
            ("h".into(), "d".into(), 11.0),
        ];
        return edges;
    }

    fn expected_label_mappings() -> (HashMap<String, usize>, Vec<String>) {
        let label_order: Vec<String> = vec![
            "a".into(),
            "b".into(),
            "d".into(),
            "e".into(),
            "c".into(),
            "f".into(),
            "g".into(),
            "h".into(),
        ];
        let label_to_id_vec: Vec<(String, usize)> = label_order
            .clone()
            .into_iter()
            .enumerate()
            .map(|(index, label)| (label, index))
            .collect();
        let label_to_id: HashMap<String, usize> = HashMap::from_iter(label_to_id_vec.into_iter());
        return (label_to_id, label_order);
    }

    #[test]
    fn test_from_modularity() {
        let edges = edge_list();
        let labeled_network: LabeledNetwork = LabeledNetwork::from(edges, true);
        let (expected_label_map, expected_label_order) = expected_label_mappings();
        assert_eq!(expected_label_order, labeled_network.id_to_labels);
        assert_eq!(expected_label_map, labeled_network.labels_to_id);
        // spot check
        let b: usize = *labeled_network.labels_to_id.get("b").unwrap();
        let a: usize = *labeled_network.labels_to_id.get("a").unwrap();
        let c: usize = *labeled_network.labels_to_id.get("c").unwrap();
        let e: usize = *labeled_network.labels_to_id.get("e").unwrap();
        let f: usize = *labeled_network.labels_to_id.get("f").unwrap();
        let g: usize = *labeled_network.labels_to_id.get("g").unwrap();
        let expected_neighbors: Vec<(usize, f64)> = vec![
            (a, 2.0),
            (e, 1.0), // `e` appears before `c` in the edge list given above, due to its relationship with `a`. it thus gets sorted lower into the neighbors array due to its smaller CompactNodeId
            (c, 6.0),
            (f, 4.0),
            (g, 3.0),
        ];
        let node = labeled_network.network_structure.node(b);
        assert_eq!(16.0, node.weight);
        let actual_neighbors: Vec<(usize, f64)> = node
            .neighbors()
            .map(
                |CompactNeighborItem {
                     id, edge_weight, ..
                 }| { (id, edge_weight) },
            )
            .collect();
        assert_eq!(expected_neighbors, actual_neighbors);
    }

    #[test]
    fn test_from_cpm() {
        let edges = edge_list();
        let labeled_network: LabeledNetwork = LabeledNetwork::from(edges, false);
        let (expected_label_map, expected_label_order) = expected_label_mappings();
        assert_eq!(expected_label_order, labeled_network.id_to_labels);
        assert_eq!(expected_label_map, labeled_network.labels_to_id);
        let b: usize = *labeled_network.labels_to_id.get("b").unwrap();
        assert_eq!(1.0, labeled_network.network_structure.node(b).weight);
    }
}
