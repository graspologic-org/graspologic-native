// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use super::resolution::adjust_resolution;
use crate::clustering::Clustering;
use crate::errors::CoreError;
use crate::network::Network;

pub fn quality(
    network: &Network,
    clustering: &Clustering,
    resolution: Option<f64>,
    use_modularity: bool,
) -> Result<f64, CoreError> {
    let adjusted_resolution: f64 = adjust_resolution(resolution, network, use_modularity);

    let mut quality: f64 = 0_f64;

    for node_index in 0..network.num_nodes() {
        let node_cluster: usize = clustering.cluster_at(node_index)?;
        let (start_neighbor_node_index, end_neighbor_node_index) =
            network.neighbor_range(node_index)?;
        for neighbor_node_index in start_neighbor_node_index..end_neighbor_node_index {
            let neighbor_node: usize = network.neighbor_at(neighbor_node_index)?;
            let neighbor_weight: f64 = network.weight_at(neighbor_node_index)?;
            let neighbor_cluster: usize = clustering.cluster_at(neighbor_node)?;
            if neighbor_cluster == node_cluster {
                quality += neighbor_weight;
            }
        }
    }

    let mut cluster_weights: Vec<f64> = vec![0_f64; clustering.next_cluster_id()];
    for node_index in 0..network.num_nodes() {
        let cluster: usize = clustering.cluster_at(node_index)?;
        cluster_weights[cluster] += network.node_weight_at(node_index)?;
    }

    for cluster_weight in cluster_weights {
        quality -= cluster_weight.powi(2) * adjusted_resolution;
    }

    quality =
        quality / (2_f64 * network.total_edge_weight() + network.total_edge_weight_self_links());

    return Ok(quality);
}
