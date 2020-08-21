use rand::Rng;

use crate::clustering::{Clustering, ClusterItem};
use crate::errors::CoreError;
use crate::network::prelude::*;
use super::leiden::leiden;
use std::collections::VecDeque;

pub const MAX_CLUSTER_SIZE: u32 = 1000;

pub struct HierarchicalWork {
    subnetwork: CompactSubnetwork,
    level: u32,
}

pub struct HierarchicalCluster {
    node: CompactNodeId,
    cluster: ClusterId,
    level: u32,
    parent_cluster: Option<ClusterId>,
}

struct HierarchicalClustering {
    hierarchical_clusterings: Vec<HierarchicalCluster>
}

impl HierarchicalClustering {
    pub fn new(initial_clustering: &Clustering) -> Self {
        let hierarchical: Vec<HierarchicalCluster> = initial_clustering.into_iter().map(|item| {
            HierarchicalCluster {
                node: item.node_id,
                cluster: item.cluster,
                level: 0,
                parent_cluster: None,
            }
        }).collect();

        return HierarchicalClustering {
            hierarchical_clusterings: hierarchical
        };
    }

    pub fn insert_subnetwork_clustering(
        &mut self,
        subnetwork: CompactSubnetwork,
        subnetwork_clustering: Clustering,
        parent_cluster: ClusterId,
        starting_cluster_id: ClusterId,
        level: u32,
    ) {
        // subnetworks have the new id to old id mapping.
        for cluster_item in &subnetwork_clustering {
            self.hierarchical_clusterings.push(
                HierarchicalCluster {
                    node: subnetwork.node_id_map[cluster_item.node_id],
                    cluster: starting_cluster_id + cluster_item.cluster,
                    level: level,
                    parent_cluster: Some(parent_cluster)
                }
            )
        }
    }

}

pub fn hierarchical_leiden<T>(
    network: &CompactNetwork,
    clustering: Option<Clustering>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    randomness: Option<f64>,
    rng: &mut T,
    use_modularity: bool,
    max_cluster_size: u32,
) -> Result<Vec<HierarchicalCluster>, CoreError>
    where
        T: Rng + Clone + Send,
{
    let (_, mut updated_clustering) = leiden(network, clustering, iterations, resolution, randomness, rng, use_modularity)?;

    // TODO: create top level clustering entries for hierarchical output
    let mut hierarchical_clustering: HierarchicalClustering = HierarchicalClustering::new(&updated_clustering);
    let mut work_queue: VecDeque<HierarchicalWork> = VecDeque::new();
    let mut level: u32 = 1;

    for subnetwork in network.subnetworks_iter(
        &updated_clustering,
        Some(max_cluster_size)
    ) {
        work_queue.push_back(HierarchicalWork {
            subnetwork: subnetwork.subnetwork,
            level
        });
    }
    while !work_queue.is_empty() {
        let work_item: HierarchicalWork = work_queue.pop_front().unwrap();
        let (_, subnetwork_clustering) = leiden(
            &work_item.subnetwork.compact_network,
            None,
            iterations,
            resolution,
            randomness,
            rng,
            use_modularity,
        )?;
        for clustering_item in &subnetwork_clustering {
            updated_clustering.update_cluster_at(clustering_item.node_id, clustering_item.cluster);
        }
        // TODO: create hierarchical clustering entry and add to output array

        if work_queue.is_empty() {
            level += 1;
            for subnetwork in network.subnetworks_iter(&updated_clustering, Some(max_cluster_size)) {
                work_queue.push_back(HierarchicalWork {
                    subnetwork: subnetwork.subnetwork,
                    level
                });
            }
        }
    }
    return Ok(hierarchical_clustering.hierarchical_clusterings);
}
