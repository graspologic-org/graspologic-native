use rand::Rng;

use crate::clustering::Clustering;
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
    node: usize,
    cluster: usize,
    level: u32,
    parent_cluster: Option<usize>,
    is_final_cluster: bool,
}

impl HierarchicalCluster {
    pub fn new(
        node: usize,
        cluster: usize,
        level: u32,
        parent_cluster: Option<usize>,
        is_final_cluster: bool,
    ) -> HierarchicalCluster {
        return HierarchicalCluster {
            node,
            cluster,
            level,
            parent_cluster,
            is_final_cluster,
        };
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
) -> Result<(bool, Vec<HierarchicalCluster>), CoreError>
    where
        T: Rng + Clone + Send,
{
    let (_, top_level_clustering) = leiden(network, clustering, iterations, resolution, randomness, rng, use_modularity)?;
    // TODO: create top level clustering entries for hierarchical output
    let mut work_queue: VecDeque<HierarchicalWork> = VecDeque::new();
    for subnetwork in network.subnetworks_iter(
        top_level_clustering,
        Some(max_cluster_size)
    ) {
        work_queue.push_back(HierarchicalWork {
            subnetwork: subnetwork.subnetwork,
            level: 1
        });
    }
    while !work_queue.is_empty() {
        let work_item: HierarchicalWork = work_queue.pop_front().unwrap();
        let (_, subnetwork_clustering) = leiden(
            &work_item.network,
            None,
            iterations,
            resolution,
            randomness,
            rng,
            use_modularity,
        )?;
        // TODO: create hierarchical clustering entry and add to output array
        for subnetwork in work_item
            .network
            .subnetworks_iter(subnetwork_clustering, Some(max_cluster_size)) {
            // create work item entry
            work_queue.push_back(
                HierarchicalWork {
                    subnetwork: subnetwork.subnetwork,
                    level: work_item.level + 1,
                }
            );
        }
    }
    return Err(CoreError::ClusterIndexingError);
}
