use rand::Rng;

use crate::clustering::Clustering;
use crate::errors::CoreError;
use crate::network::Network;
use super::leiden::leiden;

pub const MAX_CLUSTER_SIZE: u32 = 1000;

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
    network: &Network,
    clustering: Option<Clustering>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    randomness: Option<f64>,
    rng: &mut T,
    use_modularity: bool,
    max_cluster_size: Option<u32>,
) -> Result<(bool, Vec<HierarchicalCluster>), CoreError>
    where
        T: Rng + Clone + Send,
{
    let max_cluster_size: u32 = max_cluster_size.unwrap_or(MAX_CLUSTER_SIZE);
    let (_, top_level_clustering) = leiden(network, clustering, iterations, resolution, randomness, rng, use_modularity)?;
    let nodes_by_cluster: Vec<Vec<usize>> = top_level_clustering.nodes_per_cluster();
    for nodes_within_cluster in nodes_by_cluster {

    }
    return Err(CoreError::ClusterIndexingError);
}
