// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::{HashMap, HashSet};

use network_partitions::clustering::Clustering;
use network_partitions::errors::CoreError;
use network_partitions::leiden::leiden as leiden_internal;
use network_partitions::log;
use network_partitions::network::{Network, NetworkBuilder};
use network_partitions::quality;
use network_partitions::safe_vectors::SafeVectors;

use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

use super::errors::PyLeidenError;

pub fn leiden(
    edges: Vec<(String, String, f64)>,
    starting_communities: Option<HashMap<String, usize>>,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    seed: Option<u64>,
) -> Result<(bool, f64, HashMap<String, usize>), PyLeidenError> {
    log!(
        "Building a NetworkBuilder for a Network whose clustering is measured by {}",
        if use_modularity { "modularity" } else { "CPM" }
    );
    let builder: NetworkBuilder = NetworkBuilder::builder(use_modularity);
    log!("Adding {} edges to network builder", edges.len());
    let builder: NetworkBuilder = builder.from(edges);

    log!("Loaded List[(str, str, f64)] into network builder");
    let network: Network = builder.build();

    log!("Network built from NetworkBuilder");
    let clustering: Option<Clustering> = match starting_communities {
        Some(starting_communities) => {
            Some(communities_to_clustering(&network, starting_communities)?)
        }
        None => None,
    };

    log!("Mapped any starting communities from a dictionary into a clustering");

    let mut rng: XorShiftRng = match seed {
        Some(seed) => XorShiftRng::seed_from_u64(seed),
        None => XorShiftRng::from_entropy(),
    };

    let (improved, clustering) = leiden_internal(
        &network,
        clustering,
        Some(iterations),
        Some(resolution),
        Some(randomness),
        &mut rng,
        use_modularity,
    )?;

    log!("Completed leiden process");
    let quality_score: f64 =
        quality::quality(&network, &clustering, Some(resolution), use_modularity)?;

    log!("Calculated quality score");
    let clustering: HashMap<String, usize> = map_from(&network, &clustering)?;

    log!("Mapped the clustering back to a dictionary: {:?}");

    return Ok((improved, quality_score, clustering));
}

pub fn modularity(
    edges: Vec<(String, String, f64)>,
    communities: HashMap<String, usize>,
    resolution: f64,
) -> Result<f64, PyLeidenError> {
    let builder: NetworkBuilder = NetworkBuilder::for_modularity();
    let network: Network = builder.from(edges).build();
    let clustering: Clustering = communities_to_clustering(&network, communities)?;
    let quality: f64 = quality::quality(&network, &clustering, Some(resolution), true)?;
    return Ok(quality);
}

fn map_from(
    network: &Network,
    clustering: &Clustering,
) -> Result<HashMap<String, usize>, CoreError> {
    let mut map: HashMap<String, usize> = HashMap::with_capacity(clustering.num_nodes());
    for node_index in 0..clustering.num_nodes() {
        let cluster: usize = clustering.cluster_at(node_index)?;
        let node_name = network.node_name(node_index)?;
        map.insert(node_name, cluster);
    }
    return Ok(map);
}

fn communities_to_clustering(
    network: &Network,
    communities: HashMap<String, usize>,
) -> Result<Clustering, PyLeidenError> {
    // if node not found in internal network, bounce
    // if max(communities) > len(set(communities)), collapse result
    // if all nodes are mapped, cool
    // if not all nodes are mapped, generate integers for each new value
    let mut max_community: usize = 0;
    let mut all_communities: HashSet<usize> = HashSet::new();
    let mut node_to_community: Vec<usize> = vec![0; communities.len()];
    for (node, community) in communities {
        all_communities.insert(community);
        max_community = max_community.max(community);
        let mapped_node: usize = network
            .index_for_name(&node)
            .ok_or(PyLeidenError::InvalidCommunityMappingError)?;
        node_to_community[mapped_node] = community;
    }
    // let's validate that all the nodes in the internal network have an entry in the
    // starting communities
    for node_id in 0..network.num_nodes() {
        node_to_community
            .get_or_err(node_id, CoreError::ClusterIndexingError)
            .map_err(|_| PyLeidenError::InvalidCommunityMappingError)?;
    }

    let mut clustering: Clustering = Clustering::as_defined(node_to_community, max_community + 1);
    if clustering.next_cluster_id() != all_communities.len() {
        // we have some gaps, compress
        clustering.remove_empty_clusters();
    }
    return Ok(clustering);
}
