// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::{HashMap, HashSet};

use network_partitions::clustering::Clustering;
use network_partitions::errors::CoreError;
use network_partitions::leiden;
use network_partitions::log;
use network_partitions::network::prelude::*;
use network_partitions::quality;
use network_partitions::safe_vectors::SafeVectors;

use super::errors::PyLeidenError;
use super::HierarchicalCluster;
use crate::errors::InvalidCommunityMappingError;
use rand::{Rng, SeedableRng};
use rand_xorshift::XorShiftRng;

pub fn leiden(
    edges: Vec<Edge>,
    starting_communities: Option<HashMap<String, usize>>,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    seed: Option<u64>,
    trials: u64,
) -> Result<(f64, HashMap<String, usize>), PyLeidenError> {
    log!(
        "Building a LabeledNetwork for quality measured by {}",
        if use_modularity { "modularity" } else { "CPM" }
    );
    log!("Adding {} edges to network builder", edges.len());

    let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
    let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), use_modularity);

    log!("Network built from edges");
    let initial_clustering: Option<Clustering> = match starting_communities {
        Some(starting_communities) => Some(communities_to_clustering(
            &labeled_network,
            starting_communities,
        )?),
        None => None,
    };

    log!("Mapped any starting communities from a dictionary into a clustering");

    let mut rng: XorShiftRng = match seed {
        Some(seed) => XorShiftRng::seed_from_u64(seed),
        None => XorShiftRng::from_entropy(),
    };

    let compact_network: &CompactNetwork = labeled_network.compact();

    let mut best_quality_score: f64 = f64::MIN;
    let mut best_clustering: Option<Clustering> = None;

    for _i in 0..trials {
        let (_improved, clustering) = leiden::leiden(
            compact_network,
            initial_clustering.clone(),
            Some(iterations),
            Some(resolution),
            Some(randomness),
            &mut rng,
            use_modularity,
        )?;

        log!("Completed leiden process");
        let quality_score: f64 = quality::quality(
            compact_network,
            &clustering,
            Some(resolution),
            use_modularity,
        )?;
        if quality_score > best_quality_score {
            best_quality_score = quality_score;
            best_clustering = Some(clustering);
        }
    }

    log!("Calculated quality score");
    let clustering: HashMap<String, usize> = map_from(&labeled_network, &best_clustering.unwrap())?;

    log!("Mapped the clustering back to a dictionary: {:?}");

    return Ok((best_quality_score, clustering));
}

pub fn modularity(
    edges: Vec<Edge>,
    communities: HashMap<String, usize>,
    resolution: f64,
) -> Result<f64, PyLeidenError> {
    let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
    let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);
    let clustering: Clustering = communities_to_clustering(&labeled_network, communities)?;
    let quality: f64 = quality::quality(
        labeled_network.compact(),
        &clustering,
        Some(resolution),
        true,
    )?;
    return Ok(quality);
}

pub fn hierarchical_leiden(
    edges: Vec<Edge>,
    starting_communities: Option<HashMap<String, usize>>,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    max_cluster_size: u32,
    seed: Option<u64>,
) -> Result<Vec<HierarchicalCluster>, PyLeidenError> {
    log!(
        "Building a LabeledNetwork for quality measured by {}",
        if use_modularity { "modularity" } else { "CPM" }
    );
    log!("Adding {} edges to network builder", edges.len());

    let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
    let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), use_modularity);

    log!("Network built from edges");
    let clustering: Option<Clustering> = match starting_communities {
        Some(starting_communities) => Some(communities_to_clustering(
            &labeled_network,
            starting_communities,
        )?),
        None => None,
    };

    log!("Mapped any starting communities from a dictionary into a clustering");

    let mut rng: XorShiftRng = match seed {
        Some(seed) => XorShiftRng::seed_from_u64(seed),
        None => XorShiftRng::from_entropy(),
    };

    log!("Running hierarchical leiden over a network of {} nodes, {} edges, with a max_cluster_size of {}", labeled_network.num_nodes(), labeled_network.num_edges(), max_cluster_size);
    let compact_network: &CompactNetwork = labeled_network.compact();
    let internal_clusterings: Vec<leiden::HierarchicalCluster> = leiden::hierarchical_leiden(
        compact_network,
        clustering,
        Some(iterations),
        Some(resolution),
        Some(randomness),
        &mut rng,
        use_modularity,
        max_cluster_size,
    )?;

    log!("Completed hierarchical leiden process");

    let mut hierarchical_clustering: Vec<HierarchicalCluster> =
        Vec::with_capacity(internal_clusterings.len());
    for internal in internal_clusterings {
        let node = labeled_network.label_for(internal.node);
        hierarchical_clustering.push(HierarchicalCluster {
            node: node.into(),
            cluster: internal.cluster,
            level: internal.level,
            parent_cluster: internal.parent_cluster,
            is_final_cluster: internal.is_final_cluster,
        });
    }

    return Ok(hierarchical_clustering);
}

fn map_from(
    network: &LabeledNetwork<String>,
    clustering: &Clustering,
) -> Result<HashMap<String, usize>, CoreError> {
    let mut map: HashMap<String, usize> = HashMap::with_capacity(clustering.num_nodes());
    for item in clustering {
        let node_name = network.label_for(item.node_id);
        map.insert(node_name.into(), item.cluster);
    }
    return Ok(map);
}

fn communities_to_clustering(
    network: &LabeledNetwork<String>,
    communities: HashMap<String, usize>,
) -> Result<Clustering, PyLeidenError> {
    // if node not found in internal network, bounce
    // if max(communities) > len(set(communities)), collapse result
    // if all nodes are mapped, cool
    // if not all nodes are mapped, generate integers for each new value

    let mut clustering: Clustering = Clustering::as_self_clusters(network.num_nodes());

    let mut all_communities: HashSet<usize> = HashSet::new();

    for (node, community) in communities {
        all_communities.insert(community);
        let mapped_node: CompactNodeId = network
            .compact_id_for(node)
            .ok_or(PyLeidenError::InvalidCommunityMappingError)?;
        clustering
            .update_cluster_at(mapped_node, community)
            .map_err(|_| PyLeidenError::InvalidCommunityMappingError)?;
    }

    if clustering.next_cluster_id() != all_communities.len() {
        // we have some gaps, compress
        clustering.remove_empty_clusters();
    }

    return Ok(clustering);
}
