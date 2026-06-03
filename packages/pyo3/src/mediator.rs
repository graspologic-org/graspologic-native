// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::HashMap;

use network_partitions::clustering::Clustering;
use network_partitions::errors::CoreError;
use network_partitions::leiden;
use network_partitions::network::prelude::*;
use network_partitions::quality;

use super::HierarchicalCluster;
use super::errors::PyLeidenError;
use rand::SeedableRng;
use rand::rngs::SmallRng;

pub fn leiden(
    edges: Vec<Edge>,
    starting_communities: Option<HashMap<String, usize>>,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    seed: Option<u64>,
    trials: u64,
    max_local_moving_iterations: Option<u32>,
) -> Result<(f64, HashMap<String, usize>), PyLeidenError> {
    if trials == 0 {
        return Err(PyLeidenError::ParameterRangeError(
            "trials must be >= 1".to_string(),
        ));
    }

    let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
    let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), use_modularity);

    let initial_clustering: Option<Clustering> = match starting_communities {
        Some(starting_communities) => Some(communities_to_clustering(
            &labeled_network,
            starting_communities,
        )?),
        None => None,
    };

    let mut rng: SmallRng = match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_rng(&mut rand::rng()),
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
            max_local_moving_iterations,
        )?;

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
    let clustering: HashMap<String, usize> = map_from(&labeled_network, &best_clustering.unwrap())?;

    Ok((best_quality_score, clustering))
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
    Ok(quality)
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
    max_local_moving_iterations: Option<u32>,
) -> Result<Vec<HierarchicalCluster>, PyLeidenError> {
    let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
    let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), use_modularity);

    let clustering: Option<Clustering> = match starting_communities {
        Some(starting_communities) => Some(communities_to_clustering(
            &labeled_network,
            starting_communities,
        )?),
        None => None,
    };
    let mut rng: SmallRng = match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_rng(&mut rand::rng()),
    };

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
        max_local_moving_iterations,
    )?;

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

    Ok(hierarchical_clustering)
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
    Ok(map)
}

pub fn leiden_csr(
    indptr: &[i64],
    indices: &[i32],
    data: &[f64],
    n_nodes: usize,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    seed: Option<u64>,
    trials: u64,
    max_local_moving_iterations: Option<u32>,
) -> Result<(f64, HashMap<usize, usize>), PyLeidenError> {
    use crate::scipy_csr::ScipyCsrView;
    use network_partitions::network::network_view::NetworkView;

    if trials == 0 {
        return Err(PyLeidenError::ParameterRangeError(
            "trials must be >= 1".to_string(),
        ));
    }

    let csr_view = ScipyCsrView::new(indptr, indices, data, n_nodes, use_modularity)
        .map_err(|e| PyLeidenError::ParameterRangeError(format!("CSR validation failed: {e}")))?;

    let mut rng: SmallRng = match seed {
        Some(seed) => SmallRng::seed_from_u64(seed),
        None => SmallRng::from_rng(&mut rand::rng()),
    };

    // Materialize CompactNetwork once for quality scoring (O(nnz) allocation)
    let compact_network = csr_view.to_compact_network();

    let mut best_quality_score: f64 = f64::MIN;
    let mut best_clustering: Option<Clustering> = None;

    for _i in 0..trials {
        // Zero-copy: runs local moving directly on the scipy CSR view
        let (_improved, clustering) = leiden::leiden_view(
            &csr_view,
            None,
            Some(iterations),
            Some(resolution),
            Some(randomness),
            &mut rng,
            use_modularity,
            max_local_moving_iterations,
        )?;

        let quality_score: f64 = quality::quality(
            &compact_network,
            &clustering,
            Some(resolution),
            use_modularity,
        )?;
        if quality_score > best_quality_score {
            best_quality_score = quality_score;
            best_clustering = Some(clustering);
        }
    }

    let clustering = best_clustering.unwrap();
    let mut result: HashMap<usize, usize> = HashMap::with_capacity(clustering.num_nodes());
    for item in &clustering {
        result.insert(item.node_id, item.cluster);
    }

    Ok((best_quality_score, result))
}

fn communities_to_clustering(
    network: &LabeledNetwork<String>,
    communities: HashMap<String, usize>,
) -> Result<Clustering, PyLeidenError> {
    // best effort for using a provided community mapping and keeping generated singleton ids
    // otherwise

    let mut clustering: Clustering = match communities.values().max() {
        Some(max_community_id) => {
            // if we have values, we have the max community id, which means we'll actually create
            // a cluster mapping where every node gets a singleton cluster starting after the actual
            // last max cluster.
            let mut cluster_mapping: Vec<usize> = Vec::with_capacity(network.num_nodes());
            let start: usize = max_community_id + 1;
            let end: usize = start + network.num_nodes();
            cluster_mapping.extend(start..end);
            Clustering::as_defined(cluster_mapping, end)
        }
        None => Clustering::as_self_clusters(network.num_nodes()),
    };

    for (node, community) in communities {
        let mapping: Option<CompactNodeId> = network.compact_id_for(node);
        if let Some(compact_node_id) = mapping {
            clustering
                .update_cluster_at(compact_node_id, community)
                .map_err(|_| PyLeidenError::ClusterIndexingError)?;
        }
    }

    // we can't easily know if there are empty clusters, so we'll just presume there are
    // and compress any gaps
    clustering.remove_empty_clusters();

    Ok(clustering)
}
