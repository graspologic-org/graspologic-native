// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::{HashMap, HashSet};

use rand::Rng;

use crate::clustering::{ClusterItem, Clustering};
use crate::errors::CoreError;
use crate::network::prelude::*;
use crate::resolution::adjust_resolution;

use super::full_network_clustering;
use super::subnetwork;
use super::subnetwork::SubnetworkClusteringGenerator;

const DEFAULT_ITERATIONS: usize = 1;

/// Performs the Leiden community detection algorithm on a `CompactNetwork`.
///
/// Each outer iteration runs the full Leiden cycle on the original network:
/// 1. **Local moving** — greedily moves nodes between clusters to maximize the
///    quality function.
/// 2. **Refinement** — stochastically splits each cluster into sub-clusters to
///    escape local optima.
/// 3. **Aggregation** — builds an induced (coarsened) network from the refined
///    clustering and recursively repeats steps 1–3 on it until the network
///    stops shrinking (inner convergence).
/// 4. **Mapping back** — projects the coarsened clustering onto the original nodes.
///
/// Because these phases include randomness, running multiple outer iterations
/// gives the algorithm additional chances to escape suboptimal partitions.
///
/// # Parameters
///
/// - `network`: The network to cluster.
/// - `clustering`: Optional initial clustering. If `None`, each node starts in
///   its own singleton cluster.
/// - `iterations`: Number of outer iterations (default 1). Each iteration
///   re-runs the full cycle on the original network, using the previous
///   clustering as the starting point. This refines a single solution
///   progressively (distinct from `trials`, which is implemented at the
///   Python layer and runs independent attempts keeping the best).
/// - `resolution`: Quality-function resolution (default 1.0, must be > 0).
///   Higher values produce more communities; lower values produce fewer.
/// - `randomness`: Controls exploration during refinement (default 1e-2, must
///   be > 0). Higher values allow more exploration of the partition space.
/// - `rng`: A seeded random number generator for reproducibility.
/// - `use_modularity`: If `true`, optimizes modularity; if `false`, uses CPM.
///   The network must be constructed appropriately for the chosen mode.
/// - `max_local_moving_iterations`: Limits the number of node-processing sweeps
///   within a single local-moving call. `None` or `Some(0)` means unlimited.
pub fn leiden<T>(
    network: &CompactNetwork,
    clustering: Option<Clustering>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    randomness: Option<f64>,
    rng: &mut T,
    use_modularity: bool,
    max_local_moving_iterations: Option<u32>,
) -> Result<(bool, Clustering), CoreError>
where
    T: Rng + Clone + Send,
{
    let iterations: usize = iterations.unwrap_or(DEFAULT_ITERATIONS);
    let randomness: f64 = randomness.unwrap_or(subnetwork::DEFAULT_RANDOMNESS);
    let max_local: u32 = max_local_moving_iterations.unwrap_or(0);

    let adjusted_resolution: f64 = adjust_resolution(resolution, network, use_modularity);

    if randomness <= 0_f64 || adjusted_resolution <= 0_f64 {
        return Err(CoreError::ParameterRangeError);
    } else if network.num_nodes() == 0 {
        return Err(CoreError::EmptyNetworkError);
    }

    let mut clustering: Clustering =
        clustering.unwrap_or(Clustering::as_self_clusters(network.num_nodes()));

    guarantee_clustering_sanity(network, &mut clustering)?;

    let mut improved: bool = false;

    for _i in 0..iterations {
        improved |= improve_clustering(
            network,
            &mut clustering,
            use_modularity,
            adjusted_resolution,
            randomness,
            rng,
            max_local,
        )?;
    }

    Ok((improved, clustering))
}

/// Like [`leiden`], but operates on any [`NetworkView`] implementation for zero-copy support.
///
/// The first local-moving pass runs directly on the provided view (avoiding
/// materialization). If aggregation is needed, a `CompactNetwork` is created
/// internally for the recursive phases. This gives the best of both worlds:
/// zero-copy for the expensive initial pass on large external data structures,
/// with full recursive aggregation when needed.
pub fn leiden_view<N, T>(
    network: &N,
    clustering: Option<Clustering>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    randomness: Option<f64>,
    rng: &mut T,
    use_modularity: bool,
    max_local_moving_iterations: Option<u32>,
) -> Result<(bool, Clustering), CoreError>
where
    N: NetworkView,
    T: Rng + Clone + Send,
{
    let iterations: usize = iterations.unwrap_or(DEFAULT_ITERATIONS);
    let randomness: f64 = randomness.unwrap_or(subnetwork::DEFAULT_RANDOMNESS);
    let max_local: u32 = max_local_moving_iterations.unwrap_or(0);

    let adjusted_resolution: f64 = adjust_resolution(resolution, network, use_modularity);

    if randomness <= 0_f64 || adjusted_resolution <= 0_f64 {
        return Err(CoreError::ParameterRangeError);
    } else if network.num_nodes() == 0 {
        return Err(CoreError::EmptyNetworkError);
    }

    let mut clustering: Clustering =
        clustering.unwrap_or(Clustering::as_self_clusters(network.num_nodes()));

    // Note: guarantee_clustering_sanity needs neighbors iteration, which NetworkView provides
    guarantee_clustering_sanity_view(network, &mut clustering)?;

    let mut improved: bool = false;

    for _i in 0..iterations {
        improved |= improve_clustering_view(
            network,
            &mut clustering,
            use_modularity,
            adjusted_resolution,
            randomness,
            rng,
            max_local,
        )?;
    }

    Ok((improved, clustering))
}

/// Single outer-iteration pass using a generic NetworkView.
///
/// Runs local moving on the view (zero-copy), then — if clusters were formed —
/// materializes a CompactNetwork and delegates to [`improve_clustering_recursive`]
/// for the refinement and aggregation phases. Inner recursion always runs to
/// convergence (no depth limit).
fn improve_clustering_view<N, T>(
    network: &N,
    clustering: &mut Clustering,
    use_modularity: bool,
    adjusted_resolution: f64,
    randomness: f64,
    rng: &mut T,
    max_local_moving_iterations: u32,
) -> Result<bool, CoreError>
where
    N: NetworkView,
    T: Rng + Clone + Send,
{
    // Run local moving on the generic view (zero-copy for CSR)
    let mut improved: bool = full_network_clustering::full_network_clustering(
        network,
        clustering,
        adjusted_resolution,
        rng,
        max_local_moving_iterations,
    )?;

    if clustering.next_cluster_id() < network.num_nodes() {
        // Clusters were formed — materialize to CompactNetwork for refinement
        // and recursive aggregation.
        let compact_network = network.to_compact_network();

        improved |= improve_clustering_recursive(
            &compact_network,
            clustering,
            use_modularity,
            adjusted_resolution,
            randomness,
            rng,
            max_local_moving_iterations,
        )?;
    }
    Ok(improved)
}

/// Refinement and recursive aggregation phase — always operates on CompactNetwork.
///
/// Given a clustering produced by local moving, this function:
/// 1. Refines each cluster via stochastic sub-clustering.
/// 2. Builds an induced (coarsened) network from the refined clustering.
/// 3. If the induced network is smaller, recursively calls [`improve_clustering`]
///    on it (which repeats LM → refine → aggregate until convergence).
/// 4. If the induced network is NOT smaller (no aggregation progress), runs one
///    final LM pass on it without further recursion to avoid infinite oscillation.
/// 5. Maps the induced-network clustering back onto the original nodes.
fn improve_clustering_recursive<T>(
    network: &CompactNetwork,
    clustering: &mut Clustering,
    use_modularity: bool,
    adjusted_resolution: f64,
    randomness: f64,
    rng: &mut T,
    max_local_moving_iterations: u32,
) -> Result<bool, CoreError>
where
    T: Rng + Clone + Send,
{
    let nodes_by_cluster: Vec<Vec<CompactNodeId>> = clustering.nodes_per_cluster();
    let subnetworks_iterator = network.subnetworks_iter(clustering, &nodes_by_cluster, None);
    let num_nodes_per_cluster: Vec<u64> = clustering.num_nodes_per_cluster();

    let num_subnetworks: usize = clustering.next_cluster_id();

    clustering.reset_next_cluster_id();

    let mut num_nodes_per_cluster_induced_network: Vec<usize> = Vec::with_capacity(num_subnetworks);
    let max_subnetwork_size: u64 = *num_nodes_per_cluster.iter().max().unwrap();
    let mut subnetwork_clusterer =
        SubnetworkClusteringGenerator::with_capacity(max_subnetwork_size as usize);

    for item in subnetworks_iterator {
        if num_nodes_per_cluster[item.id] == 1 && item.subnetwork.num_nodes() == 0 {
            let single_node_vec: &Vec<CompactNodeId> = &nodes_by_cluster[item.id];
            let singleton_node: &usize = single_node_vec
                .first()
                .expect("There should be one node here");
            clustering.update_cluster_at(*singleton_node, clustering.next_cluster_id())?;
            num_nodes_per_cluster_induced_network.push(1);
        } else if item.subnetwork.num_nodes() == 0 {
            // Multi-node cluster with no internal edges — split into singletons.
            let cluster_nodes: &Vec<CompactNodeId> = &nodes_by_cluster[item.id];
            for node in cluster_nodes {
                clustering.update_cluster_at(*node, clustering.next_cluster_id())?;
                num_nodes_per_cluster_induced_network.push(1);
            }
        } else {
            let subnetwork_clustering: Clustering = subnetwork_clusterer.subnetwork_clustering(
                item.subnetwork.compact(),
                use_modularity,
                adjusted_resolution,
                randomness,
                rng,
            )?;
            num_nodes_per_cluster_induced_network.push(subnetwork_clustering.next_cluster_id());
            clustering.merge_subnetwork_clustering(&item.subnetwork, &subnetwork_clustering);
        }
    }

    let induced_clustering_network: CompactNetwork =
        network.induce_clustering_network(clustering)?;

    let mut induced_network_clustering = initial_clustering_for_induced(
        num_nodes_per_cluster_induced_network,
        induced_clustering_network.num_nodes(),
    );

    let mut improved = false;

    if induced_clustering_network.num_nodes() < network.num_nodes() {
        // Induced network is smaller — recurse to convergence.
        improved |= improve_clustering(
            &induced_clustering_network,
            &mut induced_network_clustering,
            use_modularity,
            adjusted_resolution,
            randomness,
            rng,
            max_local_moving_iterations,
        )?;
    } else {
        // No shrinkage — run one final LM pass on the induced network
        // (refinement may have split clusters that LM can re-merge) but
        // don't recurse further to avoid infinite oscillation.
        improved |= full_network_clustering::full_network_clustering(
            &induced_clustering_network,
            &mut induced_network_clustering,
            adjusted_resolution,
            rng,
            max_local_moving_iterations,
        )?;
    }
    clustering.merge_clustering(&induced_network_clustering);

    Ok(improved)
}

fn guarantee_clustering_sanity_view<N: NetworkView>(
    network: &N,
    clustering: &mut Clustering,
) -> Result<(), CoreError> {
    let mut node_neighbors: HashMap<CompactNodeId, HashSet<CompactNodeId>> = HashMap::new();
    for node in 0..network.num_nodes() {
        let mut neighbors: HashSet<CompactNodeId> = HashSet::new();
        for neighbor in network.neighbors_for(node) {
            neighbors.insert(neighbor.id);
        }
        node_neighbors.insert(node, neighbors);
    }
    let mut cluster_membership: HashMap<ClusterId, HashSet<CompactNodeId>> = HashMap::new();
    for ClusterItem { node_id, cluster } in clustering.into_iter() {
        let cluster_members: &mut HashSet<CompactNodeId> =
            cluster_membership.entry(cluster).or_default();
        cluster_members.insert(node_id);
    }

    for cluster_members in cluster_membership.values() {
        if cluster_members.len() > 1 {
            for cluster_member in cluster_members {
                let neighbors = node_neighbors.get(cluster_member).unwrap();
                if neighbors.is_disjoint(cluster_members) {
                    let new_cluster: ClusterId = clustering.next_cluster_id();
                    clustering.update_cluster_at(*cluster_member, new_cluster)?;
                }
            }
        }
    }
    Ok(())
}

/// Single outer-iteration pass on a CompactNetwork.
///
/// Runs local moving, then — if any nodes were merged — delegates to
/// [`improve_clustering_recursive`] for the refinement and aggregation phases.
/// Inner recursion always runs to convergence (no depth limit).
fn improve_clustering<T>(
    network: &CompactNetwork,
    clustering: &mut Clustering,
    use_modularity: bool,
    adjusted_resolution: f64,
    randomness: f64,
    rng: &mut T,
    max_local_moving_iterations: u32,
) -> Result<bool, CoreError>
where
    T: Rng + Clone + Send,
{
    // Local moving: greedily reassign nodes to maximize the quality function
    let mut improved: bool = full_network_clustering::full_network_clustering(
        network,
        clustering,
        adjusted_resolution,
        rng,
        max_local_moving_iterations,
    )?;

    if clustering.next_cluster_id() < network.num_nodes() {
        improved |= improve_clustering_recursive(
            network,
            clustering,
            use_modularity,
            adjusted_resolution,
            randomness,
            rng,
            max_local_moving_iterations,
        )?;
    }
    Ok(improved)
}

fn initial_clustering_for_induced(
    num_nodes_per_cluster_induced_network: Vec<usize>,
    num_nodes: usize,
) -> Clustering {
    // Create an initial clustering for the induced network based on the non-refined clustering
    let mut clusters_induced_network: Vec<usize> = Vec::with_capacity(num_nodes);
    for (num_nodes_per_induced_cluster_index, repetitions) in
        num_nodes_per_cluster_induced_network.iter().enumerate()
    {
        // fill num_nodes_per_induced_cluster_index into positions from clusters_induced_network_index to clusters_induced_network_index + num_nodes_per_cluster_reduced_network[num_nodes_per_induced_cluster_index]
        clusters_induced_network.extend(std::iter::repeat_n(
            num_nodes_per_induced_cluster_index,
            *repetitions,
        ));
    }
    let next_cluster_id: usize = match clusters_induced_network.last() {
        Some(largest_cluster) => *largest_cluster + 1,
        None => 0,
    };
    Clustering::as_defined(clusters_induced_network, next_cluster_id)
}

fn guarantee_clustering_sanity(
    network: &CompactNetwork,
    clustering: &mut Clustering,
) -> Result<(), CoreError> {
    // verify initial clustering provided is in a sane state for leiden to operate
    // any node in a cluster must either be a singleton in that cluster or be connected to at least
    // one other node in that cluster
    let mut node_neighbors: HashMap<CompactNodeId, HashSet<CompactNodeId>> = HashMap::new();
    for CompactNodeItem { id: node, .. } in network.into_iter() {
        let mut neighbors: HashSet<CompactNodeId> = HashSet::new();
        for neighbor in network.neighbors_for(node) {
            neighbors.insert(neighbor.id);
        }
        node_neighbors.insert(node, neighbors);
    }
    let mut cluster_membership: HashMap<ClusterId, HashSet<CompactNodeId>> = HashMap::new();
    for ClusterItem { node_id, cluster } in clustering.into_iter() {
        let cluster_members: &mut HashSet<CompactNodeId> =
            cluster_membership.entry(cluster).or_default();
        cluster_members.insert(node_id);
    }

    for cluster_members in cluster_membership.values() {
        if cluster_members.len() > 1 {
            // we are only trying to move non-singletons if they don't have a possible connection
            for cluster_member in cluster_members {
                let neighbors = node_neighbors.get(cluster_member).unwrap();
                if neighbors.is_disjoint(cluster_members) {
                    // we have no reason to be in this partition, because we have no links to anyone
                    // else in it. we should make our own partition, with ___ and ___.
                    let new_cluster: ClusterId = clustering.next_cluster_id();
                    clustering.update_cluster_at(*cluster_member, new_cluster)?;
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::{LabeledNetwork, LabeledNetworkBuilder};

    // todo: this is common to LabeledNetwork and here, and probably should just be written in one place?
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
        edges
    }

    #[test]
    fn test_initial_clustering_for_induced() {
        let num_nodes_per_cluster: Vec<usize> = vec![1, 1, 2, 3, 5, 8];
        let expected: Clustering = Clustering::as_defined(
            vec![0, 1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5],
            6,
        );

        let actual: Clustering = initial_clustering_for_induced(num_nodes_per_cluster, 20);
        assert_eq!(actual, expected);
        assert_eq!(actual.num_nodes(), 20);
    }

    #[test]
    fn test_guarantee_clustering_sanity() {
        let edges = edge_list();
        let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
        let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);
        let compact_network: &CompactNetwork = labeled_network.compact();
        let mut clustering: Clustering = Clustering::as_self_clusters(compact_network.num_nodes());
        // node 'a' and node 'h' do not share an edge
        let a_compact = labeled_network.compact_id_for("a".into()).unwrap();
        let h_compact = labeled_network.compact_id_for("h".into()).unwrap();
        clustering
            .update_cluster_at(a_compact, clustering.next_cluster_id())
            .expect("Updating this known cluster for a should work");
        clustering
            .update_cluster_at(h_compact, clustering[a_compact])
            .expect("Updating this known cluster for h should work");
        clustering.remove_empty_clusters();
        assert_eq!(clustering[a_compact], clustering[h_compact]);
        guarantee_clustering_sanity(compact_network, &mut clustering)
            .expect("guarantee clustering sanity should not throw an error");
        assert_ne!(clustering[a_compact], clustering[h_compact]);
        let isolate_clusters: Vec<ClusterId> = vec![clustering[a_compact], clustering[h_compact]];
        let mut isolates: HashSet<ClusterId> = HashSet::new();
        isolates.extend(isolate_clusters);
        clustering
            .into_iter()
            .filter(|item| item.node_id != a_compact && item.node_id != h_compact)
            .for_each(|item| {
                assert!(!isolates.contains(&item.cluster));
            })
    }

    #[test]
    fn test_max_local_moving_iterations_through_leiden() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let edges = edge_list();
        let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
        let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);

        let mut rng1: SmallRng = SmallRng::seed_from_u64(200);
        let mut rng2: SmallRng = SmallRng::seed_from_u64(200);

        // Very limited local moving
        let (_, clustering_limited) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng1,
            true,
            Some(1),
        )
        .unwrap();

        // Unlimited local moving
        let (_, clustering_unlimited) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng2,
            true,
            Some(0),
        )
        .unwrap();

        // Both should produce valid clusterings
        for node_id in 0..labeled_network.num_nodes() {
            assert!(clustering_limited.cluster_at(node_id).is_ok());
            assert!(clustering_unlimited.cluster_at(node_id).is_ok());
        }

        // Cluster count is not guaranteed to be monotonic with additional local-moving steps; only
        // sanity-check that both results are bounded and non-empty.
        let limited_clusters = clustering_limited.next_cluster_id();
        let unlimited_clusters = clustering_unlimited.next_cluster_id();
        assert!(limited_clusters >= 1 && limited_clusters <= labeled_network.num_nodes());
        assert!(unlimited_clusters >= 1 && unlimited_clusters <= labeled_network.num_nodes());
    }
}
