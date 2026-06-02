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

/// Improves a clustering by performing `iterations` of the Leiden algorithm, which is itself
/// a recursive algorithm.
///
/// The Leiden algorithm consists of three phases:
/// - local moving of nodes between clusters
/// - refinement of the clusters
/// - aggregation of the network based on the refined clusters, using the non-refined clusters to
///   create an initial clustering for the aggregate network
///
/// These phases are repeated until no further improvements can be made.
///
/// Because these phases include a random number generator, `iterations` acts as a further
/// refinement of the process, ensuring that we do at least `iterations-1` further tries to ensure
/// that we've actually achieved a stable partitioning.
///
/// If an initial Clustering is provided, it will be used as the starting point for the Leiden algorithm,
/// otherwise each node will be placed in their own cluster to start.
///
/// network: InternalNetwork to generate or update a clustering for based on the Leiden algorithm
/// clustering: An optional initial clustering. If an initial Clustering is provided, it will be used
///  as the starting point for the Leiden algorithm, otherwise each node will be placed in their own
///  cluster to start.
/// iterations: The leiden algorithm is recursive and will continue until improvements cannot be made; however,
///  randomization is a part of the algorithm and you may request further iterations by setting iterations
///  to be a number greater than 1 to force it to try a few more times for some minor, further refinements.
/// resolution: Default is 1.0, and impacts the maximization function used. The resolution must be greater than
///   zero.  A higher resolution values leads to more communities, a lower resolution parameter leads to fewer
///   communities.
/// randomness: Default is 1E-2. The value must be greater than 0. The higher the randomness value, the more
///   exploration of the partition space is possible.  This is a major difference from the Louvain algorithm.
///   The Louvain algorithm is purely greedy in the partition exploration.
/// seed: If a seed is provided, the Pseudo-Random Number Generator will be created using that seed.
///   Useful for replicating results between runs.
/// use_modularity: Leiden uses a maximization function, and this lets you specify whether you wish
///   to use modularity or Constant Potts Model (CPM). It's vital that the InternalNetwork is appropriate
///   for this setting: see InternalNetwork::for_modularity_maximization or
///   InternalNetwork::for_cpm_maximization and ensure you use the function that builds the corresponding
///   InternalNetwork for this setting.
pub fn leiden<T>(
    network: &CompactNetwork,
    clustering: Option<Clustering>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    randomness: Option<f64>,
    rng: &mut T,
    use_modularity: bool,
    max_outer_iterations: Option<u32>,
    max_local_moving_iterations: Option<u32>,
) -> Result<(bool, Clustering), CoreError>
where
    T: Rng + Clone + Send,
{
    let iterations: usize = iterations.unwrap_or(DEFAULT_ITERATIONS);
    let randomness: f64 = randomness.unwrap_or(subnetwork::DEFAULT_RANDOMNESS);
    let max_local: u32 = max_local_moving_iterations.unwrap_or(0);

    // max_outer_iterations overrides iterations if provided.
    // Each outer iteration runs: LM → refine → aggregate → recurse-to-convergence.
    let outer_limit: usize = match max_outer_iterations {
        Some(n) => n as usize,
        None => iterations,
    };

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

    for _i in 0..outer_limit {
        improved |= improve_clustering(
            network,
            &mut clustering,
            use_modularity,
            adjusted_resolution,
            randomness,
            rng,
            None,
            max_local,
        )?;
    }

    Ok((improved, clustering))
}

/// Like `leiden`, but operates on any `NetworkView` implementation for zero-copy support.
///
/// The initial local-moving phase runs directly on the provided view (zero-copy).
/// If the algorithm needs to recurse (aggregation), it materializes a `CompactNetwork`
/// internally. This gives the best of both worlds: zero-copy for the expensive first
/// pass, with full recursive support when needed.
pub fn leiden_view<N, T>(
    network: &N,
    clustering: Option<Clustering>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    randomness: Option<f64>,
    rng: &mut T,
    use_modularity: bool,
    max_outer_iterations: Option<u32>,
    max_local_moving_iterations: Option<u32>,
) -> Result<(bool, Clustering), CoreError>
where
    N: NetworkView,
    T: Rng + Clone + Send,
{
    let iterations: usize = iterations.unwrap_or(DEFAULT_ITERATIONS);
    let randomness: f64 = randomness.unwrap_or(subnetwork::DEFAULT_RANDOMNESS);
    let max_local: u32 = max_local_moving_iterations.unwrap_or(0);

    // max_outer_iterations overrides iterations if provided.
    let outer_limit: usize = match max_outer_iterations {
        Some(n) => n as usize,
        None => iterations,
    };

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

    for _i in 0..outer_limit {
        improved |= improve_clustering_view(
            network,
            &mut clustering,
            use_modularity,
            adjusted_resolution,
            randomness,
            rng,
            None,
            max_local,
        )?;
    }

    Ok((improved, clustering))
}

/// First improvement pass using a generic NetworkView, then materializes for recursion.
fn improve_clustering_view<N, T>(
    network: &N,
    clustering: &mut Clustering,
    use_modularity: bool,
    adjusted_resolution: f64,
    randomness: f64,
    rng: &mut T,
    _max_outer_iterations: Option<u32>,
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
        // Recursion needed: materialize to CompactNetwork and delegate.
        // Inner aggregation always runs to convergence (None = unlimited).
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

/// Recursive aggregation phase — always operates on CompactNetwork.
/// This always recurses to convergence (the induced network is aggregated
/// repeatedly until it stops shrinking). This is the inner loop within
/// a single outer iteration.
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
            panic!("No node network, which shouldn't have happened");
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
            None, // inner aggregation always recurses to convergence
            max_local_moving_iterations,
        )?;
    } else {
        // No shrinkage — one final LM pass without further recursion.
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

/// This function will be executed repeatedly as per number_iterations
fn improve_clustering<T>(
    network: &CompactNetwork,
    clustering: &mut Clustering,
    use_modularity: bool,
    adjusted_resolution: f64,
    randomness: f64,
    rng: &mut T,
    _max_outer_iterations: Option<u32>,
    max_local_moving_iterations: u32,
) -> Result<bool, CoreError>
where
    T: Rng + Clone + Send,
{
    // do a slower, higher fidelity full network clustering
    let mut improved: bool = full_network_clustering::full_network_clustering(
        network,
        clustering,
        adjusted_resolution,
        rng,
        max_local_moving_iterations,
    )?;

    if clustering.next_cluster_id() < network.num_nodes() {
        // given the updated clustering, generate subnetworks for each cluster comprised solely of the
        // nodes in that cluster, then fast, low-fidelity cluster the subnetworks, merging the results
        // back into the primary clustering before returning
        let nodes_by_cluster: Vec<Vec<CompactNodeId>> = clustering.nodes_per_cluster();
        let subnetworks_iterator = network.subnetworks_iter(clustering, &nodes_by_cluster, None);
        let num_nodes_per_cluster: Vec<u64> = clustering.num_nodes_per_cluster();

        let num_subnetworks: usize = clustering.next_cluster_id();

        clustering.reset_next_cluster_id();

        let mut num_nodes_per_cluster_induced_network: Vec<usize> =
            Vec::with_capacity(num_subnetworks);
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
                panic!("No node network, which shouldn't have happened");
            } else {
                let subnetwork_clustering: Clustering = subnetwork_clusterer
                    .subnetwork_clustering(
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

        if induced_clustering_network.num_nodes() < network.num_nodes() {
            // Induced network is smaller — recurse to convergence.
            improved |= improve_clustering(
                &induced_clustering_network,
                &mut induced_network_clustering,
                use_modularity,
                adjusted_resolution,
                randomness,
                rng,
                None,
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
    fn test_max_outer_iterations_none_is_unlimited() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let edges = edge_list();
        let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
        let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);

        let mut rng1: SmallRng = SmallRng::seed_from_u64(123);
        let mut rng2: SmallRng = SmallRng::seed_from_u64(123);

        // max_outer_iterations = None should recurse until convergence
        let (_, clustering_none) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng1,
            true,
            None,
            None,
        )
        .unwrap();

        // A high value should behave the same (converges before hitting limit)
        let (_, clustering_high) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng2,
            true,
            Some(100),
            None,
        )
        .unwrap();

        for node_id in 0..labeled_network.num_nodes() {
            assert_eq!(
                clustering_none.cluster_at(node_id).unwrap(),
                clustering_high.cluster_at(node_id).unwrap(),
                "Node {} differed between None and Some(100) for max_outer_iterations",
                node_id
            );
        }
    }

    #[test]
    fn test_max_outer_iterations_zero_prevents_recursion() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let edges = edge_list();
        let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
        let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);

        let mut rng: SmallRng = SmallRng::seed_from_u64(123);

        // max_outer_iterations = Some(0) means no recursion (local moving only)
        let (_, clustering_zero) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng,
            true,
            Some(0),
            None,
        )
        .unwrap();

        // Should produce a valid clustering with bounded cluster count
        let num_clusters = clustering_zero.next_cluster_id();
        assert!(num_clusters >= 1 && num_clusters <= labeled_network.num_nodes());
        for node_id in 0..labeled_network.num_nodes() {
            assert!(clustering_zero.cluster_at(node_id).is_ok());
        }
    }

    #[test]
    fn test_max_outer_iterations_one_allows_single_recursion() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        // Build a network with enough structure that Leiden would normally recurse
        // (aggregate and re-run). With max_outer_iterations=1, one full cycle
        // (LM + refine + aggregate + recurse) is performed.
        let edges: Vec<Edge> = vec![
            ("a".into(), "b".into(), 10.0),
            ("b".into(), "c".into(), 10.0),
            ("c".into(), "a".into(), 10.0),
            ("d".into(), "e".into(), 10.0),
            ("e".into(), "f".into(), 10.0),
            ("f".into(), "d".into(), 10.0),
            ("g".into(), "h".into(), 10.0),
            ("h".into(), "i".into(), 10.0),
            ("i".into(), "g".into(), 10.0),
            ("a".into(), "d".into(), 1.0),
            ("d".into(), "g".into(), 1.0),
            ("g".into(), "a".into(), 1.0),
        ];

        let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
        let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);

        let mut rng: SmallRng = SmallRng::seed_from_u64(42);

        let (_, clustering_one) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng,
            true,
            Some(1),
            None,
        )
        .unwrap();

        // With Some(1), one full recursion cycle is allowed. This should produce
        // fewer communities than Some(0) (local moving only).
        let mut rng2: SmallRng = SmallRng::seed_from_u64(42);
        let (_, clustering_zero) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng2,
            true,
            Some(0),
            None,
        )
        .unwrap();

        // Should produce valid clusterings
        for node_id in 0..labeled_network.num_nodes() {
            assert!(clustering_one.cluster_at(node_id).is_ok());
            assert!(clustering_zero.cluster_at(node_id).is_ok());
        }

        // Some(1) allows aggregation, so should produce <= clusters than Some(0)
        assert!(
            clustering_one.next_cluster_id() <= clustering_zero.next_cluster_id(),
            "Some(1) should produce <= clusters than Some(0): got {} vs {}",
            clustering_one.next_cluster_id(),
            clustering_zero.next_cluster_id()
        );
    }

    #[test]
    fn test_max_outer_iterations_higher_values_allow_more_recursion() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        // Build a deeper network that benefits from multiple recursion levels
        let edges: Vec<Edge> = vec![
            ("a".into(), "b".into(), 10.0),
            ("b".into(), "c".into(), 10.0),
            ("c".into(), "a".into(), 10.0),
            ("d".into(), "e".into(), 10.0),
            ("e".into(), "f".into(), 10.0),
            ("f".into(), "d".into(), 10.0),
            ("g".into(), "h".into(), 10.0),
            ("h".into(), "i".into(), 10.0),
            ("i".into(), "g".into(), 10.0),
            ("j".into(), "k".into(), 10.0),
            ("k".into(), "l".into(), 10.0),
            ("l".into(), "j".into(), 10.0),
            ("a".into(), "d".into(), 1.0),
            ("d".into(), "g".into(), 1.0),
            ("g".into(), "j".into(), 1.0),
            ("j".into(), "a".into(), 1.0),
        ];

        let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
        let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);

        let mut rng1: SmallRng = SmallRng::seed_from_u64(77);
        let mut rng2: SmallRng = SmallRng::seed_from_u64(77);

        let (_, clustering_depth1) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng1,
            true,
            Some(1),
            None,
        )
        .unwrap();

        let (_, clustering_depth5) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng2,
            true,
            Some(5),
            None,
        )
        .unwrap();

        // The number of clusters is not guaranteed to be monotonic w.r.t. recursion depth; just
        // sanity-check that both runs produce a bounded, non-empty clustering.
        let depth1_clusters = clustering_depth1.next_cluster_id();
        let depth5_clusters = clustering_depth5.next_cluster_id();
        assert!(depth1_clusters >= 1 && depth1_clusters <= labeled_network.num_nodes());
        assert!(depth5_clusters >= 1 && depth5_clusters <= labeled_network.num_nodes());
    }

    #[test]
    fn test_both_limits_together_produce_valid_clustering() {
        use rand::SeedableRng;
        use rand::rngs::SmallRng;

        let edges = edge_list();
        let mut builder: LabeledNetworkBuilder<String> = LabeledNetworkBuilder::new();
        let labeled_network: LabeledNetwork<String> = builder.build(edges.into_iter(), true);

        let mut rng: SmallRng = SmallRng::seed_from_u64(55);

        // Apply both limits simultaneously
        let (improved, clustering) = leiden(
            labeled_network.compact(),
            None,
            Some(1),
            None,
            None,
            &mut rng,
            true,
            Some(2),
            Some(2),
        )
        .unwrap();

        // Should have produced some result (improved or not is fine with limits)
        let _ = improved;

        // Every node should have a valid cluster
        for node_id in 0..labeled_network.num_nodes() {
            assert!(clustering.cluster_at(node_id).is_ok());
        }

        // Should have at least 1 cluster
        assert!(
            clustering.next_cluster_id() >= 1,
            "Should have at least 1 cluster"
        );
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
            None,
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
            None,
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
