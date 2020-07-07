// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::iter;

use rand::Rng;

use crate::clustering::Clustering;
use crate::errors::CoreError;
use crate::log;
use crate::network::Network;
use crate::progress_meter;
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
    network: &Network,
    clustering: Option<Clustering>,
    iterations: Option<usize>,
    resolution: Option<f64>,
    randomness: Option<f64>,
    rng: &mut T,
    use_modularity: bool,
) -> Result<(bool, Clustering), CoreError>
where
    T: Rng + Clone + Send,
{
    let iterations: usize = iterations.unwrap_or(DEFAULT_ITERATIONS);
    let randomness: f64 = randomness.unwrap_or(subnetwork::DEFAULT_RANDOMNESS);

    let adjusted_resolution: f64 = adjust_resolution(resolution, network, use_modularity);

    if randomness <= 0_f64 || adjusted_resolution <= 0_f64 {
        return Err(CoreError::ParameterRangeError);
    } else if network.num_nodes() == 0 {
        return Err(CoreError::EmptyNetworkError);
    }

    let mut clustering: Clustering =
        clustering.unwrap_or(Clustering::as_self_clusters(network.num_nodes().clone()));
    let mut improved: bool = false;

    log!(
        "Running Leiden with the maximization function {} for {} iterations over a network with {} nodes and {} edges with a total edge weight of {} and total node weight of {}",
        if use_modularity { "modularity" } else { "cpm" },
        iterations,
        &network.num_nodes(),
        &network.num_edges(),
        &network.total_edge_weight(),
        &network.total_node_weight(),
    );
    for _i in 0..iterations {
        improved |= improve_clustering(
            network,
            &mut clustering,
            adjusted_resolution,
            randomness,
            rng,
        )?;
    }

    return Ok((improved, clustering));
}

/// This function will be executed repeatedly as per number_iterations
fn improve_clustering<T>(
    network: &Network,
    clustering: &mut Clustering,
    adjusted_resolution: f64,
    randomness: f64,
    rng: &mut T,
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
    )?;

    log!(
        "Full network clustering completed, determined there should be {} clusters for {} nodes",
        &clustering.next_cluster_id(),
        &clustering.num_nodes()
    );

    if clustering.next_cluster_id() < network.num_nodes().clone() {
        // given the updated clustering, generate subnetworks for each cluster comprised solely of the
        // nodes in that cluster, then fast, low-fidelity cluster the subnetworks, merging the results
        // back into the primary clustering before returning
        let subnetworks: Vec<Network> = network.subnetworks_for_clustering(clustering)?;
        let nodes_per_cluster: Vec<Vec<usize>> = clustering.nodes_per_cluster();
        clustering.reset_next_cluster_id();

        let num_nodes_per_cluster_induced_network: Vec<usize> =
            enrich_clustering_with_subnetwork_clusterings_serial(
                subnetworks,
                clustering,
                &nodes_per_cluster,
                adjusted_resolution,
                randomness,
                rng,
            )?;

        let induced_clustering_network: Network = network.induce_clustering_network(clustering)?;

        let mut induced_network_clustering = initial_clustering_for_induced(
            num_nodes_per_cluster_induced_network,
            induced_clustering_network.num_nodes(),
        );

        log!(
            "Induced network with {} nodes and {} edges with a total edge weight of {} and total node weight of {}",
            &induced_clustering_network.num_nodes(),
            &induced_clustering_network.num_edges(),
            &induced_clustering_network.total_edge_weight(),
            &induced_clustering_network.total_node_weight(),
        );

        improved |= improve_clustering(
            &induced_clustering_network,
            &mut induced_network_clustering,
            adjusted_resolution,
            randomness,
            rng,
        )?;
        clustering.merge_clustering(&induced_network_clustering);
    }
    return Ok(improved);
}

fn enrich_clustering_with_subnetwork_clusterings_serial<T>(
    subnetworks: Vec<Network>,
    clustering: &mut Clustering,
    nodes_per_cluster: &Vec<Vec<usize>>,
    adjusted_resolution: f64,
    randomness: f64,
    rng: &mut T,
) -> Result<Vec<usize>, CoreError>
where
    T: Rng,
{
    let mut num_nodes_per_cluster_induced_network: Vec<usize> =
        Vec::with_capacity(subnetworks.len());
    let max_subnetwork_size: usize = subnetworks
        .iter()
        .map(|subnetwork| subnetwork.num_nodes())
        .max()
        .unwrap();

    let mut subnetwork_clusterer: SubnetworkClusteringGenerator =
        SubnetworkClusteringGenerator::with_capacity(max_subnetwork_size);
    log!(
        "Generating clustering for {} subnetworks",
        subnetworks.len()
    );
    for (i, subnetwork) in subnetworks.iter().enumerate() {
        progress_meter!("{}% complete", i, subnetworks.len());
        let subnetwork_clustering: Clustering = subnetwork_clusterer.subnetwork_clustering(
            subnetwork,
            adjusted_resolution,
            randomness,
            rng,
        )?;
        num_nodes_per_cluster_induced_network.push(subnetwork_clustering.next_cluster_id());
        clustering.merge_subnetwork_clustering(&nodes_per_cluster[i], &subnetwork_clustering);
    }
    std::thread::spawn(move || drop(subnetworks));
    return Ok(num_nodes_per_cluster_induced_network);
}

// fn update_clustering_by_subnetwork_multi_threaded<T>(
//     subnetworks: Vec<Network>,
//     clustering: &mut Clustering,
//     nodes_per_cluster: &Vec<Vec<usize>>,
//     adjusted_resolution: f64,
//     randomness: f64,
//     rng: &mut T,
// ) -> Result<Vec<usize>, CoreError>
// where
//     T: Rng + Clone + Send,
// {
//     let mut subnetworks_and_rng: Vec<(Network, T)> = subnetworks
//         .into_iter()
//         .map(|subnetwork| (subnetwork, rng.clone()))
//         .collect();
//     let mut num_nodes_per_cluster_induced_network: Vec<usize> = vec![0; subnetworks_and_rng.len()];
//
//     let subnetwork_clusterings: Vec<Result<Clustering, CoreError>> = subnetworks_and_rng
//         .par_iter_mut()
//         .map(|package| {
//             let subnetwork: &Network = &package.0;
//             let rng: &mut T = &mut package.1;
//             subnetwork_clustering::old_subnetwork_clustering(
//                 subnetwork,
//                 adjusted_resolution,
//                 randomness,
//                 rng,
//             )
//         })
//         .collect();
//
//     let failure: Option<&Result<Clustering, CoreError>> = subnetwork_clusterings
//         .par_iter()
//         .find_any(|result: &&Result<Clustering, CoreError>| result.is_err());
//
//     if failure.is_some() {
//         let err: CoreError = failure.unwrap().as_ref().unwrap_err().clone();
//         return Err(err);
//     }
//
//     subnetwork_clusterings
//         .iter()
//         .map(|result| result.as_ref().unwrap())
//         .enumerate()
//         .for_each(|(i, clustering_subnetwork)| {
//             num_nodes_per_cluster_induced_network[i] = clustering_subnetwork.next_cluster_id();
//             clustering.merge_subnetwork_clustering(&nodes_per_cluster[i], clustering_subnetwork);
//         });
//
//     return Ok(num_nodes_per_cluster_induced_network);
// }

fn initial_clustering_for_induced(
    num_nodes_per_cluster_induced_network: Vec<usize>,
    num_nodes: usize,
) -> Clustering {
    // Create an initial clustering for the induced network based on the non-refined clustering
    let mut clusters_induced_network: Vec<usize> = Vec::with_capacity(num_nodes);
    for num_nodes_per_induced_cluster_index in 0..num_nodes_per_cluster_induced_network.len() {
        // fill num_nodes_per_induced_cluster_index into positions from clusters_induced_network_index to clusters_induced_network_index + num_nodes_per_cluster_reduced_network[num_nodes_per_induced_cluster_index]
        let repetitions: usize =
            num_nodes_per_cluster_induced_network[num_nodes_per_induced_cluster_index];
        clusters_induced_network
            .extend(iter::repeat(num_nodes_per_induced_cluster_index).take(repetitions));
    }
    let next_cluster_id: usize = match clusters_induced_network.last() {
        Some(largest_cluster) => largest_cluster.clone() + 1,
        None => 0,
    };
    return Clustering::as_defined(clusters_induced_network, next_cluster_id);
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
