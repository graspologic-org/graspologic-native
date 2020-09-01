// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#![allow(unused_imports)]

mod errors;
mod mediator;

use std::collections::{HashMap, HashSet};

use pyo3::exceptions::{TypeError, ValueError};
use pyo3::prelude::*;
use pyo3::{create_exception, wrap_pyfunction, wrap_pymodule};

use pyo3::type_object::initialize_type;
use pyo3::types::{PyDict, PyInt, PyList, PyString, PyTuple};

use network_partitions::clustering::Clustering;
use network_partitions::errors::CoreError;
use network_partitions::leiden::leiden as leiden_internal;
use network_partitions::log;
use network_partitions::network::prelude::*;
use network_partitions::quality;

use network_partitions::safe_vectors::SafeVectors;

use errors::*;

#[pyclass]
pub struct HierarchicalCluster {
    #[pyo3(get)]
    node: String,
    #[pyo3(get)]
    cluster: ClusterId,
    #[pyo3(get)]
    level: u32,
    #[pyo3(get)]
    parent_cluster: Option<ClusterId>,
    #[pyo3(get)]
    is_final_cluster: bool
}

#[pyfunction(
    "*",
    resolution = "1.0",
    randomness = "0.001",
    iterations = "1",
    use_modularity = "true"
)]
#[text_signature = "(edges, /starting_communities, resolution, randomness, iterations, use_modularity, seed)"]
/// Leiden is a community detection algorithm based on the Louvain algorithm.
///
/// :param edges: A list of edges.
/// :type edges: List[Tuple[str, str, float]]
/// :param starting_communities: An optional initial mapping of nodes to their community. Note that
///     this function does require that all nodes in the edge list have a community and nodes in the
///     community dictionary exist as a node in the provided edge list. The community values must
///     also be a non negative number.
/// :type starting_communities: Optional[Dict[str, int]]
/// :param float resolution: Default is `1.0`. Higher resolution values lead to more communities and
///     lower resolution values leads to fewer communities. Must be greater than 0.
/// :param float randomness: Default is `0.001`. The larger the randomness value, the more
///     exploration of the partition space is possible. This is a major difference from the Louvain
///     algorithm. The Louvain algorithm is purely greedy in the partition exploration.
/// :param int iterations: Default is `1`. The leiden algorithm is recursive, but subject to pseudo-random
///     number generators which sometimes lead to suboptimal community membership. Setting a number
///     greater than 1 will force leiden to run at minimum `iterations - 1` more times seeking a
///     more optimal partitioning.
/// :param bool use_modularity: Default is `True`. Whether to use modularity or CPM as the
///     maximization function.
/// :param Optional[int] seed: Default is `None`. If provided, the seed will be used in creating the
///     Pseudo-Random Number Generator at a known state, making runs over the same network and
///     starting_communities with the same parameters end with the same results.
/// :return: A dictionary of node to community ids. The community ids will start at 0 and increment.
/// :rtype: Dict[str, int]
/// :raises ClusterIndexingError:
/// :raises EmptyNetworkError:
/// :raises InvalidCommunityMappingError: The starting communities dictionary either did not include
///     a community for each node in the edge list or the edge list did not contain a node present
///     in the starting community mapping.
/// :raises InternalNetworkIndexingError: An internal algorithm error. Please report with reproduction steps.
/// :raises ParameterRangeError: One of the parameters provided did not meet the requirements in the documentation.
/// :raises UnsafeInducementError: An internal algorithm error. Please report with reproduction steps.
fn leiden(
    py: Python,
    edges: Vec<Edge>,
    starting_communities: Option<HashMap<String, usize>>,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    seed: Option<u64>,
) -> PyResult<(bool, f64, HashMap<String, usize>)> {
    #[cfg(feature = "logging")]
    use std::time::Instant;
    #[cfg(feature = "logging")]
    let now: Instant = Instant::now();

    log!("pyo3 converted {} edges from Python's representation to a Vec<(String, String, f64)> representation at {:?}", edges.len(), now);

    let result: Result<(bool, f64, HashMap<String, usize>), PyLeidenError> =
        py.allow_threads(move || {
            mediator::leiden(
                edges,
                starting_communities,
                resolution,
                randomness,
                iterations,
                use_modularity,
                seed,
            )
        });
    return result.map_err(|err| PyErr::from(err));
}

#[pyfunction(
    "*",
    resolution = "1.0",
    randomness = "0.001",
    iterations = "1",
    use_modularity = "true",
    max_cluster_size = "1000",
)]
#[text_signature = "(edges, /starting_communities, resolution, randomness, iterations, use_modularity, max_cluster_size, seed)"]
/// Leiden is a community detection algorithm based on the Louvain algorithm.
///
/// :param edges: A list of edges.
/// :type edges: List[Tuple[str, str, float]]
/// :param starting_communities: An optional initial mapping of nodes to their community. Note that
///     this function does require that all nodes in the edge list have a community and nodes in the
///     community dictionary exist as a node in the provided edge list. The community values must
///     also be a non negative number.
/// :type starting_communities: Optional[Dict[str, int]]
/// :param float resolution: Default is `1.0`. Higher resolution values lead to more communities and
///     lower resolution values leads to fewer communities. Must be greater than 0.
/// :param float randomness: Default is `0.001`. The larger the randomness value, the more
///     exploration of the partition space is possible. This is a major difference from the Louvain
///     algorithm. The Louvain algorithm is purely greedy in the partition exploration.
/// :param int iterations: Default is `1`. The leiden algorithm is recursive, but subject to pseudo-random
///     number generators which sometimes lead to suboptimal community membership. Setting a number
///     greater than 1 will force leiden to run at minimum `iterations - 1` more times seeking a
///     more optimal partitioning.
/// :param bool use_modularity: Default is `True`. Whether to use modularity or CPM as the
///     maximization function.
/// :param int max_cluster_size: Default is `1000`. Any cluster larger than 1000 will be broken into
///     an isolated subnetwork and have leiden run over it for further refinement.
/// :param Optional[int] seed: Default is `None`. If provided, the seed will be used in creating the
///     Pseudo-Random Number Generator at a known state, making runs over the same network and
///     starting_communities with the same parameters end with the same results.
/// :return: A dictionary of node to community ids. The community ids will start at 0 and increment.
/// :rtype: List[HierarchicalCluster]
/// :raises ClusterIndexingError:
/// :raises EmptyNetworkError:
/// :raises InvalidCommunityMappingError: The starting communities dictionary either did not include
///     a community for each node in the edge list or the edge list did not contain a node present
///     in the starting community mapping.
/// :raises InternalNetworkIndexingError: An internal algorithm error. Please report with reproduction steps.
/// :raises ParameterRangeError: One of the parameters provided did not meet the requirements in the documentation.
/// :raises UnsafeInducementError: An internal algorithm error. Please report with reproduction steps.
fn hierarchical_leiden(
    py: Python,
    edges: Vec<Edge>,
    starting_communities: Option<HashMap<String, usize>>,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    max_cluster_size: u32,
    seed: Option<u64>,
) -> PyResult<Vec<HierarchicalCluster>> {
    #[cfg(feature = "logging")]
    use std::time::Instant;
    #[cfg(feature = "logging")]
        let now: Instant = Instant::now();

    log!("pyo3 converted {} edges from Python's representation to a Vec<(String, String, f64)> representation at {:?}", edges.len(), now);

    let result: Result<Vec<HierarchicalCluster>, PyLeidenError> =
        py.allow_threads(move || {
            mediator::hierarchical_leiden(
                edges,
                starting_communities,
                resolution,
                randomness,
                iterations,
                use_modularity,
                max_cluster_size,
                seed,
            )
        });
    return result.map_err(|err| PyErr::from(err));
}

#[pyfunction("*", resolution = "1.0")]
#[text_signature = "(edges, communities, /resolution)"]
fn modularity(
    py: Python,
    edges: Vec<Edge>,
    communities: HashMap<String, usize>,
    resolution: f64,
) -> PyResult<f64> {
    let result: Result<f64, PyLeidenError> =
        py.allow_threads(move || mediator::modularity(edges, communities, resolution));

    return result.map_err(|err| PyErr::from(err));
}

/// topologic_native currently supports global network partitioning via the Leiden University
/// algorithm described by https://arxiv.org/abs/1810.08473
#[pymodule]
fn topologic_native(
    py: Python,
    module: &PyModule,
) -> PyResult<()> {
    module.add_class::<HierarchicalCluster>()?;
    module.add_wrapped(wrap_pyfunction!(leiden))?;
    module.add_wrapped(wrap_pyfunction!(hierarchical_leiden))?;
    module.add_wrapped(wrap_pyfunction!(modularity))?;

    module.add(
        "ClusterIndexingError",
        py.get_type::<ClusterIndexingError>(),
    )?;
    module.add("EmptyNetworkError", py.get_type::<EmptyNetworkError>())?;
    module.add(
        "InvalidCommunityMappingError",
        py.get_type::<InvalidCommunityMappingError>(),
    )?;
    module.add(
        "InternalNetworkIndexingError",
        py.get_type::<InternalNetworkIndexingError>(),
    )?;
    module.add("ParameterRangeError", py.get_type::<ParameterRangeError>())?;
    module.add(
        "UnsafeInducementError",
        py.get_type::<UnsafeInducementError>(),
    )?;
    module.add("QueueError", py.get_type::<QueueError>())?;
    Ok(())
}
