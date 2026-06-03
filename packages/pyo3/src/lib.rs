// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

mod errors;
mod mediator;
mod scipy_csr;

use std::collections::HashMap;

use numpy::PyReadonlyArray1;
use pyo3::PyTypeInfo;
use pyo3::prelude::*;

use network_partitions::network::prelude::*;

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
    is_final_cluster: bool,
}

#[pymethods]
impl HierarchicalCluster {
    fn __repr__(&self) -> PyResult<String> {
        let parent: String = self
            .parent_cluster
            .map(|level| level.to_string())
            .unwrap_or("None".into());
        Ok(format!(
            "HierarchicalCluster(node=\"{}\", cluster=\"{}\", level={}, parent_cluster={}, is_final_cluster={})",
            self.node, self.cluster, self.level, parent, self.is_final_cluster,
        ))
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }
}

/// Leiden is a global network partitioning algorithm. Given a list of edges and a maximization
/// function, it will iterate through the network attempting to find an optimal partitioning of
/// the entire network.
///
/// :param edges: A list of edges, defined with the source and target encoded as strings and the edge weight being a float.
/// :type edges: List[Tuple[str, str, float]]
/// :param starting_communities: An optional initial mapping of nodes to their community. Note that
///     this function does require that all nodes in the edge list have a community and nodes in the
///     community dictionary exist as a node in the provided edge list. The community values must
///     also be a non negative integer.
/// :type starting_communities: Optional[Dict[str, int]]
/// :param float resolution: Default is `1.0`. Higher resolution values lead to more communities and
///     lower resolution values leads to fewer communities. Must be greater than 0.
/// :param float randomness: Default is `0.001`. The larger the randomness value, the more
///     exploration of the partition space is possible. This is a major difference from the Louvain
///     algorithm. The Louvain algorithm is purely greedy in the partition exploration.
/// :param int iterations: Default is `1`. How many times to run the full Leiden cycle
///     (local moving → refinement → recursive aggregation) on the original network. Each
///     iteration uses the previous result as its starting clustering, giving the algorithm
///     additional chances to escape suboptimal partitions. This is distinct from ``trials``,
///     which runs independent attempts and keeps the best result.
/// :param bool use_modularity: Default is `True`. Whether to use modularity or CPM as the
///     maximization function.
/// :param Optional[int] seed: Default is `None`. If provided, the seed will be used in creating the
///     Pseudo-Random Number Generator at a known state, making runs over the same network and
///     starting_communities with the same parameters end with the same results.
/// :param int trials: Default is `1`. Number of independent Leiden runs. Each trial starts
///     from scratch (or from ``starting_communities`` if provided) and the result with the
///     highest quality score is returned.
/// :param Optional[int] max_local_moving_iterations: Default is `None`. When set, limits the
///     number of sweeps through the local-moving work queue. One sweep is defined as processing
///     ``N`` node-pop operations from the queue, where ``N`` is the number of nodes in the
///     network. A value of 1 therefore caps local moving at ``N`` queue pops for that phase.
///     When ``None`` or 0, local moving continues until convergence (default behavior).
/// :return: The quality score of the best community partitioning and a dictionary of node to
///     community ids. The community ids will start at 0 and increment.
/// :rtype: Tuple[float, Dict[str, int]]
/// :raises ClusterIndexingError:
/// :raises EmptyNetworkError:
/// :raises InternalNetworkIndexingError: An internal algorithm error. Please report with reproduction steps.
/// :raises ParameterRangeError: One of the parameters provided did not meet the requirements in the documentation.
/// :raises UnsafeInducementError: An internal algorithm error. Please report with reproduction steps.
#[pyfunction]
#[pyo3(signature=(/, edges, starting_communities=None, resolution=1.0, randomness=0.001, iterations=1, use_modularity=true, seed=None, trials=1, max_local_moving_iterations=None))]
fn leiden(
    py: Python,
    edges: Vec<Edge>,
    starting_communities: Option<HashMap<String, usize>>,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    seed: Option<u64>,
    trials: u64,
    max_local_moving_iterations: Option<u32>,
) -> PyResult<(f64, HashMap<String, usize>)> {
    let result: Result<(f64, HashMap<String, usize>), PyLeidenError> = py.detach(move || {
        mediator::leiden(
            edges,
            starting_communities,
            resolution,
            randomness,
            iterations,
            use_modularity,
            seed,
            trials,
            max_local_moving_iterations,
        )
    });
    result.map_err(PyErr::from)
}

/// Hierarchical leiden builds upon the leiden function by further breaking down exceptionally large clusters.
///
/// The process followed is to run leiden the first time, then each cluster with membership
/// counts >= `max_cluster_size` (default 1000) are isolated and turned into a subnetwork, which
/// then has the leiden process run over it. The resulting clusters are merged into the original
/// clustering as new clusters, the original cluster ID will be left without any nodes belonging to
/// it.  This is done for each subnetwork, and it is done iteratively until no cluster contains more
/// than `max_cluster_size` entries.
///
/// The results are different from the regular hierarchical leiden as well.  A List of the `HierarchicalCluster`
/// items is returned.  This HierarchicalCluster describes a node->cluster relationship, by level, and also contains
/// a link back to the parent/previous cluster, and a flag denoting whether it is the final clustering
/// for a given node or not.
///
/// This hierarchical structure allows us to navigate our clusterings by breaking down truly large
/// clusters into smaller, fine grained clusters, but still be able to see the larger clustered structure.
///
/// :param edges: A list of edges, defined with the source and target encoded as strings and the edge weight being a float.
/// :type edges: List[Tuple[str, str, float]]
/// :param starting_communities: An optional initial mapping of nodes to their community. Note that
///     this function does require that all nodes in the edge list have a community and nodes in the
///     community dictionary exist as a node in the provided edge list. The community values must
///     also be a non negative integer.
/// :type starting_communities: Optional[Dict[str, int]]
/// :param float resolution: Default is `1.0`. Higher resolution values lead to more communities and
///     lower resolution values leads to fewer communities. Must be greater than 0.
/// :param float randomness: Default is `0.001`. The larger the randomness value, the more
///     exploration of the partition space is possible. This is a major difference from the Louvain
///     algorithm. The Louvain algorithm is purely greedy in the partition exploration.
/// :param int iterations: Default is `1`. How many times to run the full Leiden cycle
///     (local moving → refinement → recursive aggregation) on the original network. Each
///     iteration uses the previous result as its starting clustering, giving the algorithm
///     additional chances to escape suboptimal partitions.
/// :param bool use_modularity: Default is `True`. Whether to use modularity or CPM as the
///     maximization function.
/// :param int max_cluster_size: Default is `1000`. Any cluster larger than 1000 will be broken into
///     an isolated subnetwork and have leiden run over it for further refinement.
/// :param Optional[int] seed: Default is `None`. If provided, the seed will be used in creating the
///     Pseudo-Random Number Generator at a known state, making runs over the same network and
///     starting_communities with the same parameters end with the same results.
/// :param Optional[int] max_local_moving_iterations: Default is `None`. When set, limits the
///     number of sweeps through the local-moving work queue. One sweep is defined as processing
///     `N` node-pop operations from the queue, where `N` is the number of nodes in the network.
///     A value of 1 therefore caps local moving at `N` queue pops for that phase. When `None` or 0,
///     local moving continues until convergence (default behavior).
/// :return: A list of HierarchicalCluster entries. A hierarchical cluster contains a node id, the
///     cluster id, the level, an optional parent, and whether or not it is the final entry for that
///     node.
/// :rtype: List[HierarchicalCluster]
/// :raises ClusterIndexingError:
/// :raises EmptyNetworkError:
/// :raises InternalNetworkIndexingError: An internal algorithm error. Please report with reproduction steps.
/// :raises ParameterRangeError: One of the parameters provided did not meet the requirements in the documentation.
/// :raises UnsafeInducementError: An internal algorithm error. Please report with reproduction steps.
#[pyfunction]
#[pyo3(signature=(/, edges, starting_communities=None, resolution=1.0, randomness=0.001, iterations=1, use_modularity=true, max_cluster_size=1000, seed=None, max_local_moving_iterations=None))]
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
    max_local_moving_iterations: Option<u32>,
) -> PyResult<Vec<HierarchicalCluster>> {
    let result: Result<Vec<HierarchicalCluster>, PyLeidenError> = py.detach(move || {
        mediator::hierarchical_leiden(
            edges,
            starting_communities,
            resolution,
            randomness,
            iterations,
            use_modularity,
            max_cluster_size,
            seed,
            max_local_moving_iterations,
        )
    });
    result.map_err(PyErr::from)
}

/// Measures the modularity for a global partitioning of a network described by a list of edges.
///
/// :param edges: A list of edges, defined with the source and target encoded as strings and the edge weight being a float.
/// :type edges: List[Tuple[str, str, float]]
/// :param communities: An optional initial mapping of nodes to their community. Note that
///     this function does require that all nodes in the edge list have a community and nodes in the
///     community dictionary exist as a node in the provided edge list. The community values must
///     also be a non-negative number.
/// :type communities: Dict[str, int]
/// :param float resolution: Default is `1.0`. Higher resolution values lead to more communities and
///     lower resolution values leads to fewer communities. Must be greater than 0.
/// :return: The modularity of the community partitioning provided for the network.
/// :rtype: float
#[pyfunction]
#[pyo3(signature=(/, edges, communities, resolution=1.0))]
fn modularity(
    py: Python,
    edges: Vec<Edge>,
    communities: HashMap<String, usize>,
    resolution: f64,
) -> PyResult<f64> {
    let result: Result<f64, PyLeidenError> =
        py.detach(move || mediator::modularity(edges, communities, resolution));

    result.map_err(PyErr::from)
}

/// Run Leiden community detection directly on a scipy.sparse.csr_matrix.
///
/// This function accepts the raw CSR components (indptr, indices, data) from a scipy sparse
/// matrix and runs the Leiden algorithm with zero-copy access to the input graph. The initial
/// local moving phase operates directly on the borrowed numpy memory; recursive aggregation
/// and quality scoring internally materialize a compact representation.
///
/// The input matrix must represent an undirected graph (symmetric adjacency matrix).
/// Node IDs are integer indices from 0 to n_nodes-1.
///
/// .. warning::
///     The input arrays must not be mutated from another thread while this function is
///     running. The GIL is released during computation for performance; concurrent mutation
///     of the input arrays constitutes undefined behavior.
///
/// :param indptr: The index pointer array from the CSR matrix (int64, length n_nodes+1).
/// :type indptr: numpy.ndarray[numpy.int64]
/// :param indices: The column indices array from the CSR matrix (int32).
/// :type indices: numpy.ndarray[numpy.int32]
/// :param data: The edge weight data array from the CSR matrix (float64).
/// :type data: numpy.ndarray[numpy.float64]
/// :param int n_nodes: The number of nodes in the graph.
/// :param float resolution: Default is `1.0`. Higher resolution values lead to more communities
///     and lower resolution values leads to fewer communities. Must be greater than 0.
/// :param float randomness: Default is `0.001`. The larger the randomness value, the more
///     exploration of the partition space is possible.
/// :param int iterations: Default is `1`. How many times to run the full Leiden cycle on the
///     original network. Each iteration uses the previous clustering as its starting point.
/// :param bool use_modularity: Default is `True`. Whether to use modularity or CPM.
/// :param Optional[int] seed: Default is `None`. Random seed for reproducibility.
/// :param int trials: Default is `1`. Number of independent runs, returning the best result.
/// :param Optional[int] max_local_moving_iterations: Default is `None`. When set, limits the
///     number of sweeps through the local-moving work queue. When ``None`` or 0, local moving
///     continues until convergence.
/// :return: The quality score and a dictionary mapping node ID (int) to community ID (int).
/// :rtype: Tuple[float, Dict[int, int]]
/// :raises ParameterRangeError: If CSR validation fails or parameters are out of range.
#[pyfunction]
#[pyo3(signature=(/, indptr, indices, data, n_nodes, resolution=1.0, randomness=0.001, iterations=1, use_modularity=true, seed=None, trials=1, max_local_moving_iterations=None))]
fn leiden_csr<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<'py, i64>,
    indices: PyReadonlyArray1<'py, i32>,
    data: PyReadonlyArray1<'py, f64>,
    n_nodes: usize,
    resolution: f64,
    randomness: f64,
    iterations: usize,
    use_modularity: bool,
    seed: Option<u64>,
    trials: u64,
    max_local_moving_iterations: Option<u32>,
) -> PyResult<(f64, HashMap<usize, usize>)> {
    let indptr_slice = indptr.as_slice()?;
    let indices_slice = indices.as_slice()?;
    let data_slice = data.as_slice()?;

    // Release the GIL for the compute phase.
    // Safety: The numpy array slices are borrowed immutably for the duration of this call.
    // Under CPython's GIL, no other Python thread can execute while we hold the GIL to
    // extract slices, and once we release it via detach(), no Python code in this thread
    // can mutate the arrays. A concurrent thread *could* theoretically acquire the GIL and
    // mutate the underlying buffers, but this would require the caller to deliberately share
    // mutable references to the input arrays across threads — which is unsound usage on the
    // caller's part. Callers must not mutate the input arrays from another thread while
    // leiden_csr is running.
    let result = py.detach(move || {
        mediator::leiden_csr(
            indptr_slice,
            indices_slice,
            data_slice,
            n_nodes,
            resolution,
            randomness,
            iterations,
            use_modularity,
            seed,
            trials,
            max_local_moving_iterations,
        )
    });
    result.map_err(PyErr::from)
}

/// graspologic_native currently supports global network partitioning via the Leiden University
/// algorithm described by https://arxiv.org/abs/1810.08473
#[pymodule]
fn graspologic_native(
    py: Python<'_>,
    module: &Bound<'_, PyModule>,
) -> PyResult<()> {
    module.add_class::<HierarchicalCluster>()?;
    module.add_wrapped(wrap_pyfunction!(leiden))?;
    module.add_wrapped(wrap_pyfunction!(leiden_csr))?;
    module.add_wrapped(wrap_pyfunction!(hierarchical_leiden))?;
    module.add_wrapped(wrap_pyfunction!(modularity))?;

    module.add(
        "ClusterIndexingError",
        ClusterIndexingError::type_object(py),
    )?;
    module.add("EmptyNetworkError", EmptyNetworkError::type_object(py))?;
    module.add(
        "InvalidCommunityMappingError",
        InvalidCommunityMappingError::type_object(py),
    )?;
    module.add(
        "InternalNetworkIndexingError",
        InternalNetworkIndexingError::type_object(py),
    )?;
    module.add("ParameterRangeError", ParameterRangeError::type_object(py))?;
    module.add(
        "UnsafeInducementError",
        UnsafeInducementError::type_object(py),
    )?;
    module.add("QueueError", QueueError::type_object(py))?;
    Ok(())
}
