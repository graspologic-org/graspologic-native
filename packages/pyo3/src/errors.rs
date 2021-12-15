// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use pyo3::create_exception;
use pyo3::prelude::*;

use network_partitions::errors::CoreError;

create_exception!(
    leiden,
    ClusterIndexingError,
    pyo3::exceptions::PyRuntimeError
);
create_exception!(leiden, EmptyNetworkError, pyo3::exceptions::PyValueError);
create_exception!(
    leiden,
    InvalidCommunityMappingError,
    pyo3::exceptions::PyValueError
);
create_exception!(
    leiden,
    InternalNetworkIndexingError,
    pyo3::exceptions::PyRuntimeError
);
create_exception!(leiden, ParameterRangeError, pyo3::exceptions::PyValueError);
create_exception!(
    leiden,
    UnsafeInducementError,
    pyo3::exceptions::PyRuntimeError
);
create_exception!(leiden, QueueError, pyo3::exceptions::PyRuntimeError);

#[derive(Debug)]
pub enum PyLeidenError {
    ClusterIndexingError,
    EmptyNetworkError,
    InvalidCommunityMappingError,
    InternalNetworkIndexingError,
    ParameterRangeError,
    UnsafeInducementError,
    QueueError,
}

// mapping from our CoreError enum to the PyLeidenError enum.  The PyLeidenError
impl From<CoreError> for PyLeidenError {
    fn from(err: CoreError) -> Self {
        match err {
            CoreError::ClusterIndexingError => PyLeidenError::ClusterIndexingError,
            CoreError::EmptyNetworkError => PyLeidenError::EmptyNetworkError,
            CoreError::InternalNetworkIndexingError => PyLeidenError::InternalNetworkIndexingError,
            CoreError::ParameterRangeError => PyLeidenError::ParameterRangeError,
            CoreError::UnsafeInducementError => PyLeidenError::UnsafeInducementError,
            CoreError::QueueError => PyLeidenError::QueueError,
        }
    }
}

impl From<PyLeidenError> for PyErr {
    fn from(err: PyLeidenError) -> Self {
        match err {
            PyLeidenError::ClusterIndexingError => PyErr::new::<ClusterIndexingError, _>(format!(
                "{:?}",
                PyLeidenError::ClusterIndexingError
            )),
            PyLeidenError::EmptyNetworkError => PyErr::new::<EmptyNetworkError, _>(format!(
                "{:?}",
                PyLeidenError::EmptyNetworkError
            )),
            PyLeidenError::InvalidCommunityMappingError => {
                PyErr::new::<InvalidCommunityMappingError, _>(format!(
                    "{:?}",
                    PyLeidenError::InvalidCommunityMappingError
                ))
            }
            PyLeidenError::InternalNetworkIndexingError => {
                PyErr::new::<InternalNetworkIndexingError, _>(format!(
                    "{:?}",
                    PyLeidenError::InternalNetworkIndexingError
                ))
            }
            PyLeidenError::ParameterRangeError => PyErr::new::<ParameterRangeError, _>(format!(
                "{:?}",
                PyLeidenError::ParameterRangeError
            )),
            PyLeidenError::UnsafeInducementError => PyErr::new::<UnsafeInducementError, _>(
                format!("{:?}", PyLeidenError::UnsafeInducementError),
            ),
            PyLeidenError::QueueError => {
                PyErr::new::<QueueError, _>(format!("{:?}", PyLeidenError::QueueError))
            }
        }
    }
}
