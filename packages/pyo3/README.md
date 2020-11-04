# graspologic-native

`graspologic-native` is a companion library to `graspologic`.  This module is a Python native module created by using
the `network_partitions` crate from the same repository.

The purpose of this module is to provide a faster implementations of graph/network analysis algorithms in a native
without trying to work through the troubles of releasing Rust crates and Python modules at the same time (in specific 
as the Python `graspologic` module is expected to be far more active than the Rust crates or native modules are).

The only capability currently implemented by this module is the Leiden algorithm, described in the paper
[From Louvain to Leiden: guaranteeing well-connected communities](https://openaccess.leidenuniv.nl/handle/1887/78029), 
Traag, V.A.; Waltman, L.; Van, Eck N.J., Scientific Reports, Vol. 9, 2019.  In addition to the paper, the reference 
implementation provided at [https://github.com/CWTSLeiden/networkanalysis](https://github.com/CWTSLeiden/networkanalysis)
was used as a starting point.

## Releases
Builds are provided for x86_64 architectures only, for Windows, macOS, and Linux, for Python versions 3.6->3.9.

## Build Tools
Rust nightly 1.37+ (we are currently using 1.40)
The python package [maturin](https://github.com/pyo3/maturin)

Please consider using [graspologic](https://github.com/microsoft/graspologic) in lieu of `graspologic-native`, as the 
former will contain some nice wrappers to make usage of this library more pythonic.
