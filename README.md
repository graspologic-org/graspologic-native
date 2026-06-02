# graspologic-native

[graspologic](https://github.com/graspologic-org/graspologic) is a Python package for graph statistics.

Some functionality is best served compiled into a native Python extension module for performance. `graspologic-native` holds the Rust implementation of the [Leiden community detection algorithm](https://arxiv.org/abs/1810.08473), exposed to Python via [PyO3](https://github.com/pyo3/pyo3) and built with [maturin](https://github.com/PyO3/maturin).

## Requirements

- Rust stable (edition 2024)
- Python 3.9+
- 64-bit operating system

## Published Versions

We build wheels for x86_64 and aarch64 on Linux, macOS (universal2), and Windows, for Python 3.9–3.13.

## Building

If the published wheels don't match your platform, or you want to build from source:

```bash
git clone git@github.com:graspologic-org/graspologic-native.git
cd graspologic-native/packages/pyo3
pip install maturin
maturin build --release
```

The output wheel will be in `target/wheels/`.

Alternatively, using [uv](https://github.com/astral-sh/uv):

```bash
cd graspologic-native
uv build packages/pyo3
```

## Contributing

This project welcomes contributions and suggestions. Please open an issue or pull request on GitHub.

## Privacy

`graspologic-native` does not collect, store, or transmit any information of any kind.
