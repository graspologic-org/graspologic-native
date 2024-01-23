# graspologic-native

[graspologic](https://github.com/microsoft/graspologic) is a python package for graph statistics. 

Some functionality can be best served if compiled into a python native module, both for performance purposes and to 
share that functionality with web assembly.

`graspologic-native` is a repository that holds Rust packages. The core packages will be published as crate libraries, 
and a package using [pyo3](https://github.com/pyo3/pyo3) will expose the functionality of that library to Python.  

## Requirements
- Rust nightly 1.37+ (we are currently using 1.40)
- Python 3.5+ (we are currently using 3.8)
- 64 bit operating system

## Published Versions
We currently build for x86_64 platforms only, Windows, macOS, and Ubuntu, for python versions 3.6 - 3.11.

## Building
If for any reason, the published wheels do not match your architecture or if you have a particularly old version of glibc that isn't sufficiently accounted for in our current build matrix, or you just want to build it yourself, the following build instructions should help.

Note that these instructions are for Linux specifically, though they also should work for MacOS. Unfortunately, the instructions for Windows are a bit more convoluted and I will comment the sections that deviate between the three, as I'm aware of issues.

Before running these instructions, ensure you have installed Rust on your system and you have the Python development headers (e.g. `python3.8-dev`) for your system.

```bash
rustup default nightly
git clone git@github.com:microsoft/graspologic-native.git
cd graspologic-native
python3.8 -m venv venv
pip install -U pip setuptools wheel
pip install maturin
cd packages/pyo3
maturin build --release -i python3.8  # this is where things break on windows.  instead of `python3.8` here, you will need the full path to the correct python.exe on your windows machine, something like `-i "C:\python38\bin\python.exe"`
```

Presuming a successful build, your output wheel should be in: `graspologic-native/target/wheels/`

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Privacy

`graspologic-native` does not collect, store, or transmit any information of any kind back to Microsoft.

For your convenience, here is the link to the general [Microsoft Privacy Statement](https://privacy.microsoft.com/en-us/privacystatement/). 
