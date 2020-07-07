# topologic-native
[topologic](https://github.com/microsoft/topologic) is an opinionated Python library built around a collection of the 
best network/graph embedding techniques and other network related functionality best practices. 

Some functionality can be best served if compiled into a python native module, both for performance purposes and to 
share that functionality with web assembly.

`topologic-native` is a repository that holds Rust packages. The core packages will be published as crate libraries, 
and a package using [pyo3](https://github.com/pyo3/pyo3) will expose the functionality of that library to Python.  

## Requirements
- Rust nightly 1.37+ (we are currently using 1.40)
- Python 3.5+ (we are currently using 3.8)
- 64 bit operating system

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

`topologic-native` does not collect, store, or transmit any information of any kind back to Microsoft.

For your convenience, here is the link to the general [Microsoft Privacy Statement](https://privacy.microsoft.com/en-us/privacystatement/). 
