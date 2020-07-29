// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//use super::CompactNetwork;
use std::collections::HashMap;

pub struct LabeledNetwork {
    //network_structure: CompactNetwork,
    labels_to_id: HashMap<String, usize>,
    id_to_labels: Vec<String>,
}
