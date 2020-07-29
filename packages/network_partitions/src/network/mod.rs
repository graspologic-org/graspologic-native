// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

pub type Edge = (String, String, f64);

pub use self::network::Network;
pub use self::network_builder::NetworkBuilder;

pub use self::compact_network::CompactNetwork;
pub use self::labeled_network::LabeledNetwork;

mod compact_network;
mod labeled_network;
mod network;
mod network_builder;
