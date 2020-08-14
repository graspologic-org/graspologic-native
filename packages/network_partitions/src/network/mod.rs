// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

pub type Edge = (String, String, f64);

pub use self::network::Network;
pub use self::network_builder::NetworkBuilder;

pub use self::compact_network::{
    ClusterId, CompactNetwork, CompactNodeId, CompactNodeItem, CompactSubnetwork,
    CompactSubnetworkItem,
};
pub use self::labeled_network::LabeledNetwork;
pub use self::networks::NetworkDetails;

mod compact_network;
mod identifier;
mod labeled_network;
mod network;
mod network_builder;
mod networks;
pub mod prelude;
