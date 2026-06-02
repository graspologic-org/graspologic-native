// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

pub type Edge = (String, String, f64);

// pub use self::network::Network;
// pub use self::network_builder::NetworkBuilder;

pub use self::compact_network::{
    ClusterId, CompactNeighborViewIterator, CompactNetwork, CompactNodeId, CompactNodeItem,
    CompactSubnetworkItem,
};
pub use self::csr_view::{CsrNetworkView, CsrValidationError};
pub use self::identifier::Identifier;
pub use self::labeled_network::{LabeledNetwork, LabeledNetworkBuilder};
pub use self::network_view::{Neighbor, NetworkView};

mod compact_network;
pub mod csr_view;
mod identifier;
mod labeled_network;
pub mod network_view;
pub mod prelude;
