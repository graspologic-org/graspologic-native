// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

//! Interop implementations for third-party graph libraries.
//!
//! Each submodule is gated behind a feature flag.

#[cfg(feature = "petgraph")]
pub mod petgraph_interop;

#[cfg(feature = "sprs")]
pub mod sprs_interop;

#[cfg(test)]
mod tests {
    use crate::leiden::{leiden, leiden_view};
    use crate::network::CompactNetwork;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// Verify leiden_view produces equivalent results to leiden on the same CompactNetwork.
    #[test]
    fn test_leiden_view_vs_leiden_on_compact_network() {
        // Two triangles (weight 10) connected by a weak bridge (weight 0.01)
        // Node weights = sum of incident edge weights (modularity convention)
        let nodes = vec![
            (20.01_f64, 0_usize),
            (20.0, 2),
            (20.01, 4),
            (20.01, 7),
            (20.0, 10),
            (20.01, 12),
        ];
        let neighbors = vec![
            (1_usize, 10.0_f64),
            (2, 10.0),
            (3, 0.01), // node 0
            (0, 10.0),
            (2, 10.0), // node 1
            (0, 10.0),
            (1, 10.0),
            (3, 0.01), // node 2 -- wait, bridge is 2-3
            (2, 0.01),
            (4, 10.0),
            (5, 10.0), // node 3
            (3, 10.0),
            (5, 10.0), // node 4
            (3, 10.0),
            (4, 10.0), // node 5
        ];
        let compact = CompactNetwork::from(nodes, neighbors, 0.0);

        let mut rng1 = SmallRng::seed_from_u64(42);
        let (_improved1, c1) = leiden(
            &compact,
            None,
            Some(1),
            None,
            None,
            &mut rng1,
            true,
            None,
            None,
        )
        .unwrap();

        let mut rng2 = SmallRng::seed_from_u64(42);
        let (_improved2, c2) = leiden_view(
            &compact,
            None,
            Some(1),
            None,
            None,
            &mut rng2,
            true,
            Some(10),
            None,
        )
        .unwrap();

        // Both should find 2 communities
        assert_eq!(c1.next_cluster_id(), 2);
        assert_eq!(c2.next_cluster_id(), 2);
        // Same assignments
        for i in 0..6 {
            assert_eq!(
                c1.cluster_at(i).unwrap(),
                c2.cluster_at(i).unwrap(),
                "Node {i} cluster mismatch between leiden and leiden_view"
            );
        }
    }
}
