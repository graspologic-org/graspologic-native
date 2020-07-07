// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[cfg(test)]
mod tests {
    use network_partitions::errors::NetworkError;
    use network_partitions::network::NetworkBuilder;
    use std::collections::HashSet;

    #[test]
    fn test_load_network_from_file() {
        let org_network_path = "tests/simple_org_graph.csv";
        let broken_network_path = "tests/broken_org_graph.csv";
        let builder = NetworkBuilder::for_modularity()
            .load_from_file(org_network_path, ",", 0, 1, Some(2), false)
            .expect("Could not load from file");
        assert_eq!(10, builder.num_nodes());
        assert_eq!(15.0, builder.get_edge_weight("david", "amber").unwrap());

        let builder = NetworkBuilder::builder(true);
        let result: Result<NetworkBuilder, NetworkError> =
            builder.load_from_file(broken_network_path, ",", 0, 1, Some(2), false);
        assert!(result.is_err());
        match result.err() {
            Some(NetworkError::EdgeFileFormatError) => assert!(true),
            Some(err) => assert!(
                false,
                "Actual NetworkError returned was not EdgeFileFormatError but an {:?}",
                err
            ),
            _ => assert!(
                false,
                "Somehow this file was parsed correctly, which is certainly wrong"
            ),
        }
    }

    #[test]
    fn test_network_builder() {
        let network_path = "tests/sbm_network.csv";
        let builder = NetworkBuilder::for_modularity()
            .load_from_file(network_path, ",", 0, 1, Some(2), false)
            .expect("Could not load from file");

        let cloned_builder = builder.clone();

        let network = builder.build();

        assert_eq!(cloned_builder.index_to_node.len(), network.num_nodes());
        assert_eq!(
            if cloned_builder.total_edge_weight_self_links == 0_f64 {
                1_f64
            } else {
                cloned_builder.total_edge_weight_self_links
            },
            network.total_edge_weight_self_links()
        );

        for i in 0..network.num_nodes() {
            // check node weight
            assert_eq!(
                cloned_builder.node_weights[i],
                network
                    .node_weight_at(i)
                    .expect("node weight to exist here")
            );
            // generate neighbor list to check, but also check against the weights
            let (start_range, end_range) = network.neighbor_range(i).expect("neighbor range");
            let mut network_neighbors: HashSet<usize> = HashSet::new();
            for neighbor_index in start_range..end_range {
                let neighbor = network
                    .neighbor_at(neighbor_index)
                    .expect("Expected a neighbor index");
                network_neighbors.insert(neighbor);

                let weight = network
                    .weight_at(neighbor_index)
                    .expect("expected a weight");
                assert_eq!(cloned_builder.edges.get(&(i, neighbor)), Some(&weight))
            }
            assert_eq!(cloned_builder.node_neighbors[i], network_neighbors);
        }
    }
}
