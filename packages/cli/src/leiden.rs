// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use network_partitions::clustering::Clustering;
use network_partitions::errors::CoreError;
use network_partitions::leiden::leiden as leiden_internal;
use network_partitions::network::prelude::*;
use network_partitions::quality;

use rand::SeedableRng;
use rand_xorshift::XorShiftRng;

use std::fs::File;
use std::io::prelude::*;
use std::time::Instant;

pub fn leiden(
    source_edges: &str,
    output_path: &str,
    separator: &str,
    source_index: usize,
    target_index: usize,
    weight_index: Option<usize>,
    seed: Option<usize>,
    iterations: usize,
    resolution: f64,
    randomness: f64,
    use_modularity: bool,
    skip_first_line: bool,
) {
    let start_instant: Instant = Instant::now();
    let labeled_network: LabeledNetwork<String> = LabeledNetwork::<String>::load_from(
        source_edges,
        separator,
        source_index,
        target_index,
        weight_index,
        skip_first_line,
        use_modularity,
    )
    .expect("Something went wrong loading");

    let network: &CompactNetwork = labeled_network.compact();

    let loaded_file_instant: Instant = Instant::now();

    let mut rng: XorShiftRng = match seed {
        Some(seed) => {
            println!("Using {} for PRNG seed", seed as u64);
            XorShiftRng::seed_from_u64(seed as u64)
        }
        None => XorShiftRng::from_entropy(),
    };

    let result: Result<(bool, Clustering), CoreError> = leiden_internal(
        network,
        None,
        Some(iterations),
        Some(resolution),
        Some(randomness),
        &mut rng,
        use_modularity,
    );

    let leiden_completion_instant: Instant = Instant::now();
    match result {
        Ok((improved, clustering)) => {
            println!("Clustering improved?  {}", improved);
            let quality_score: f64 =
                quality::quality(network, &clustering, Some(resolution), use_modularity)
                    .expect("An error occurred when determining quality");
            println!("Quality score (modularity): {:?}", quality_score);
            println!("Output to {}", output_path);

            let mut output_file: File =
                File::create(output_path).expect("Unable to open output file for writing");
            for item in &clustering {
                write!(
                    output_file,
                    "{},{}\n",
                    labeled_network.label_for(item.node_id),
                    item.cluster
                )
                .expect("Could not write entry to file");
            }
        }
        Err(err) => {
            println!("An error occurred when running leiden: {:?}", err);
        }
    }

    let file_writer_instant: Instant = Instant::now();
    println!(
        "Time to load file: {:?}",
        loaded_file_instant.duration_since(start_instant)
    );
    println!(
        "Time to run leiden: {:?}",
        leiden_completion_instant.duration_since(loaded_file_instant)
    );
    println!(
        "Time to output: {:?}",
        file_writer_instant.duration_since(leiden_completion_instant)
    );
    println!(
        "Total time: {:?}",
        file_writer_instant.duration_since(start_instant)
    );
}
