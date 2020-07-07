// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#![feature(in_band_lifetimes)]
use clap::{App, Arg};
use std::convert::TryFrom;

mod args;
mod leiden;

use crate::args::*;

fn main() {
    let matches = App::new("leiden_cli")
        .version("0.1.0")
        .author("Dwayne Pryce <dwpryce@microsoft.com>")
        .about("Runs leiden over a provided edge list and outputs the results")
        .arg(
            Arg::with_name(SOURCE_EDGES)
                .help("The edge list that defines the graph's connections")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::with_name(OUTPUT)
                .help("The output for the communities detected")
                .required(true)
                .index(2),
        )
        .arg(
            Arg::with_name(SEPARATOR)
                .short("s")
                .help("The character to split the edge list on")
                .takes_value(true)
                .default_value("\t"),
        )
        .arg(
            Arg::with_name(SOURCE_INDEX)
                .takes_value(true)
                .help("0-based index of source column from edge file")
                .default_value("0"),
        )
        .arg(
            Arg::with_name(TARGET_INDEX)
                .takes_value(true)
                .help("0-based index of target column from edge file")
                .default_value("1"),
        )
        .arg(
            Arg::with_name(WEIGHT_INDEX)
                .takes_value(true)
                .help("0-based index of weight column from edge file")
        )
        .arg(
            Arg::with_name(SEED)
                .takes_value(true)
                .help("A seed value to start the PRNG")
                .long("seed"),
        )
        .arg(
            Arg::with_name(ITERATIONS)
                .takes_value(true)
                .help("Leiden is an inherently recursive algorithm, however it may find itself (due to randomness) at a localized maximum. Setting iterations to a number larger than 1 may allow you to jump out of a local maximum and continue until a better optimum partitioning is found (note that any n > 1 will mean that leiden will be run again for a minimum of n-1 more times, though it may be run for many more than that")
                .short("i")
                .default_value("1"),
        )
        .arg(
            Arg::with_name(RESOLUTION)
                .takes_value(true)
                .help("")
                .short("r")
                .default_value("1.0")
        )
        .arg(
            Arg::with_name(RANDOMNESS)
                .takes_value(true)
                .help("")
                .default_value("1E-2"),
        )
        .arg(
            Arg::with_name(QUALITY)
                .takes_value(true)
                .help("Quality function to use")
                .short("q")
                .possible_value("modularity")
                .possible_value("cpm")
                .default_value("modularity"),
        )
        .arg(
            Arg::with_name(HAS_HEADER)
                .help("Flag must be added if the source file contains a header line")
                .long("has_header")
        )
        .get_matches();

    match CliArgs::try_from(matches) {
        Ok(cli_args) => leiden::leiden(
            &cli_args.source_edges,
            &cli_args.output_path,
            &cli_args.separator,
            cli_args.source_index,
            cli_args.target_index,
            cli_args.weight_index,
            cli_args.seed,
            cli_args.iterations,
            cli_args.resolution,
            cli_args.randomness,
            cli_args.use_modularity,
            cli_args.skip_first_line,
        ),
        Err(err) => println!("{:?}", err),
    }
}
