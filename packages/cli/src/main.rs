// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use clap::{Arg, ArgAction, Command};
use std::convert::TryFrom;

mod args;
mod leiden;

use crate::args::*;

fn main() {
    let matches = Command::new("leiden_cli")
        .version("0.1.0")
        .author("Dwayne Pryce <dwpryce@microsoft.com>")
        .about("Runs leiden over a provided edge list and outputs the results")
        .arg(
            Arg::new(SOURCE_EDGES)
                .help("The edge list that defines the graph's connections")
                .required(true)
                .index(1),
        )
        .arg(
            Arg::new(OUTPUT)
                .help("The output for the communities detected")
                .required(true)
                .index(2),
        )
        .arg(
            Arg::new(SEPARATOR)
                .short('s')
                .help("The character to split the edge list on")
                .action(ArgAction::Set)
                .default_value("\t"),
        )
        .arg(
            Arg::new(SOURCE_INDEX)
                .action(ArgAction::Set)
                .help("0-based index of source column from edge file")
                .default_value("0"),
        )
        .arg(
            Arg::new(TARGET_INDEX)
                .action(ArgAction::Set)
                .help("0-based index of target column from edge file")
                .default_value("1"),
        )
        .arg(
            Arg::new(WEIGHT_INDEX)
                .action(ArgAction::Set)
                .help("0-based index of weight column from edge file")
        )
        .arg(
            Arg::new(SEED)
                .action(ArgAction::Set)
                .help("A seed value to start the PRNG")
                .long("seed"),
        )
        .arg(
            Arg::new(ITERATIONS)
                .action(ArgAction::Set)
                .help("Leiden is an inherently recursive algorithm, however it may find itself (due to randomness) at a localized maximum. Setting iterations to a number larger than 1 may allow you to jump out of a local maximum and continue until a better optimum partitioning is found (note that any n > 1 will mean that leiden will be run again for a minimum of n-1 more times, though it may be run for many more than that")
                .short('i')
                .default_value("1"),
        )
        .arg(
            Arg::new(RESOLUTION)
                .action(ArgAction::Set)
                .help("")
                .short('r')
                .default_value("1.0")
        )
        .arg(
            Arg::new(RANDOMNESS)
                .action(ArgAction::Set)
                .help("")
                .default_value("1E-2"),
        )
        .arg(
            Arg::new(QUALITY)
                .action(ArgAction::Set)
                .help("Quality function to use")
                .short('q')
                .value_parser(["modularity", "cpm"])
                .default_value("modularity"),
        )
        .arg(
            Arg::new(HAS_HEADER)
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
