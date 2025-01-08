// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use clap::ArgMatches;
use std::convert::TryFrom;
use std::num::{ParseFloatError, ParseIntError};

pub const SOURCE_EDGES: &str = "EDGE_LIST_FILE";
pub const OUTPUT: &str = "OUTPUT_PATH";
pub const SEPARATOR: &str = "separator";
pub const SOURCE_INDEX: &str = "source_index";
pub const TARGET_INDEX: &str = "target_index";
pub const WEIGHT_INDEX: &str = "weight_index";
pub const SEED: &str = "seed";
pub const ITERATIONS: &str = "iterations";
pub const RESOLUTION: &str = "resolution";
pub const RANDOMNESS: &str = "randomness";
pub const QUALITY: &str = "quality";
pub const HAS_HEADER: &str = "has_header";

pub struct CliArgs {
    pub source_edges: String,
    pub output_path: String,
    pub separator: String,
    pub source_index: usize,
    pub target_index: usize,
    pub weight_index: Option<usize>,
    pub seed: Option<usize>,
    pub iterations: usize,
    pub resolution: f64,
    pub randomness: f64,
    pub use_modularity: bool,
    pub skip_first_line: bool,
}

impl TryFrom<ArgMatches> for CliArgs {
    type Error = ParseCliError;

    fn try_from(matches: ArgMatches) -> Result<Self, Self::Error> {
        let source_edges: &str = matches
            .get_one(SOURCE_EDGES)
            .cloned()
            .ok_or(ParseCliError::RequiredValueError)?;
        let output: &str = matches
            .get_one(OUTPUT)
            .cloned()
            .ok_or(ParseCliError::RequiredValueError)?;
        let separator: &str = matches
            .get_one(SEPARATOR)
            .cloned()
            .ok_or(ParseCliError::RequiredValueError)?;
        let source_index: usize = *matches.get_one(SOURCE_INDEX).unwrap();
        let target_index: usize = *matches.get_one(TARGET_INDEX).unwrap();
        let weight_index: Option<usize> = matches.get_one(WEIGHT_INDEX).map(|v| *v);
        let seed: Option<usize> = matches.get_one(SEED).cloned();
        let iterations: usize = *matches.get_one(ITERATIONS).unwrap();
        let resolution: f64 = *matches.get_one(RESOLUTION).unwrap();
        let randomness: f64 = *matches.get_one(RANDOMNESS).unwrap();
        let quality_option: Option<&str> = matches.get_one(QUALITY).cloned();
        let use_modularity: bool = match quality_option {
            Some(quality_value) => {
                if quality_value == "cpm" {
                    Ok(false)
                } else if quality_value == "modularity" {
                    Ok(true)
                } else {
                    Err(ParseCliError::InvalidQualityFunctionError)
                }
            }
            None => Err(ParseCliError::RequiredValueError),
        }?;
        let skip_first_line: bool = matches.contains_id(HAS_HEADER);
        let cli_args: CliArgs = CliArgs {
            source_edges: source_edges.into(),
            output_path: output.into(),
            separator: separator.into(),
            source_index,
            target_index,
            weight_index,
            seed,
            iterations,
            resolution,
            randomness,
            use_modularity,
            skip_first_line,
        };
        return Ok(cli_args);
    }
}

#[derive(Debug)]
pub enum ParseCliError {
    RequiredValueError,
    NotANumber,
    InvalidQualityFunctionError,
}

impl From<ParseFloatError> for ParseCliError {
    fn from(_: ParseFloatError) -> Self {
        return ParseCliError::NotANumber;
    }
}

impl From<ParseIntError> for ParseCliError {
    fn from(_: ParseIntError) -> Self {
        return ParseCliError::NotANumber;
    }
}
