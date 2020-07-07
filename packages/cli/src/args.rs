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

impl TryFrom<ArgMatches<'_>> for CliArgs {
    type Error = ParseCliError;

    fn try_from(matches: ArgMatches<'_>) -> Result<Self, Self::Error> {
        let source_edges = matches
            .value_of(SOURCE_EDGES)
            .ok_or(ParseCliError::RequiredValueError)?;
        let output = matches
            .value_of(OUTPUT)
            .ok_or(ParseCliError::RequiredValueError)?;
        let separator = matches
            .value_of(SEPARATOR)
            .ok_or(ParseCliError::RequiredValueError)?;
        let source_index: usize = matches.value_of(SOURCE_INDEX).as_a()?;
        let target_index: usize = matches.value_of(TARGET_INDEX).as_a()?;
        let weight_index: Option<usize> = matches.value_of(WEIGHT_INDEX).as_a()?;
        let seed: Option<usize> = matches.value_of(SEED).as_a()?;
        let iterations: usize = matches.value_of(ITERATIONS).as_a()?;
        let resolution: f64 = matches.value_of(RESOLUTION).as_a()?;
        let randomness: f64 = matches.value_of(RANDOMNESS).as_a()?;
        let quality_option: Option<&str> = matches.value_of(QUALITY);
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
        let skip_first_line: bool = matches.is_present(HAS_HEADER);
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

trait As<T> {
    fn as_a(&self) -> Result<T, ParseCliError>;
}

impl As<f64> for Option<&str> {
    fn as_a(&self) -> Result<f64, ParseCliError> {
        self.map(|cli_arg| cli_arg.parse::<f64>().unwrap())
            .ok_or(ParseCliError::RequiredValueError)
    }
}

impl As<usize> for Option<&str> {
    fn as_a(&self) -> Result<usize, ParseCliError> {
        self.map(|cli_arg| cli_arg.parse::<usize>().unwrap())
            .ok_or(ParseCliError::RequiredValueError)
    }
}

impl As<Option<usize>> for Option<&str> {
    fn as_a(&self) -> Result<Option<usize>, ParseCliError> {
        let result = match self {
            Some(cli_arg) => {
                let parse_result = cli_arg.parse::<usize>();
                Ok(parse_result.map(|value| Some(value))?)
            }
            None => Ok(None),
        };
        return result;
    }
}
