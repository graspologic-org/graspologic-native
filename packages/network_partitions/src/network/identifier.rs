// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

use std::collections::HashMap;
use std::hash::Hash;

pub fn identify<T>(
    original_to_new: &mut HashMap<T, usize>,
    new_to_original: &mut Vec<T>,
    original: T,
) -> usize
where
    T: Clone + Hash + Eq,
{
    return match original_to_new.get(&original) {
        Some(id) => *id,
        None => {
            let new_id: usize = new_to_original.len();
            original_to_new.insert(original.clone(), new_id);
            new_to_original.push(original);
            new_id
        }
    };
}
