// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#[macro_export]
macro_rules! log {
    ($message:expr) => {{
        #[cfg(feature = "logging")]
        {
            use chrono::Local;
            println!("{}: {}", Local::now().format("%H:%M:%S%.3f"), $message);
            //println!($message);
        }
    }};
    ($fmt:expr, $($args:tt)*) => {{
        #[cfg(feature = "logging")]
        {
            use chrono::Local;
            let message = format!($fmt, $($args)*);
            println!("{}: {}", Local::now().format("%H:%M:%S%.3f"), message);
            //println!($fmt, $($args)*);
        }
    }};
}

#[macro_export]
macro_rules! progress_meter {
    ($fmt: expr, $current_work_index: expr, $total_work_length: expr) => {{
        #[cfg(feature = "logging")]
        {
            if $current_work_index == $total_work_length - 1 {
                log!($fmt, "100");
            } else {
                let ten_percent: f64 = ($total_work_length as f64 / 10_f64).ceil();
                if $current_work_index as f64 % ten_percent
                    > ($current_work_index + 1) as f64 % ten_percent
                {
                    let numerator: f64 = ($current_work_index + 1) as f64;
                    let denominator: f64 = $total_work_length as f64;

                    let decile: f64 = (numerator / denominator * 10_f64).floor() * 10_f64;
                    log!($fmt, decile);
                }
            }
        }
    }};
}
