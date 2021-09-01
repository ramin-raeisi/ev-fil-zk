use log::{info, warn, error};
use rust_gpu_tools::*;
use std::collections::HashMap;
use std::env;
use super::super::settings;

lazy_static::lazy_static! {
    static ref CORE_COUNTS: HashMap<String, (usize, usize)> = {
        let mut core_counts : HashMap<String, (usize, usize)> = vec![
            // AMD
            ("gfx1010".to_string(), (2560, 0)),
            // This value was chosen to give (approximately) empirically best performance for a Radeon Pro VII.
            ("gfx906".to_string(), (7400, 0)),

            // NVIDIA
            ("Quadro RTX 6000".to_string(), (4608, 0)),

            ("TITAN RTX".to_string(), (4608, 0)),

            ("RTX A6000".to_string(), (10752, 134217728)),
            ("Tesla V100".to_string(), (5120, 67108864)),
            ("Tesla V100S".to_string(), (5120, 67108864)),
            ("Tesla P100".to_string(), (3584, 0)),
            ("Tesla T4".to_string(), (2560, 0)),
            ("Quadro M5000".to_string(), (2048, 0)),

            ("GeForce RTX 3060".to_string(), (3840, 0)),
            ("GeForce RTX 3060 Ti".to_string(), (4864, 0)),
            ("GeForce RTX 3070".to_string(), (5888, 0)),
            ("GeForce RTX 3070 16GB".to_string(), (5888, 0)),
            ("GeForce RTX 3080".to_string(), (8704, 33554466)),
            ("GeForce RTX 3080 20GB".to_string(), (8704, 0)),
            ("GeForce RTX 3080 Ti".to_string(), (10240, 39475842)),
            ("GeForce RTX 3090".to_string(), (10496, 67108864)),

            ("GeForce RTX 2080 Ti".to_string(), (4352, 0)),
            ("GeForce RTX 2080 SUPER".to_string(), (3072, 0)),
            ("GeForce RTX 2080".to_string(), (2944, 0)),
            ("GeForce RTX 2070 SUPER".to_string(), (2560, 0)),

            ("GeForce GTX 1080 Ti".to_string(), (3584, 0)),
            ("GeForce GTX 1080".to_string(), (2560, 0)),
            ("GeForce GTX 2060".to_string(), (1920, 0)),
            ("GeForce GTX 1660 Ti".to_string(), (1536, 0)),
            ("GeForce GTX 1060".to_string(), (1280, 0)),
            ("GeForce GTX 1650 SUPER".to_string(), (1280, 0)),
            ("GeForce GTX 1650".to_string(), (896, 0)),
        ].into_iter().collect();

        match env::var("FIL_ZK_CUSTOM_GPU").and_then(|var| {
            for card in var.split(",") {
                let splitted = card.split(":").collect::<Vec<_>>();
                if splitted.len() != 2 { panic!("Invalid FIL_ZK_CUSTOM_GPU!"); }
                let name = splitted[0].trim().to_string();
                let cores : usize = splitted[1].trim().parse().expect("Invalid FIL_ZK_CUSTOM_GPU!");
                info!("Adding \"{}\" to GPU list with {} CUDA cores.", name, cores);
                core_counts.insert(name, (cores, 0));
            }
            Ok(())
        }) { Err(_) => { }, Ok(_) => { } }

        core_counts
    };

    static ref CONST_SETTINGS: HashMap<String, (usize, (usize, usize))> = {
        let const_settings : HashMap<String, (usize, (usize, usize))> = vec![
            ("Tesla V100S".to_string(), (67108864, (12, 10))),

            //("GeForce RTX 3080".to_string(), (33554466, (11, 8))),
            ("GeForce RTX 3090".to_string(), (67108864, (12, 10))),
        ].into_iter().collect();

        const_settings
    };
}

const DEFAULT_CORE_COUNT: usize = 2560;
//const WORK_SIZE_MULTIPLIER: usize = 2;

pub fn best_work_size(d: &opencl::Device, over_g2: bool) -> usize {
    let work_size_multiplier: f64 = std::env::var("FIL_ZK_WORK_SIZE_MULTIPLIER")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_ZK_WORK_SIZE_MULTIPLIER! Defaulting to {}", settings::FILSETTINGS.lock().unwrap().work_size_multiplier);
                Ok(settings::FILSETTINGS.lock().unwrap().work_size_multiplier)
            }
        })
        .unwrap_or(settings::FILSETTINGS.lock().unwrap().work_size_multiplier);

    // points from G2 have bigger size
    if over_g2 {
        return (get_core_count(d) as f64 * work_size_multiplier) as usize;
    }
    // (g2_size + exp_size) / (g1_size + exp_size) = 1.75
    // ((get_core_count(d) as f64) * (work_size_multiplier as f64) * 1.85f64) as usize

    (get_core_count(d) as f64 * work_size_multiplier * 2f64) as usize
}

pub fn try_get_chunk_size(d: &opencl::Device) -> usize {
    get_params(d).0
}

pub fn try_get_window_size(d: &opencl::Device, over_g2: bool) -> usize {
    if over_g2 {
        return get_params(d).1.1;
    }

    get_params(d).1.0
}

fn get_params(d: &opencl::Device) -> (usize, (usize, usize)) {
    let name = d.name();
    match CONST_SETTINGS.get(&name[..]) {
        Some(&params) => params,
        None => {
            (0, (0, 0))
        }
    } 
}

pub fn get_core_count(d: &opencl::Device) -> usize {
    let name = d.name();
    match CORE_COUNTS.get(&name[..]) {
        Some(&cores) => cores.0,
        None => {
            warn!(
                "Number of CUDA cores for your device ({}) is unknown! Best performance is \
                 only achieved when the number of CUDA cores is known! You can find the \
                 instructions on how to support custom GPUs here: \
                 https://lotu.sh/en+hardware-mining",
                name
            );
            DEFAULT_CORE_COUNT
        }
    }
}