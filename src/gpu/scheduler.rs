use std::sync::Mutex;
use std::fmt::Write;

use futures::{Future, lazy};
use lazy_static::*;
use log::info;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use rayon_core::scope;
use rayon_futures::ScopeFutureExt;
use rust_gpu_tools::opencl as cl;
use sha2::{Digest, Sha256};

use crate::bls::Bls12;
use crate::SynthesisError;

use super::*;

pub struct DevicePool {
    pub devices: Vec<Mutex<cl::Program>>
}

pub fn cache_path(device: &cl::Device, cl_source: &str) -> std::io::Result<std::path::PathBuf> {
    let path = dirs::home_dir().unwrap().join(".rust-gpu-tools");
    if !std::path::Path::exists(&path) {
        std::fs::create_dir(&path)?;
    }
    let mut hasher = Sha256::new();
    // If there are multiple devices with the same name and neither has a Bus-Id,
    // then there will be a collision. Bus-Id can be missing in the case of an Apple
    // GPU. For now, we assume that in the unlikely event of a collision, the same
    // cache can be used.
    // TODO: We might be able to get around this issue by using cl_vendor_id instead of Bus-Id.
    hasher.update(device.name().as_bytes());
    if let Some(bus_id) = device.bus_id() {
        hasher.update(bus_id.to_be_bytes());
    }
    hasher.update(cl_source.as_bytes());
    let mut digest = String::new();
    for &byte in hasher.finalize()[..].iter() {
        write!(&mut digest, "{:x}", byte).unwrap();
    }
    write!(&mut digest, ".bin").unwrap();

    Ok(path.join(digest))
}

impl DevicePool {
    pub fn new() -> Self {
        Self {
            devices: cl::Device::all().into_par_iter().map(|d| {
                info!("Compiling kernels on device: {} (Bus-id: {})", d.name(), d.bus_id().unwrap());
                let src = sources::kernel::<Bls12>(d.brand() == cl::Brand::Nvidia);
                let program = cl::Program::from_opencl(d.clone(), &src).unwrap_or_else(|_| {
                    let cached = cache_path(&d, &src).unwrap();
                    let bin = std::fs::read(cached).unwrap();
                    cl::Program::from_binary(d.clone(), bin).unwrap()
                });
                Mutex::<cl::Program>::new(program)
            }).collect::<Vec<_>>()
        }
    }
}

lazy_static! {
    pub static ref DEVICE_POOL: DevicePool = DevicePool::new();
    pub static ref DEVICE_POOL_FIL_PROOFS: DevicePool = DevicePool::new();
}

pub static mut DEVICE_NUM: usize = 0;
pub static mut DEVICE_NUM_FIL_PROOFS: usize = 0;

pub fn get_next_device() -> &'static Mutex<cl::Program> {
    unsafe {
        DEVICE_NUM = (DEVICE_NUM + 1) % DEVICE_POOL.devices.len();
        return &DEVICE_POOL.devices[DEVICE_NUM];
    }
}

pub fn get_next_device_second_pool() -> &'static Mutex<cl::Program> {
    unsafe {
        DEVICE_NUM_FIL_PROOFS = (DEVICE_NUM_FIL_PROOFS + 1) % DEVICE_POOL_FIL_PROOFS.devices.len();
        return &DEVICE_POOL_FIL_PROOFS.devices[DEVICE_NUM_FIL_PROOFS];
    }
}

pub fn schedule<F, T>(f: F) -> Box<dyn Future<Item=T, Error=SynthesisError> + Send>
    where
        F: FnOnce(&cl::Program) -> T + Send + 'static,
        T: Send + 'static {
    scope(|s| {
        Box::new(s.spawn_future(lazy(move || Ok::<_, SynthesisError>({
            let program = get_next_device().lock().unwrap();
            f(&program)
        }))))
    })
}
