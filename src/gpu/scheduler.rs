use std::sync::Mutex;

use futures::{Future, lazy};
use lazy_static::*;
use log::info;
use rayon::iter::ParallelIterator;
use rayon::prelude::*;
use rayon_core::scope;
use rayon_futures::ScopeFutureExt;
use rust_gpu_tools::opencl as cl;

use crate::bls::Bls12;
use crate::SynthesisError;

use super::*;

pub struct DevicePool {
    pub devices: Vec<Mutex<cl::Program>>
}

impl DevicePool {
    pub fn new() -> Self {
        Self {
            devices: cl::Device::all().into_par_iter().map(|d| {
                info!("Compiling kernels on device: {} (Bus-id: {})", d.name(), d.bus_id().unwrap());
                let src = sources::kernel::<Bls12>(d.brand() == cl::Brand::Nvidia);
                let program = cl::Program::from_opencl(d.clone(), &src).unwrap();
                Mutex::<cl::Program>::new(program)
            }).collect::<Vec<_>>()
        }
    }
}

lazy_static! {
    pub static ref DEVICE_POOL: DevicePool = DevicePool::new();
}

pub static mut DEVICE_NUM: usize = 0;

pub fn get_next_device() -> &'static Mutex<cl::Program> {
    unsafe {
        DEVICE_NUM = DEVICE_NUM + 1 % DEVICE_POOL.devices.len();
        return &DEVICE_POOL.devices[DEVICE_NUM];
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