use super::*;
use log::info;
use crate::bls::Bls12;
use rust_gpu_tools::opencl as cl;
use lazy_static::*;
use std::sync::{Mutex, atomic::AtomicUsize, atomic::Ordering::Relaxed};
use futures::{Future, lazy};
use crate::SynthesisError;
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};
use rayon_core::scope;
use rayon_futures::ScopeFutureExt;

pub struct DevicePool {
    pub devices: Vec<Mutex<cl::Program>>
}

impl DevicePool {
    pub fn new() -> Self {
        Self {
            devices: cl::Device::all().unwrap().iter().map(|d| {
                info!("Compiling kernels on device: {} (Bus-id: {})", d.name(), d.bus_id());
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

pub static mut DEVICE_NUM: AtomicUsize = AtomicUsize::new(0);

pub fn get_next_device() -> &'static Mutex<cl::Program> {
    unsafe {
        if DEVICE_NUM.load(Relaxed) >= DEVICE_POOL.devices.len() - 1 {
            DEVICE_NUM.store(0, Relaxed);
        } else {
            DEVICE_NUM.store(DEVICE_NUM.load(Relaxed) + 1, Relaxed);
        }
        return &DEVICE_POOL.devices[DEVICE_NUM.load(Relaxed)];
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