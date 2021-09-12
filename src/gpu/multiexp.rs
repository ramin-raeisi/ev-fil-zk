use super::error::{GPUError, GPUResult};
use super::utils;
use crate::bls::{Engine, Bls12};
use ff::{Field, PrimeField, ScalarEngine};
use groupy::{CurveAffine, CurveProjective};
use log::{error, info, debug};
use rust_gpu_tools::*;
use std::any::TypeId;
use std::sync::{mpsc, Arc};
use futures::future::Future;
use rayon::iter::{ParallelIterator, IntoParallelRefIterator};
use crate::gpu::scheduler;
use super::super::settings;

use crate::multiexp::{multiexp_cpu}; // for cpu-based parallel computations

//const MAX_WINDOW_SIZE: usize = 10;
const LOCAL_WORK_SIZE: usize = 256;
const MEMORY_PADDING: f64 = 0.1f64;
// Let 20% of GPU memory be free
//const CPU_UTILIZATION: f64 = 0.2;
// Increase GPU memory usage via inner loop, 1 for default value
//const CHUNK_SIZE_MULTIPLIER: f64 = 2.0;

pub fn get_cpu_utilization() -> f64 {
    std::env::var("FIL_ZK_CPU_UTILIZATION")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_ZK_CPU_UTILIZATION! Defaulting to {}", settings::FILSETTINGS.lock().unwrap().cpu_utilization);
                Ok(settings::FILSETTINGS.lock().unwrap().cpu_utilization)
            }
        })
        .unwrap_or(settings::FILSETTINGS.lock().unwrap().cpu_utilization)
        .max(0f64)
        .min(1f64)
        
}

pub fn get_memory_padding() -> f64 {
    std::env::var("FIL_ZK_GPU_MEMORY_PADDING")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_ZK_GPU_MEMORY_PADDING! Defaulting to {}", MEMORY_PADDING);
                Ok(MEMORY_PADDING)
            }
        })
        .unwrap_or(MEMORY_PADDING)
        .max(0f64)
        .min(1f64)
}

pub fn get_max_window() -> usize {
    let max_window_size = settings::FILSETTINGS.lock().unwrap().max_window_size as usize;
    std::env::var("FIL_ZK_MAX_WINDOW")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_ZK_MAX_WINDOW! Defaulting to {}", max_window_size);
                Ok(max_window_size)
            }
        })
        .unwrap_or(max_window_size)
}

// Multiexp kernel for a single GPU
pub struct MultiexpKernel<E>
    where
        E: Engine,
{
    _phantom: std::marker::PhantomData<E::Fr>,
}

fn calc_num_groups(work_size: usize, num_windows: usize) -> usize {
    // Observations show that we get the best performance when work_size ~= 2 * CUDA_CORES
    work_size / num_windows
}

fn calc_window_size(n: usize, exp_bits: usize, work_size: usize) -> usize {
    // window_size = ln(n / num_groups)
    // num_windows = exp_bits / window_size
    // num_groups = work_size / num_windows = work_size * window_size / exp_bits
    // window_size = ln(n / num_groups) = ln(n * exp_bits / (work_size * window_size))
    // window_size = ln(exp_bits * n / (work_size)) - ln(window_size)
    //
    // Thus we need to solve the following equation:
    // window_size + ln(window_size) = ln(exp_bits * n / (work_size))
    let lower_bound = (((exp_bits * n) as f64) / ((work_size) as f64)).ln();
    let max_window_size = settings::FILSETTINGS.lock().unwrap().max_window_size as usize;
    for w in 0..max_window_size {
        if (w as f64) + (w as f64).ln() > lower_bound {
            info!("calculated window size: {}", w);
            return w;
        }
    }
    get_max_window()
}

fn calc_best_chunk_size(max_window_size: usize, work_size: usize, exp_bits: usize) -> usize {
    let chunk_size_multiplier: f64 = std::env::var("FIL_ZK_CHUNK_SIZE_MULTIPLIER")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_ZK_CHUNK_SIZE_MULTIPLIER! Defaulting to {}", settings::FILSETTINGS.lock().unwrap().chunk_size_multiplier);
                Ok(settings::FILSETTINGS.lock().unwrap().chunk_size_multiplier)
            }
        })
        .unwrap_or(settings::FILSETTINGS.lock().unwrap().chunk_size_multiplier);

    // Best chunk-size (N) can also be calculated using the same logic as calc_window_size:
    // n = e^window_size * window_size * work_size / exp_bits
    (((max_window_size as f64).exp() as f64) * (max_window_size as f64) * (work_size as f64) * chunk_size_multiplier
        / (exp_bits as f64))
        .ceil() as usize
}

fn calc_max_chunk_size<E>(mem: u64, work_size: usize, over_g2: bool) -> usize
    where
        E: Engine
{
    let memory_padding = get_memory_padding();
    let fil_max_window_size = settings::FILSETTINGS.lock().unwrap().max_window_size as usize;
    //let aff_size = std::cmp::max(std::mem::size_of::<E::G1Affine>(), std::mem::size_of::<E::G2Affine>());
    let aff_size =
        if over_g2 { std::mem::size_of::<E::G2Affine>() } else { std::mem::size_of::<E::G1Affine>() };
    let exp_size = exp_size::<E>();
    //let proj_size = std::mem::size_of::<E::G1>() + std::mem::size_of::<E::G2>();
    let proj_size =
        if over_g2 { std::mem::size_of::<E::G2>() } else { std::mem::size_of::<E::G1>() };
    let chunk_size = ((((mem as f64) * (1f64 - memory_padding)) as usize)
        - (work_size * ((1 << fil_max_window_size) + 1) * proj_size))
        / (2 * aff_size + exp_size);
    debug!("Memory usage by max chunks size: {}", (2 * aff_size + exp_size) * chunk_size + (work_size * ((1 << fil_max_window_size) + 1) * proj_size) );
    chunk_size
}

fn exp_size<E: Engine>() -> usize {
    std::mem::size_of::<<E::Fr as ff::PrimeField>::Repr>()
}

impl<E> MultiexpKernel<E>
    where
        E: Engine,
{
    fn ensure_curve() -> GPUResult<()> {
        if TypeId::of::<E>() == TypeId::of::<Bls12>() {
            Ok(())
        } else {
            Err(GPUError::CurveNotSupported)
        }
    }

    fn chunk_size_of(program: &opencl::Program, work_size: usize, over_g2: bool) -> usize {
        let exp_bits = exp_size::<E>() * 8;
        let max_n = calc_max_chunk_size::<E>(program.device().memory(), work_size, over_g2);
        let fil_max_window_size = settings::FILSETTINGS.lock().unwrap().max_window_size as usize;
        let best_n = calc_best_chunk_size(fil_max_window_size, work_size, exp_bits);
        if max_n < best_n {
            info!("the best chunks size > max chunk size ({} / {}). Probably, settings are wrong for this machine", best_n, max_n);
        }
        std::cmp::min(max_n, best_n)
    }

    fn multiexp_on<G>(
        program: &opencl::Program,
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        n: usize,
        work_size: usize,
        start_idx_bases: usize,
        start_idx_exps: usize,
        over_g2: bool,
    ) -> GPUResult<<G as CurveAffine>::Projective>
        where
            G: CurveAffine,
    {
        MultiexpKernel::<E>::ensure_curve()?;

        let mut work_size = work_size;
        
        let bases = &bases[start_idx_bases .. start_idx_bases + n];
        let exps = &exps[start_idx_exps .. start_idx_exps + n];

        let exp_bits = exp_size::<E>() * 8;
        let mut window_size = utils::try_get_window_size(&program.device(), over_g2);
        if window_size == 0 {
            window_size = calc_window_size(n as usize, exp_bits, work_size);
        } else { // don't use work_size_multiplier for magic constants
            work_size = work_size / (settings::FILSETTINGS.lock().unwrap().work_size_multiplier as usize);
        }

        let num_windows = ((exp_bits as f64) / (window_size as f64)).ceil() as usize;
        let num_groups = calc_num_groups(work_size, num_windows);
        let bucket_len = 1 << window_size;

        // Each group will have `num_windows` threads and as there are `num_groups` groups, there will
        // be `num_groups` * `num_windows` threads in total.
        // Each thread will use `num_groups` * `num_windows` * `bucket_len` buckets.

        let mut base_buffer = program.create_buffer::<G>(n)?;
        base_buffer.write_from(0, bases)?;
        let mut exp_buffer =
            program.create_buffer::<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>(n)?;
        exp_buffer.write_from(0, exps)?;

        let bucket_buffer =
            program.create_buffer::<<G as CurveAffine>::Projective>(work_size * bucket_len)?;
        let result_buffer = program.create_buffer::<<G as CurveAffine>::Projective>(work_size)?;

        // Make global work size divisible by `LOCAL_WORK_SIZE`
        let mut global_work_size = num_windows * num_groups;
        global_work_size +=
            (LOCAL_WORK_SIZE - (global_work_size % LOCAL_WORK_SIZE)) % LOCAL_WORK_SIZE;

        let kernel = program.create_kernel(
            if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
                "G1_bellman_multiexp"
            } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
                "G2_bellman_multiexp"
            } else {
                return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
            },
            global_work_size,
            None,
        );

        call_kernel!(
            kernel,
            &base_buffer,
            &bucket_buffer,
            &result_buffer,
            &exp_buffer,
            n as u32,
            num_groups as u32,
            num_windows as u32,
            window_size as u32
        )?;

        let mut results = vec![<G as CurveAffine>::Projective::zero(); num_groups * num_windows];
        result_buffer.read_into(0, &mut results)?;

        // Using the algorithm below, we can calculate the final result by accumulating the results
        // of those `NUM_GROUPS` * `NUM_WINDOWS` threads.
        let mut acc = <G as CurveAffine>::Projective::zero();
        let mut bits = 0;
        for i in 0..num_windows {
            let w = std::cmp::min(window_size, exp_bits - bits);
            for _ in 0..w {
                acc.double();
            }
            for g in 0..num_groups {
                acc.add_assign(&results[g * num_windows + i]);
            }
            bits += w; // Process the next window
        }

        Ok(acc)
    }

    pub fn multiexp<G>(
        bases: Arc<Vec<G>>,
        exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        skip: usize,
        n: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
        where
            G: CurveAffine,
            <G as groupy::CurveAffine>::Engine: Engine,
    {
        MultiexpKernel::<E>::ensure_curve()?;

        let mut chunk_size: usize = std::usize::MAX;

        info!("Running multiexp with n = {}", n);

        let over_g2 = if TypeId::of::<G>() == TypeId::of::<E::G1Affine>() {
            false
        } else if TypeId::of::<G>() == TypeId::of::<E::G2Affine>() {
            true
        } else {
            return Err(GPUError::Simple("Only E::G1 and E::G2 are supported!"));
        };

        // use cpu for parallel calculations
        //let mut cpu_n = ((n as f64) * get_cpu_utilization()) as usize;
        let cpu_util = get_cpu_utilization();
        let mut cpu_n = ((n as f64) * cpu_util) as usize;
        if n < 10000 {
            cpu_n = n;
        }
        let n = n - cpu_n;

        for p in scheduler::DEVICE_POOL.devices.iter() {
            let data = p.lock().unwrap();
            let mut cur = utils::try_get_chunk_size(&data.device());
            if cur == 0 {
                cur = MultiexpKernel::<E>::chunk_size_of(&data,
                    utils::best_work_size(&data
                        .device(), over_g2),
                    over_g2);
            }
            if cur < chunk_size {
                chunk_size = cur;
            }
        }

        chunk_size = std::cmp::min(chunk_size, n);

        let chunks_amount: usize = ((n as f64) / (chunk_size as f64)).ceil() as usize;
        let chunk_idxs: Vec<usize> = (0..chunks_amount).collect();

        rayon::scope(|s| {
            // concurrent computing
            let (tx_gpu, rx_gpu) = mpsc::channel();
            let (tx_cpu, rx_cpu) = mpsc::channel();

            let cpu_bases = bases.clone();
            let cpu_exps = exps.clone();

            // GPU calculations
            s.spawn(move |_| {
                let mut result = <G as CurveAffine>::Projective::zero();
                if n > 0 {
                    result = chunk_idxs.par_iter()
                    .map(|chunk_idx| {
                        let start_idx = chunk_idx * chunk_size;
                        let chunk_size = std::cmp::min(chunk_size, n - start_idx);
                        let bases = bases.clone();
                        let exps = exps.clone();

                        scheduler::schedule(move |prog| -> GPUResult<<G as CurveAffine>::Projective> {
                            MultiexpKernel::<E>::multiexp_on(
                                prog,
                                bases,
                                exps,
                                chunk_size,
                                utils::best_work_size(&prog.device(), over_g2),
                                // Bases are skipped by `self.1` elements, when converted from (Arc<Vec<G>>, usize) to Source
                                // https://github.com/zkcrypto/bellman/blob/10c5010fd9c2ca69442dc9775ea271e286e776d8/src/multiexp.rs#L38
                                start_idx + skip,
                                start_idx,
                                over_g2,
                            )
                        })
                    })
                    .map(|future| future.wait().unwrap())
                    .collect::<GPUResult<Vec<<G as CurveAffine>::Projective>>>().unwrap()
                    .iter()
                    .fold(<G as CurveAffine>::Projective::zero(), |mut a, b| {
                        a.add_assign(b);
                        a
                    });
                }

                tx_gpu.send(result).unwrap();
            });

            // CPU calculations
            s.spawn(move |_| {
                info!("CPU run multiexp over {} elements", cpu_n);

                let mut cpu_acc = <G as CurveAffine>::Projective::zero();

                if cpu_n > 0 {
                    cpu_acc = multiexp_cpu(
                        cpu_bases.clone(),
                        cpu_exps.clone(),
                        cpu_n,
                        n + skip, // use last values of the vec
                        n,
                    ).unwrap();
                }

                tx_cpu.send(cpu_acc).unwrap();
            });

            let mut acc = <G as CurveAffine>::Projective::zero();

            // waiting results...
            let gpu_r = rx_gpu.recv().unwrap();
            let cpu_r = rx_cpu.recv().unwrap();

            acc.add_assign(&gpu_r);
            acc.add_assign(&cpu_r);
            
            Ok(acc)
        })
    }

    pub fn calibrate<G>(program: &opencl::Program, n: usize) -> GPUResult<usize>
        where
            G: CurveAffine,
            <G as groupy::CurveAffine>::Engine: crate::bls::Engine,
    {
        fn n_of<F, T: Clone>(n: usize, mut f: F) -> Vec<T>
            where
                F: FnMut() -> T,
        {
            let init = (0..1024).map(|_| f()).collect::<Vec<_>>();
            init.into_iter().cycle().take(n).collect::<Vec<_>>()
        }
        use std::time::Instant;
        let rng = &mut rand::thread_rng();
        let bases = Arc::new(
            n_of(n, || {
            <G as CurveAffine>::Projective::random(rng).into_affine()
        }));
        let exps = Arc::new(
            n_of(n, || <G as CurveAffine>::Scalar::random(rng).into_repr()));
        let mut best_work_size = 128;
        let mut best_dur = None;
        loop {
            let now = Instant::now();
            let bases = bases.clone();
            let exps = exps.clone();
            Self::multiexp_on(program, bases, exps, n, best_work_size + 128, 0, 0, true)?;
            let dur = now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64;
            if best_dur.is_some() && dur > ((best_dur.unwrap() as f64) * 1.1f64) as u64 {
                break;
            } else {
                best_dur = Some(dur);
            }
            best_work_size += 128;
        }
        println!("Best: {}", best_work_size);
        Ok(best_work_size)
    }
   
}
