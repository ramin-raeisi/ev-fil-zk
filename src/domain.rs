//! This module contains an [`EvaluationDomain`] abstraction for performing
//! various kinds of polynomial arithmetic on top of the scalar field.
//!
//! In pairing-based SNARKs like [Groth16], we need to calculate a quotient
//! polynomial over a target polynomial with roots at distinct points associated
//! with each constraint of the constraint system. In order to be efficient, we
//! choose these roots to be the powers of a 2<sup>n</sup> root of unity in the
//! field. This allows us to perform polynomial operations in O(n) by performing
//! an O(n log n) FFT over such a domain.
//!
//! [`EvaluationDomain`]: crate::domain::EvaluationDomain
//! [Groth16]: https://eprint.iacr.org/2016/260

use crate::bls::Engine;
use ff::{Field, PrimeField, ScalarEngine};
use groupy::CurveProjective;

use super::SynthesisError;

use crate::gpu;

use rayon::current_num_threads;
use rayon::slice::{ParallelSliceMut, ParallelSlice};
use rayon::iter::{ParallelIterator, IndexedParallelIterator, IntoParallelRefMutIterator};

use log::{warn, error};

fn get_disble_gpu() -> usize {
    std::env::var("FIL_ZK_DISABLE_FFT_GPU")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_ZK_DISABLE_FFT_GPU! Defaulting to 0...");
                Ok(0)
            }
        })
        .unwrap_or(0)
}

fn get_inplace_fft() -> bool {
    let res = std::env::var("FIL_ZK_INPLACE_FFT")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_ZK_INPLACE_FFT! Defaulting to 0...");
                Ok(0)
            }
        })
        .unwrap_or(0);
    res != 0
}

pub struct EvaluationDomain<E: ScalarEngine, G: Group<E>> {
    coeffs: Vec<G>,
    exp: u32,
    omega: E::Fr,
    omegainv: E::Fr,
    geninv: E::Fr,
    minv: E::Fr,
}

impl<E: ScalarEngine, G: Group<E>> AsRef<[G]> for EvaluationDomain<E, G> {
    fn as_ref(&self) -> &[G] {
        &self.coeffs
    }
}

impl<E: ScalarEngine, G: Group<E>> AsMut<[G]> for EvaluationDomain<E, G> {
    fn as_mut(&mut self) -> &mut [G] {
        &mut self.coeffs
    }
}

impl<E: Engine, G: Group<E>> EvaluationDomain<E, G> {
    pub fn into_coeffs(self) -> Vec<G> {
        self.coeffs
    }

    pub fn from_coeffs(mut coeffs: Vec<G>) -> Result<EvaluationDomain<E, G>, SynthesisError> {
        // Compute the size of our evaluation domain
        let mut m = 1;
        let mut exp = 0;
        while m < coeffs.len() {
            m *= 2;
            exp += 1;

            // The pairing-friendly curve may not be able to support
            // large enough (radix2) evaluation domains.
            if exp >= E::Fr::S {
                return Err(SynthesisError::PolynomialDegreeTooLarge);
            }
        }
        // Compute omega, the 2^exp primitive root of unity
        let mut omega = E::Fr::root_of_unity();
        for _ in exp..E::Fr::S {
            omega.square();
        }

        // Extend the coeffs vector with zeroes if necessary
        coeffs.resize(m, G::group_zero());

        Ok(EvaluationDomain {
            coeffs,
            exp,
            omega,
            omegainv: omega.inverse().unwrap(),
            geninv: E::Fr::multiplicative_generator().inverse().unwrap(),
            minv: E::Fr::from_str(&format!("{}", m))
                .unwrap()
                .inverse()
                .unwrap(),
        })
    }

    pub fn fft(&mut self, devices: Option<&gpu::DevicePool>) -> gpu::GPUResult<()> {
        best_fft(devices, &mut self.coeffs, &self.omega, self.exp)?;
        Ok(())
    }


    pub fn mul_all(&mut self, val: E::Fr) {
        let chunk_size = if self.coeffs.len() < current_num_threads() {
            1
        } else {
            self.coeffs.len() / current_num_threads()
        };

        self.coeffs.par_chunks_mut(chunk_size).for_each(|chunk| {
            for v in chunk.iter_mut() {
                v.group_mul_assign(&val);
            }
        });
    }

    pub fn ifft(&mut self, devices: Option<&gpu::DevicePool>) -> gpu::GPUResult<()> {
        best_fft(devices, &mut self.coeffs, &self.omegainv, self.exp)?;

        let chunk_size = if self.coeffs.len() < current_num_threads() {
            1
        } else {
            self.coeffs.len() / current_num_threads()
        };

        let minv = self.minv;

        self.coeffs.par_chunks_mut(chunk_size).for_each(|chunk| {
            for v in chunk.iter_mut() {
                v.group_mul_assign(&minv);
            }
        });

        Ok(())
    }

    pub fn distribute_powers(&mut self, g: E::Fr, devices: Option<&gpu::DevicePool>) -> gpu::GPUResult<()> {
        if get_disble_gpu() == 0 {
            if let Some(ref _devices) = devices {
                gpu_distribute_powers(&mut self.coeffs, &g, self.exp)?;
                return Ok(());
            }
        }
        let chunk_size = if self.coeffs.len() < current_num_threads() {
            1
        } else {
            self.coeffs.len() / current_num_threads()
        };

        self.coeffs.par_chunks_mut(chunk_size).enumerate().for_each(|(i, chunk)| {
            let mut u = g.pow(&[(i * chunk_size) as u64]);
            for v in chunk.iter_mut() {
                v.group_mul_assign(&u);
                u.mul_assign(&g);
            }
        });
        Ok(())
    }

    pub fn coset_fft(&mut self, devices: Option<&gpu::DevicePool>) -> gpu::GPUResult<()> {
        self.distribute_powers(E::Fr::multiplicative_generator(), devices)?;
        self.fft(devices)?;
        Ok(())
    }

    pub fn icoset_fft(&mut self, devices: Option<&gpu::DevicePool>) -> gpu::GPUResult<()> {
        let geninv = self.geninv;
        self.ifft(devices)?;
        self.distribute_powers(geninv, devices)?;
        Ok(())
    }



    /// This evaluates t(tau) for this domain, which is
    /// tau^m - 1 for these radix-2 domains.
    pub fn z(&self, tau: &E::Fr) -> E::Fr {
        let mut tmp = tau.pow(&[self.coeffs.len() as u64]);
        tmp.sub_assign(&E::Fr::one());

        tmp
    }

    /// The target polynomial is the zero polynomial in our
    /// evaluation domain, so we must perform division over
    /// a coset.
    pub fn divide_by_z_on_coset(&mut self) {
        let i = self
            .z(&E::Fr::multiplicative_generator())
            .inverse()
            .unwrap();
        let chunk_size = if self.coeffs.len() < current_num_threads() {
            1
        } else {
            self.coeffs.len() / current_num_threads()
        };

        self.coeffs.par_chunks_mut(chunk_size).for_each(|chunk| {
            for v in chunk {
                v.group_mul_assign(&i);
            }
        });
    }

    /// Perform O(n) multiplication of two polynomials in the domain.
    pub fn mul_assign(&mut self, other: &EvaluationDomain<E, Scalar<E>>, devices: Option<&gpu::DevicePool>) -> gpu::GPUResult<()> {
        assert_eq!(self.coeffs.len(), other.coeffs.len());
        if get_disble_gpu() == 0 {
            if let Some(ref _devices) = devices {
                let n = self.coeffs.len();
                gpu_mul(&mut self.coeffs, &other.coeffs, n)?;
                return Ok(());
            }
        }
        let chunk_size = if self.coeffs.len() < current_num_threads() {
            1
        } else {
            self.coeffs.len() / current_num_threads()
        };

        self.coeffs.par_chunks_mut(chunk_size)
            .zip(other.coeffs.par_chunks(chunk_size))
            .for_each(|(chunk, other_chunk)| {
                for (a, b) in chunk.iter_mut().zip(other_chunk.iter()) {
                    a.group_mul_assign(&b.0);
                }
            });
        Ok(())
    }

    /// Perform O(n) subtraction of one polynomial from another in the domain.
    pub fn sub_assign(&mut self, other: &EvaluationDomain<E, G>,
                      devices: Option<&gpu::DevicePool>) -> gpu::GPUResult<()> {
        let len = self.coeffs.len();
        assert_eq!(len, other.coeffs.len());

        if get_disble_gpu() == 0 {
            if let Some(ref _devices) = devices {
                gpu_sub(&mut self.coeffs, &other.coeffs, len)?;
                return Ok(());
            }
        }
        let chunk_size = if len < current_num_threads() {
            1
        } else {
            len / current_num_threads()
        };

        self.coeffs.par_chunks_mut(chunk_size)
            .zip(other.coeffs.par_chunks(chunk_size))
            .for_each(|(chunk, other_chunk)| {
                for (a, b) in chunk.iter_mut().zip(other_chunk.iter()) {
                    a.group_sub_assign(&b);
                }
            });
        Ok(())
    }
}

pub trait Group<E: ScalarEngine>: Sized + Copy + Clone + Send + Sync {
    fn group_zero() -> Self;
    fn group_mul_assign(&mut self, by: &E::Fr);
    fn group_add_assign(&mut self, other: &Self);
    fn group_sub_assign(&mut self, other: &Self);
}

pub struct Point<G: CurveProjective>(pub G);

impl<G: CurveProjective> PartialEq for Point<G> {
    fn eq(&self, other: &Point<G>) -> bool {
        self.0 == other.0
    }
}

impl<G: CurveProjective> Copy for Point<G> {}

impl<G: CurveProjective> Clone for Point<G> {
    fn clone(&self) -> Point<G> {
        *self
    }
}

impl<G: CurveProjective> Group<G::Engine> for Point<G> {
    fn group_zero() -> Self {
        Point(G::zero())
    }
    fn group_mul_assign(&mut self, by: &G::Scalar) {
        self.0.mul_assign(by.into_repr());
    }
    fn group_add_assign(&mut self, other: &Self) {
        self.0.add_assign(&other.0);
    }
    fn group_sub_assign(&mut self, other: &Self) {
        self.0.sub_assign(&other.0);
    }
}

pub struct Scalar<E: ScalarEngine>(pub E::Fr);

impl<E: ScalarEngine> PartialEq for Scalar<E> {
    fn eq(&self, other: &Scalar<E>) -> bool {
        self.0 == other.0
    }
}

impl<E: ScalarEngine> Copy for Scalar<E> {}

impl<E: ScalarEngine> Clone for Scalar<E> {
    fn clone(&self) -> Scalar<E> {
        *self
    }
}

impl<E: ScalarEngine> Group<E> for Scalar<E> {
    fn group_zero() -> Self {
        Scalar(E::Fr::zero())
    }
    fn group_mul_assign(&mut self, by: &E::Fr) {
        self.0.mul_assign(by);
    }
    fn group_add_assign(&mut self, other: &Self) {
        self.0.add_assign(&other.0);
    }
    fn group_sub_assign(&mut self, other: &Self) {
        self.0.sub_assign(&other.0);
    }
}

fn log2_floor(num: usize) -> u32 {
    assert!(num > 0);

    let mut pow = 0;

    while (1 << (pow + 1)) <= num {
        pow += 1;
    }

    pow
}

fn best_fft<E: Engine, T: Group<E>>(
    devices: Option<&gpu::DevicePool>,
    a: &mut [T],
    omega: &E::Fr,
    log_n: u32,
) -> gpu::GPUResult<()> {
    let disable_gpu: usize = get_disble_gpu();

    if 1u32 << log_n < 2 << 18 {
        warn!("FFT elements amount is small (<= 2^18). GPU data transfer may probably take \
            longer than perfoming FFT on CPU. Consider disabling GPU FFT with environment \
            variable FIL_ZK_DISABLE_FFT_GPU=1");
    }

    if disable_gpu == 0 {
        if let Some(ref _devices) = devices {
            match gpu_fft(a, omega, log_n) {
                Ok(_) => {
                    return Ok(());
                }
                Err(e) => {
                    error!("GPU FFT failed! Error: {}", e);
                }
            }
        }
    }

    let log_cpus = log2_floor(rayon::current_num_threads());
    if log_n <= log_cpus {
        serial_fft(a, omega, log_n);
    } else {
        parallel_fft(a, omega, log_n, log_cpus);
    }

    Ok(())
}

pub fn gpu_fft<E: Engine, T: Group<E>>(
    a: &mut [T],
    omega: &E::Fr,
    log_n: u32,
) -> gpu::GPUResult<()> {
    // EvaluationDomain module is supposed to work only with E::Fr elements, and not CurveProjective
    // points. The Bellman authors have implemented an unnecessarry abstraction called Group<E>
    // which is implemented for both PrimeField and CurveProjective elements. As nowhere in the code
    // is the CurveProjective version used, T and E::Fr are guaranteed to be equal and thus have same
    // size.
    // For compatibility/performance reasons we decided to transmute the array to the desired type
    // as it seems safe and needs less modifications in the current structure of Bellman library.
    let a = unsafe { std::mem::transmute::<&mut [T], &mut [E::Fr]>(a) };
    if get_inplace_fft() {
        gpu::FFTKernel::<E>::inplace_fft(a, omega, log_n)?;
    } else {
        gpu::FFTKernel::<E>::radix_fft(a, omega, log_n)?;
    }
    Ok(())
}



pub fn gpu_mul<E: Engine, T: Group<E>>(
    a: &mut [T],
    b: &[Scalar<E>],
    n: usize,
) -> gpu::GPUResult<()> {
    // The reason of unsafety is same as above.
    let a = unsafe { std::mem::transmute::<&mut [T], &mut [E::Fr]>(a) };
    let b = unsafe { std::mem::transmute::<&[Scalar<E>], &[E::Fr]>(b) };
    let mut chunk_size = n;
    if get_inplace_fft() {
        chunk_size = chunk_size / 2;
    }

    for (a_chunk, b_chunk) in a.chunks_mut(chunk_size)
        .zip(b.chunks(chunk_size)) {
            gpu::FFTKernel::<E>::mul_sub(a_chunk, b_chunk, chunk_size, false)?;
    }
    Ok(())
}

pub fn gpu_sub<E: Engine, T: Group<E>>(
    a: &mut [T],
    b: &[T],
    n: usize,
) -> gpu::GPUResult<()> {
    // The reason of unsafety is same as above.
    let a = unsafe { std::mem::transmute::<&mut [T], &mut [E::Fr]>(a) };
    let b = unsafe { std::mem::transmute::<&[T], &[E::Fr]>(b) };
    let mut chunk_size = n;
    if get_inplace_fft() {
        chunk_size = chunk_size / 2;
    }

    for (a_chunk, b_chunk) in a.chunks_mut(chunk_size)
        .zip(b.chunks(chunk_size)) {
            gpu::FFTKernel::<E>::mul_sub(a_chunk, b_chunk, chunk_size, true)?;
    }
    Ok(())
}

pub fn gpu_distribute_powers<E: Engine, T: Group<E>>(
    a: &mut [T],
    g: &E::Fr,
    log_n: u32,
) -> gpu::GPUResult<()> {
    // The reason of unsafety is same as above.
    let a = unsafe { std::mem::transmute::<&mut [T], &mut [E::Fr]>(a) };
    gpu::FFTKernel::<E>::distribute_powers(a, g, log_n)?;
    Ok(())
}

#[allow(clippy::many_single_char_names)]
pub fn serial_fft<E: ScalarEngine, T: Group<E>>(a: &mut [T], omega: &E::Fr, log_n: u32) {
    fn bitreverse(mut n: u32, l: u32) -> u32 {
        let mut r = 0;
        for _ in 0..l {
            r = (r << 1) | (n & 1);
            n >>= 1;
        }
        r
    }

    let n = a.len() as u32;
    assert_eq!(n, 1 << log_n);

    for k in 0..n {
        let rk = bitreverse(k, log_n);
        if k < rk {
            a.swap(rk as usize, k as usize);
        }
    }

    let mut m = 1;
    for _ in 0..log_n {
        let w_m = omega.pow(&[u64::from(n / (2 * m))]);

        let mut k = 0;
        while k < n {
            let mut w = E::Fr::one();
            for j in 0..m {
                let mut t = a[(k + j + m) as usize];
                t.group_mul_assign(&w);
                let mut tmp = a[(k + j) as usize];
                tmp.group_sub_assign(&t);
                a[(k + j + m) as usize] = tmp;
                a[(k + j) as usize].group_add_assign(&t);
                w.mul_assign(&w_m);
            }

            k += 2 * m;
        }

        m *= 2;
    }
}

fn parallel_fft<E: ScalarEngine, T: Group<E>>(
    a: &mut [T],
    omega: &E::Fr,
    log_n: u32,
    log_cpus: u32,
) {
    assert!(log_n >= log_cpus);

    let num_cpus = 1 << log_cpus;
    let log_new_n = log_n - log_cpus;
    let mut tmp = vec![vec![T::group_zero(); 1 << log_new_n]; num_cpus];
    let new_omega = omega.pow(&[num_cpus as u64]);
    tmp.par_iter_mut().enumerate().for_each(|(j, tmp)| {
        let a = &*a;
        // Shuffle into a sub-FFT
        let omega_j = omega.pow(&[j as u64]);
        let omega_step = omega.pow(&[(j as u64) << log_new_n]);

        let mut elt = E::Fr::one();
        for (i, tmp) in tmp.iter_mut().enumerate() {
            for s in 0..num_cpus {
                let idx = (i + (s << log_new_n)) % (1 << log_n);
                let mut t = a[idx];
                t.group_mul_assign(&elt);
                tmp.group_add_assign(&t);
                elt.mul_assign(&omega_step);
            }
            elt.mul_assign(&omega_j);
        }

        // Perform sub-FFT
        serial_fft(tmp, &new_omega, log_new_n);
    });

    let chunk_size = if a.len() < current_num_threads() {
        1
    } else {
        a.len() / current_num_threads()
    };

    // TODO: does this hurt or help?
    a.par_chunks_mut(chunk_size).enumerate().for_each(|(idx, a)| {
        let tmp = &tmp;

        let mut idx = idx * chunk_size;
        let mask = (1 << log_cpus) - 1;
        for a in a {
            *a = tmp[idx & mask][idx >> log_cpus];
            idx += 1;
        }
    });
}

// Test multiplying various (low degree) polynomials together and
// comparing with naive evaluations.
#[cfg(feature = "pairing")]
#[test]
fn polynomial_arith() {
    use crate::bls::{Bls12, Engine};
    use rand_core::RngCore;

    fn test_mul<E: ScalarEngine + Engine, R: RngCore>(rng: &mut R) {
        for coeffs_a in 0..70 {
            for coeffs_b in 0..70 {
                let mut a: Vec<_> = (0..coeffs_a)
                    .map(|_| Scalar::<E>(E::Fr::random(rng)))
                    .collect();
                let mut b: Vec<_> = (0..coeffs_b)
                    .map(|_| Scalar::<E>(E::Fr::random(rng)))
                    .collect();

                // naive evaluation
                let mut naive = vec![Scalar(E::Fr::zero()); coeffs_a + coeffs_b];
                for (i1, a) in a.iter().enumerate() {
                    for (i2, b) in b.iter().enumerate() {
                        let mut prod = *a;
                        prod.group_mul_assign(&b.0);
                        naive[i1 + i2].group_add_assign(&prod);
                    }
                }

                a.resize(coeffs_a + coeffs_b, Scalar(E::Fr::zero()));
                b.resize(coeffs_a + coeffs_b, Scalar(E::Fr::zero()));

                let mut a = EvaluationDomain::from_coeffs(a).unwrap();
                let mut b = EvaluationDomain::from_coeffs(b).unwrap();

                a.fft(None);
                b.fft(None);
                a.mul_assign(b);
                a.ifft(None);

                for (naive, fft) in naive.iter().zip(a.coeffs.iter()) {
                    assert!(naive == fft);
                }
            }
        }
    }

    let rng = &mut rand::thread_rng();

    test_mul::<Bls12, _>(rng);
}

#[cfg(feature = "pairing")]
#[test]
fn fft_composition() {
    use paired::bls12_381::Bls12;
    use rand_core::RngCore;

    fn test_comp<E: ScalarEngine, R: RngCore>(rng: &mut R) {
        for coeffs in 0..10 {
            let coeffs = 1 << coeffs;

            let mut v = vec![];
            for _ in 0..coeffs {
                v.push(Scalar::<E>(E::Fr::random(rng)));
            }

            let mut domain = EvaluationDomain::from_coeffs(v.clone()).unwrap();
            domain.ifft(None);
            domain.fft(None);
            assert!(v == domain.coeffs);
            domain.fft(None);
            domain.ifft(None);
            assert!(v == domain.coeffs);
            domain.icoset_fft(None);
            domain.coset_fft(None);
            assert!(v == domain.coeffs);
            domain.coset_fft(None);
            domain.icoset_fft(None);
            assert!(v == domain.coeffs);
        }
    }

    let rng = &mut rand::thread_rng();

    test_comp::<Bls12, _>(rng);
}

#[cfg(feature = "pairing")]
#[test]
fn parallel_fft_consistency() {
    use paired::bls12_381::Bls12;
    use rand_core::RngCore;
    use std::cmp::min;

    fn test_consistency<E: ScalarEngine, R: RngCore>(rng: &mut R) {
        for _ in 0..5 {
            for log_d in 0..10 {
                let d = 1 << log_d;

                let v1 = (0..d)
                    .map(|_| Scalar::<E>(E::Fr::random(rng)))
                    .collect::<Vec<_>>();
                let mut v1 = EvaluationDomain::from_coeffs(v1).unwrap();
                let mut v2 = EvaluationDomain::from_coeffs(v1.coeffs.clone()).unwrap();

                for log_cpus in log_d..min(log_d + 1, 3) {
                    parallel_fft(&mut v1.coeffs, &v1.omega, log_d, log_cpus);
                    serial_fft(&mut v2.coeffs, &v2.omega, log_d);

                    assert!(v1.coeffs == v2.coeffs);
                }
            }
        }
    }

    let rng = &mut rand::thread_rng();

    test_consistency::<Bls12, _>(rng);
}

#[cfg(feature = "gpu")]
#[cfg(test)]
mod tests {
    use crate::domain::{gpu_fft, parallel_fft, serial_fft, EvaluationDomain, Scalar, log2_floor};
    use ff::Field;

    #[test]
    pub fn gpu_fft_consistency() {
        let _ = env_logger::try_init();

        use crate::bls::{Bls12, Fr};
        use std::time::Instant;
        let rng = &mut rand::thread_rng();

        env_logger::init();
        let log_cpus = log2_floor(rayon::current_num_threads());

        for log_d in 1..=20 {
            let d = 1 << log_d;

            let elems = (0..d)
                .map(|_| Scalar::<Bls12>(Fr::random(rng)))
                .collect::<Vec<_>>();
            let mut v1 = EvaluationDomain::from_coeffs(elems.clone()).unwrap();
            let mut v2 = EvaluationDomain::from_coeffs(elems.clone()).unwrap();

            println!("Testing FFT for {} elements...", d);

            let mut now = Instant::now();
            gpu_fft(&mut v1.coeffs, &v1.omega, log_d).expect("GPU FFT failed!");
            let gpu_dur =
                now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64;
            println!("GPU took {}ms.", gpu_dur);

            now = Instant::now();
            if log_d <= log_cpus {
                serial_fft(&mut v2.coeffs, &v2.omega, log_d);
            } else {
                parallel_fft(&mut v2.coeffs, &v2.omega, log_d, log_cpus);
            }
            let cpu_dur =
                now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64;
            println!("CPU ({} cores) took {}ms.", 1 << log_cpus, cpu_dur);

            let elems = (0..d)
                .map(|_| Scalar::<Bls12>(Fr::random(rng)))
                .collect::<Vec<_>>();
            let mut v1 = EvaluationDomain::from_coeffs(elems.clone()).unwrap();
            let mut v2 = EvaluationDomain::from_coeffs(elems.clone()).unwrap();

            println!("Testing FFT for {} elements...", d);

            let mut now = Instant::now();
            gpu_fft(&mut v1.coeffs, &v1.omega, log_d).expect("GPU FFT failed!");
            let gpu_dur =
                now.elapsed().as_secs() * 1000 as u64 + now.elapsed().subsec_millis() as u64;
            println!("GPU took {}ms.", gpu_dur);

            now = Instant::now();
            if log_d <= log_cpus {
                serial_fft(&mut v2.coeffs, &v2.omega, log_d);
            } else {
                parallel_fft(&mut v2.coeffs, &v2.omega, log_d, log_cpus);
            }
            let cpu_dur = now.elapsed().as_secs() * 1000 + now.elapsed().subsec_millis() as u64;
            println!("CPU ({} cores) took {}ms.", 1 << log_cpus, cpu_dur);

            println!("Speedup: x{}", cpu_dur as f32 / gpu_dur as f32);

            assert!(v1.coeffs == v2.coeffs);
            println!("============================");
        }
    }
}
