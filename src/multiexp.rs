use bitvec::prelude::*;
use ff::{Field, PrimeField, PrimeFieldRepr, ScalarEngine};
use groupy::{CurveAffine, CurveProjective};
use log::{info, error};
use rayon::prelude::*;
use std::io;
use std::sync::Arc;

use super::SynthesisError;
use crate::gpu;
use futures::future::{Future, lazy};
use rayon_futures::ScopeFutureExt;

/// An object that builds a source of bases.
pub trait SourceBuilder<G: CurveAffine>: Send + Sync + 'static + Clone {
    type Source: Source<G>;

    fn new(self) -> Self::Source;
    fn get(self) -> (Arc<Vec<G>>, usize);
}

/// A source of bases, like an iterator.
pub trait Source<G: CurveAffine> {
    /// Parses the element from the source. Fails if the point is at infinity.
    fn add_assign_mixed(
        &mut self,
        to: &mut <G as CurveAffine>::Projective,
    ) -> Result<(), SynthesisError>;

    /// Skips `amt` elements from the source, avoiding deserialization.
    fn skip(&mut self, amt: usize) -> Result<(), SynthesisError>;
}

impl<G: CurveAffine> SourceBuilder<G> for (Arc<Vec<G>>, usize) {
    type Source = (Arc<Vec<G>>, usize);

    fn new(self) -> (Arc<Vec<G>>, usize) {
        (self.0.clone(), self.1)
    }

    fn get(self) -> (Arc<Vec<G>>, usize) {
        (self.0, self.1)
    }
}

impl<G: CurveAffine> Source<G> for (Arc<Vec<G>>, usize) {
    fn add_assign_mixed(
        &mut self,
        to: &mut <G as CurveAffine>::Projective,
    ) -> Result<(), SynthesisError> {
        if self.0.len() <= self.1 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "expected more bases from source",
            )
                .into());
        }

        if self.0[self.1].is_zero() {
            return Err(SynthesisError::UnexpectedIdentity);
        }

        to.add_assign_mixed(&self.0[self.1]);

        self.1 += 1;

        Ok(())
    }

    fn skip(&mut self, amt: usize) -> Result<(), SynthesisError> {
        if self.0.len() <= self.1 {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "expected more bases from source",
            )
                .into());
        }

        self.1 += amt;

        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DensityTracker {
    pub bv: BitVec<Lsb0, u8>,
    pub total_density: usize,
}

impl<'a> DensityTracker {
    fn get_query_size(&self) -> Option<usize> {
        Some(self.bv.len())
    }

    pub fn new() -> DensityTracker {
        DensityTracker {
            bv: BitVec::new(),
            total_density: 0,
        }
    }

    pub fn clone(&self) -> Self {
        DensityTracker {
            bv: self.bv.clone(),
            total_density: self.total_density.clone(),
        }
    }

    pub fn add_element(&mut self) {
        self.bv.push(false);
    }

    pub fn inc(&mut self, idx: usize) {
        if !self.bv.get(idx).unwrap() {
            self.bv.set(idx, true);
            self.total_density += 1;
        }
    }

    pub fn get_total_density(&self) -> usize {
        self.total_density
    }

    /// Extend by concatenating `other`. If `is_input_density` is true, then we are tracking an input density,
    /// and other may contain a redundant input for the `One` element. Coalesce those as needed and track the result.
    pub fn extend(&mut self, other: Self, is_input_density: bool) {
        if other.bv.is_empty() {
            // Nothing to do if other is empty.
            return;
        }

        if self.bv.is_empty() {
            // If self is empty, assume other's density.
            self.total_density = other.total_density;
            self.bv = other.bv;
            return;
        }

        if is_input_density {
            // Input densities need special handling to coalesce their first inputs.

            if other.bv[0] {
                // If other's first bit is set,
                if self.bv[0] {
                    // And own first bit is set, then decrement total density so the final sum doesn't overcount.
                    self.total_density -= 1;
                } else {
                    // Otherwise, set own first bit.
                    self.bv.set(0, true);
                }
            }
            // Now discard other's first bit, having accounted for it above, and extend self by remaining bits.
            self.bv.extend(other.bv.iter().skip(1));
        } else {
            // Not an input density, just extend straightforwardly.
            self.bv.extend(other.bv);
        }

        // Since any needed adjustments to total densities have been made, just sum the totals and keep the sum.
        self.total_density += other.total_density;
    }

    pub fn extend_from_element(&mut self, other: Self, unit: &Self) {
        if other.bv.is_empty() {
            // Nothing to do if other is empty.
            return;
        }

        if self.bv.is_empty() {
            // If self is empty, assume other's density.
            self.total_density = other.total_density;
            self.bv = other.bv;
            return;
        }

        self.bv.extend(other.bv.iter().skip(unit.bv.len()));

        // Since any needed adjustments to total densities have been made, just sum the totals and keep the sum.
        self.total_density += other.total_density - unit.total_density;
    }

    pub fn deallocate(&mut self, idx: usize) {
        if *self.bv.get(idx).unwrap() {
            self.total_density -= 1;
        }
        self.bv.remove(idx);
    }

    pub fn set_var_density(&mut self, idx: usize, value: bool) {
        if value {
            self.inc(idx);
        }
        else {
            if *self.bv.get(idx).unwrap() {
                self.bv.set(idx, false);
                self.total_density -= 1;
            }
        }
    }
}

pub fn multiexp_cpu<G>(
    bases: Arc<Vec<G>>,
    exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
    n: usize,
    start_idx_bases: usize,
    start_idx_exps: usize,
) -> Result<<G as CurveAffine>::Projective, SynthesisError>
    where G: CurveAffine,
{
    let c = if n < 32 {
        3u32
    } else {
        (f64::from(n as u32)).ln().ceil() as u32
    };

    // Perform this region of the multiexp
    let this = move |bases: Arc<Vec<G>>,
                     exps: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
                     skip: u32,
                     n: usize,
                     start_idx_bases: usize,
                     start_idx_exps: usize|
                     -> Result<_, SynthesisError> {
        // Accumulate the result
        let mut acc = G::Projective::zero();

        // Create space for the buckets
        let mut buckets = vec![<G as CurveAffine>::Projective::zero(); (1 << c) - 1];

        let zero = <G::Engine as ScalarEngine>::Fr::zero().into_repr();
        let one = <G::Engine as ScalarEngine>::Fr::one().into_repr();

        // only the first round uses this
        let handle_trivial = skip == 0;

        // Sort the bases into buckets
        for i in 0..n {
            let exp_value = exps[start_idx_exps + i];
            let base_value = bases[start_idx_bases + i];
            if exp_value == one {
                if handle_trivial {
                    acc.add_assign_mixed(&base_value);
                }
            } else {
                if exp_value != zero {
                    let mut exp = exp_value;
                    exp.shr(skip);
                    let exp = exp.as_ref()[0] % (1 << c);

                    if exp != 0 {
                        buckets[(exp - 1) as usize].add_assign_mixed(&base_value);
                    }
                }
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = G::Projective::zero();
        for exp in buckets.into_iter().rev() {
            running_sum.add_assign(&exp);
            acc.add_assign(&running_sum);
        }

        Ok(acc)
    };

    let parts = (0..<G::Engine as ScalarEngine>::Fr::NUM_BITS)
        .step_by(c as usize)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|skip| this(bases.clone(), exps.clone(), skip, n, start_idx_bases, start_idx_exps))
        .collect::<Vec<Result<_, _>>>();

    parts
        .into_iter()
        .rev()
        .try_fold(<G as CurveAffine>::Projective::zero(), |mut acc, part| {
            for _ in 0..c {
                acc.double();
            }

            acc.add_assign(&part?);
            Ok(acc)
        })
}

fn multiexp_inner<G, S>(
    bases: S,
    density_map: Arc<DensityTracker>,
    exponents: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
    c: u32,
) -> Result<<G as CurveAffine>::Projective, SynthesisError>
    where
            G: CurveAffine,
            S: SourceBuilder<G>,
{
    // Perform this region of the multiexp
    let this = move |bases: S,
                     density_map: Arc<DensityTracker>,
                     exponents: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
                     skip: u32|
                     -> Result<_, SynthesisError> {
        // Accumulate the result
        let mut acc = G::Projective::zero();

        // Build a source for the bases
        let mut bases = bases.new();

        // Create space for the buckets
        let mut buckets = vec![<G as CurveAffine>::Projective::zero(); (1 << c) - 1];

        let zero = <G::Engine as ScalarEngine>::Fr::zero().into_repr();
        let one = <G::Engine as ScalarEngine>::Fr::one().into_repr();

        // only the first round uses this
        let handle_trivial = skip == 0;

        // Sort the bases into buckets
        let bv = Arc::new(&density_map.bv);
        for (&exp, density) in exponents.iter().zip(bv.iter()) {
            if *density {
                if exp == zero {
                    bases.skip(1)?;
                } else if exp == one {
                    if handle_trivial {
                        bases.add_assign_mixed(&mut acc)?;
                    } else {
                        bases.skip(1)?;
                    }
                } else {
                    let mut exp = exp;
                    exp.shr(skip);
                    let exp = exp.as_ref()[0] % (1 << c);

                    if exp != 0 {
                        bases.add_assign_mixed(&mut buckets[(exp - 1) as usize])?;
                    } else {
                        bases.skip(1)?;
                    }
                }
            }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = G::Projective::zero();
        for exp in buckets.into_iter().rev() {
            running_sum.add_assign(&exp);
            acc.add_assign(&running_sum);
        }

        Ok(acc)
    };

    let parts = vec![0..<G::Engine as ScalarEngine>::Fr::NUM_BITS]
        .par_chunks(c as usize)
        .map(|skip| this(bases.clone(), density_map.clone(), exponents.clone(), skip.last()
            .unwrap().clone().last().unwrap()))
        .collect::<Vec<Result<_, _>>>();

    parts
        .into_iter()
        .rev()
        .try_fold(<G as CurveAffine>::Projective::zero(), |mut acc, part| {
            for _ in 0..c {
                acc.double();
            }

            acc.add_assign(&part?);
            Ok(acc)
        })
}

/// Perform multi-exponentiation. The caller is responsible for ensuring the
/// query size is the same as the number of exponents.
pub fn multiexp<G>(
    bases: Arc<Vec<G>>,
    bases_skip: usize,
    density_map: Arc<DensityTracker>,
    exponents: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
    devices: Option<&gpu::DevicePool>,
) -> Box<dyn Future<Item=<G as CurveAffine>::Projective, Error=SynthesisError> + Send>
    where
            G: CurveAffine,
            G::Engine: crate::bls::Engine,
{
    if let Some(ref _devices) = devices {
        let mut exps = vec![exponents[0]; exponents.len()];
        let mut n = 0;
        let bv = Arc::new(&density_map.bv);
        info!{"e = {}, d = {}", exponents.len(), bv.len()};
        for (&e, d) in exponents.iter().zip(bv.iter()) {
            if *d {
                exps[n] = e;
                n += 1;
            }
        }

        let bss = bases.clone();
        let skip = bases_skip;
        match gpu::MultiexpKernel::<G::Engine>::multiexp(
            bss,
            Arc::new(exps),
            skip,
            n,
        ) {
            Ok(p) => {
                return rayon_core::scope(|s| {
                    Box::new(s.spawn_future(lazy(move || Ok::<_, SynthesisError>(p))))
                });
            }
            Err(e) => {
                error!("GPU Multiexp failed! Error: {}", e);
            }
        }
    }

    let c = if exponents.len() < 32 {
        3u32
    } else {
        (f64::from(exponents.len() as u32)).ln().ceil() as u32
    };

    if let Some(query_size) = density_map.get_query_size() {
        // If the density map has a known query size, it should not be
        // inconsistent with the number of exponents.
        assert!(query_size == exponents.len());
    }

    let bases = (bases, bases_skip);
    rayon_core::scope(|s| {
        Box::new(s.spawn_future(lazy(move || Ok::<_, SynthesisError>(multiexp_inner(bases, density_map, exponents, c).unwrap()))))
    })
}

// skipdensity
pub fn multiexp_skipdensity<G>(
    bases: Arc<Vec<G>>,
    bases_skip: usize,
    exponents: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
    n: usize,
    devices: Option<&gpu::DevicePool>,
) -> Box<dyn Future<Item=<G as CurveAffine>::Projective, Error=SynthesisError> + Send>
where
    G: CurveAffine,
    G::Engine: crate::bls::Engine,
{
    if let Some(ref _devices) = devices {
        let bss = bases.clone();
        let skip = bases_skip;
        match gpu::MultiexpKernel::<G::Engine>::multiexp(
            bss,
            exponents.clone(),
            skip,
            n,
        ) {
            Ok(p) => {
                return rayon_core::scope(|s| {
                    Box::new(s.spawn_future(lazy(move || Ok::<_, SynthesisError>(p))))
                });
            }
            Err(e) => {
                error!("GPU Multiexp failed! Error: {}", e);
            }
        }
    }

    rayon_core::scope(|s| {
        Box::new(s.spawn_future(lazy(move || Err(SynthesisError::GPUError(gpu::GPUError::GPUDisabled)))))
    })
}

// density map filter for exponents
pub fn density_filter<G>(
    _bases: Arc<Vec<G>>,
    density_map: Arc<DensityTracker>,
    exponents: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>
) ->  (Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>, usize)
where
    G: CurveAffine,
    G::Engine: crate::bls::Engine,
{
    let mut exps = vec![exponents[0]; exponents.len()];
    let mut n = 0;
    let bv = Arc::new(&density_map.bv);
    for (&e, d) in exponents.iter().zip(bv.iter()) {
        if *d {
            exps[n] = e;
            n += 1;
        }
    }
    (Arc::new(exps), n)
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::Rng;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_extend_density_regular() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for k in &[2, 4, 8] {
            for j in &[10, 20, 50] {
                let count: usize = k * j;

                let mut tracker_full = DensityTracker::new();
                let mut partial_trackers: Vec<DensityTracker> = Vec::with_capacity(count / k);
                for i in 0..count {
                    if i % k == 0 {
                        partial_trackers.push(DensityTracker::new());
                    }

                    let index: usize = i / k;
                    if rng.gen() {
                        tracker_full.add_element();
                        partial_trackers[index].add_element();
                    }

                    if !partial_trackers[index].bv.is_empty() {
                        let idx = rng.gen_range(0, partial_trackers[index].bv.len());
                        let offset: usize = partial_trackers
                            .iter()
                            .take(index)
                            .map(|t| t.bv.len())
                            .sum();
                        tracker_full.inc(offset + idx);
                        partial_trackers[index].inc(idx);
                    }
                }

                let mut tracker_combined = DensityTracker::new();
                for tracker in partial_trackers.into_iter() {
                    tracker_combined.extend(tracker, false);
                }
                assert_eq!(tracker_combined, tracker_full);
            }
        }
    }

    #[test]
    fn test_extend_density_input() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);
        let trials = 10;
        let max_bits = 10;
        let max_density = max_bits;

        // Create an empty DensityTracker.
        let empty = || DensityTracker::new();

        // Create a random DensityTracker with first bit unset.
        let unset = |rng: &mut XorShiftRng| {
            let mut dt = DensityTracker::new();
            dt.add_element();
            let n = rng.gen_range(1, max_bits);
            let target_density = rng.gen_range(0, max_density);
            for _ in 1..n {
                dt.add_element();
            }

            for _ in 0..target_density {
                if n > 1 {
                    let to_inc = rng.gen_range(1, n);
                    dt.inc(to_inc);
                }
            }
            assert!(!dt.bv[0]);
            assert_eq!(n, dt.bv.len());
            dbg!(&target_density, &dt.total_density);

            dt
        };

        // Create a random DensityTracker with first bit set.
        let set = |mut rng: &mut XorShiftRng| {
            let mut dt = unset(&mut rng);
            dt.inc(0);
            dt
        };

        for _ in 0..trials {
            {
                // Both empty.
                let (mut e1, e2) = (empty(), empty());
                e1.extend(e2, true);
                assert_eq!(empty(), e1);
            }
            {
                // First empty, second unset.
                let (mut e1, u1) = (empty(), unset(&mut rng));
                e1.extend(u1.clone(), true);
                assert_eq!(u1, e1);
            }
            {
                // First empty, second set.
                let (mut e1, s1) = (empty(), set(&mut rng));
                e1.extend(s1.clone(), true);
                assert_eq!(s1, e1);
            }
            {
                // First set, second empty.
                let (mut s1, e1) = (set(&mut rng), empty());
                let s2 = s1.clone();
                s1.extend(e1, true);
                assert_eq!(s1, s2);
            }
            {
                // First unset, second empty.
                let (mut u1, e1) = (unset(&mut rng), empty());
                let u2 = u1.clone();
                u1.extend(e1, true);
                assert_eq!(u1, u2);
            }
            {
                // First unset, second unset.
                let (mut u1, u2) = (unset(&mut rng), unset(&mut rng));
                let expected_total = u1.total_density + u2.total_density;
                u1.extend(u2, true);
                assert_eq!(expected_total, u1.total_density);
                assert!(!u1.bv[0]);
            }
            {
                // First unset, second set.
                let (mut u1, s1) = (unset(&mut rng), set(&mut rng));
                let expected_total = u1.total_density + s1.total_density;
                u1.extend(s1, true);
                assert_eq!(expected_total, u1.total_density);
                assert!(u1.bv[0]);
            }
            {
                // First set, second unset.
                let (mut s1, u1) = (set(&mut rng), unset(&mut rng));
                let expected_total = s1.total_density + u1.total_density;
                s1.extend(u1, true);
                assert_eq!(expected_total, s1.total_density);
                assert!(s1.bv[0]);
            }
            {
                // First set, second set.
                let (mut s1, s2) = (set(&mut rng), set(&mut rng));
                let expected_total = s1.total_density + s2.total_density - 1;
                s1.extend(s2, true);
                assert_eq!(expected_total, s1.total_density);
                assert!(s1.bv[0]);
            }
        }
    }
}
