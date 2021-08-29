use super::error::{GPUError, GPUResult};
use ff::{PrimeField, ScalarEngine};
use groupy::CurveAffine;
use std::marker::PhantomData;
use std::sync::Arc;

// This module is compiled instead of `fft.rs` and `multiexp.rs` if `gpu` feature is disabled.
#[allow(clippy::upper_case_acronyms)]
pub struct FFTKernel<E>(PhantomData<E>)
    where
        E: ScalarEngine;

impl<E> FFTKernel<E>
    where
        E: ScalarEngine,
{
    pub fn create(_: bool) -> GPUResult<FFTKernel<E>> {
        Err(GPUError::GPUDisabled)
    }
    pub fn fft(_: &mut [E::Fr], _: &E::Fr, _: u32) -> GPUResult<()> {
        return Err(GPUError::GPUDisabled);
    }

    pub fn radix_fft(_: &mut [E::Fr], _: &E::Fr, _: u32) -> GPUResult<()> {
        return Err(GPUError::GPUDisabled);
    }

    pub fn distribute_powers(_: &mut [E::Fr], _: &E::Fr, _: u32) -> GPUResult<()> {
        return Err(GPUError::GPUDisabled);
    }

    pub fn mul_sub(_: &mut [E::Fr], _: &[E::Fr], _: usize, _: bool) -> GPUResult<()> {
        return Err(GPUError::GPUDisabled);
    }
}

pub struct MultiexpKernel<E>(PhantomData<E>)
    where
        E: ScalarEngine;

impl<E> MultiexpKernel<E>
    where
        E: ScalarEngine,
{
    pub fn create(_: bool) -> GPUResult<MultiexpKernel<E>> {
        Err(GPUError::GPUDisabled)
    }

    pub fn multiexp<G>(
        &mut self,
        _: &DevicePool,
        _: Arc<Vec<G>>,
        _: Arc<Vec<<<G::Engine as ScalarEngine>::Fr as PrimeField>::Repr>>,
        _: usize,
        _: usize,
    ) -> GPUResult<<G as CurveAffine>::Projective>
        where
            G: CurveAffine,
    {
        Err(GPUError::GPUDisabled)
    }
}

pub struct DevicePool;

impl Default for DevicePool {
    fn default() -> Self {
        Self
    }
}

use crate::bls::Engine;
