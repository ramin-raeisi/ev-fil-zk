use crate::bls::{Bls12, Engine};
use crate::gpu::{error::{GPUError, GPUResult}, structs, scheduler, utils};
use futures::future::Future;
use ff::Field;
use log::info;
use rust_gpu_tools::*;
use std::cmp;
use std::any::TypeId;

const LOG2_MAX_ELEMENTS: usize = 32;
// At most 2^32 elements is supported.
const MAX_LOG2_RADIX: u32 = 9;
// Radix256
const MAX_LOG2_LOCAL_WORK_SIZE: u32 = 8; // 128

#[allow(clippy::upper_case_acronyms)]
pub struct FFTKernel<E>
where
    E: Engine,
{
    _phantom: std::marker::PhantomData<E::Fr>,
}

impl<E> FFTKernel<E>
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

    /// Peforms a FFT round
    /// * `log_n` - Specifies log2 of number of elements
    /// * `log_p` - Specifies log2 of `p`, (http://www.bealto.com/gpu-fft_group-1.html)
    /// * `deg` - 1=>radix2, 2=>radix4, 3=>radix8, ...
    /// * `max_deg` - The precalculated values pq` and `omegas` are valid for radix degrees up to `max_deg`
    fn radix_fft_round(
        program: &opencl::Program,
        src_buffer: &opencl::Buffer<E::Fr>,
        dst_buffer: &opencl::Buffer<E::Fr>,
        pq_buffer: &opencl::Buffer<E::Fr>,
        omegas_buffer: &opencl::Buffer<E::Fr>,
        log_n: u32,
        log_p: u32,
        deg: u32,
        max_deg: u32,
    ) -> GPUResult<()> {
        FFTKernel::<E>::ensure_curve()?;

        let n = 1u32 << log_n;
        let local_work_size = 1 << cmp::min(deg - 1, MAX_LOG2_LOCAL_WORK_SIZE);
        let global_work_size = (n >> deg) * local_work_size;
        let kernel = program.create_kernel(
            "radix_fft",
            global_work_size as usize,
            Some(local_work_size as usize),
        );
        call_kernel!(
            kernel,
            src_buffer,
            dst_buffer,
            pq_buffer,
            omegas_buffer,
            opencl::LocalBuffer::<E::Fr>::new(1 << deg),
            n,
            log_p,
            deg,
            max_deg
        )?;
        Ok(())
    }

    /// Share some precalculated values between threads to boost the performance
    fn setup_pq_omegas(
        program: &opencl::Program,
        omega: &E::Fr,
        n: usize,
        max_deg: u32,
    ) -> GPUResult<(opencl::Buffer<E::Fr>, opencl::Buffer<E::Fr>)> {
        FFTKernel::<E>::ensure_curve()?;

        // Precalculate:
        // [omega^(0/(2^(deg-1))), omega^(1/(2^(deg-1))), ..., omega^((2^(deg-1)-1)/(2^(deg-1)))]
        let mut pq = vec![E::Fr::zero(); 1 << max_deg >> 1];
        let twiddle = omega.pow([(n >> max_deg) as u64]);
        pq[0] = E::Fr::one();
        if max_deg > 1 {
            pq[1] = twiddle;
            for i in 2..(1 << max_deg >> 1) {
                pq[i] = pq[i - 1];
                pq[i].mul_assign(&twiddle);
            }
        }
        let mut pq_buffer = program.create_buffer::<E::Fr>(1 << MAX_LOG2_RADIX >> 1)?;
        pq_buffer.write_from(0, &pq)?;

        // Precalculate [omega, omega^2, omega^4, omega^8, ..., omega^(2^31)]
        let mut omegas = vec![E::Fr::zero(); 32];
        omegas[0] = *omega;
        for i in 1..LOG2_MAX_ELEMENTS {
            omegas[i] = omegas[i - 1].pow([2u64]);
        }
        let mut omegas_buffer = program.create_buffer::<E::Fr>(LOG2_MAX_ELEMENTS)?;
        omegas_buffer.write_from(0, &omegas)?;

        Ok((pq_buffer, omegas_buffer))
    }

    /// Performs FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `log_n` - Specifies log2 of number of elements
    pub fn radix_fft(a: &'static mut [<E as blstrs::ScalarEngine>::Fr],
        omega: &E::Fr,
        log_n: u32,
    ) -> GPUResult<()> {
        FFTKernel::<E>::ensure_curve()?;

        let omega = *omega;
        scheduler::schedule(move |program| -> GPUResult<&mut [E::Fr]> {
            let n = 1 << log_n;
            info!(
                "Running new radix FFT of {} elements on {}(bus_id: {})...",
                n,
                program.device().name(),
                program.device().bus_id().unwrap()
            );
            let mut src_buffer = program.create_buffer::<E::Fr>(n)?;
            let mut dst_buffer = program.create_buffer::<E::Fr>(n)?;

            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);
            let (pq_buffer, omegas_buffer) =
                FFTKernel::<E>::setup_pq_omegas(program, &omega, n, max_deg)?;

            src_buffer.write_from(0, &a)?;
            let mut log_p = 0u32;
            while log_p < log_n {
                let deg = cmp::min(max_deg, log_n - log_p);
                FFTKernel::<E>::radix_fft_round(
                    program,
                    &src_buffer,
                    &dst_buffer,
                    &pq_buffer,
                    &omegas_buffer,
                    log_n,
                    log_p,
                    deg,
                    max_deg,
                )?;
                log_p += deg;
                std::mem::swap(&mut src_buffer, &mut dst_buffer);
            }

            src_buffer.read_into(0, a)?;

            Ok(a)
        }).wait().unwrap()?;
        Ok(())
    }

    /// Performs inplace FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `lgn` - Specifies log2 of number of elements
    pub fn inplace_fft(a: &'static mut [<E as blstrs::ScalarEngine>::Fr], omega: &E::Fr, log_n: u32) -> GPUResult<()> {
        FFTKernel::<E>::ensure_curve()?;

        let omega = *omega;
        scheduler::schedule(move |program| -> GPUResult<&mut [E::Fr]> {
            let n = 1 << log_n;
            info!(
                "Running inplace FFT of {} elements on {}(bus_id: {})...",
                n,
                program.device().name(),
                program.device().bus_id().unwrap()
            );
            let mut src_buffer = program.create_buffer::<E::Fr>(n)?;

            let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);
            let (_pq_buffer, omegas_buffer) =
                FFTKernel::<E>::setup_pq_omegas(program, &omega, n, max_deg)?;

            src_buffer.write_from(0, &a)?;
            let kernel = program.create_kernel("reverse_bits", n, None);
            call_kernel!(kernel, &src_buffer, log_n)?;

            for log_m in 0..log_n {
                let kernel = program.create_kernel("inplace_fft", n >> 1, None);
                call_kernel!(kernel, &src_buffer, &omegas_buffer, log_n, log_m)?;
            }

            src_buffer.read_into(0, a)?;

            Ok(a)
        }).wait().unwrap()?;
        Ok(())
    }


    /// Distribute powers of `g` on `a`
    /// * `lgn` - Specifies log2 of number of elements
    pub fn distribute_powers(a: &'static mut [<E as blstrs::ScalarEngine>::Fr], g: &E::Fr, lgn: u32) -> GPUResult<()> {
        let gl = *g;

        scheduler::schedule(move |program| -> GPUResult<&mut [E::Fr]>{
            let n = 1u32 << lgn;
            info!(
                "Running powers distribution of {} elements on {}(bus_id: {})...",
                n,
                program.device().name(),
                program.device().bus_id().unwrap()
            );

            let mut src_buffer = program.create_buffer::<E::Fr>(n as usize)?;
            src_buffer.write_from(0, &a)?;

            let g_arg = structs::PrimeFieldStruct::<E::Fr>(gl);

            let kernel = program.create_kernel(
                "distribute_powers",
                utils::get_core_count(&program.device()),
                None,
            );
            call_kernel!(
                kernel,
                &src_buffer,
                n,
                g_arg)?;

            src_buffer.read_into(0, a)?;

            Ok(a)
        }).wait().unwrap()?;
        Ok(())
    }


    /// Memberwise multiplication/subtraction
    /// * `lgn` - Specifies log2 of number of elements
    /// * `sub` - Set true if you want subtraction instead of multiplication
    pub fn mul_sub(a: &'static mut [<E as blstrs::ScalarEngine>::Fr], b: &'static [<E as blstrs::ScalarEngine>::Fr], n: usize, sub: bool) -> GPUResult<()> {
        scheduler::schedule(move |program| -> GPUResult<&mut [E::Fr]>{
            if sub {
                info!(
                    "Running sub of {} elements on {} (bus_id: {})...",
                    a.len(),
                    program.device().name(),
                    program.device().bus_id().unwrap()
                );
            } else {
                info!(
                    "Running mul of {} elements on {} (bus_id: {})...",
                    a.len(),
                    program.device().name(),
                    program.device().bus_id().unwrap()
                );
            }

            let mut src_buffer = program.create_buffer::<E::Fr>(a.len())?;
            let mut dst_buffer = program.create_buffer::<E::Fr>(b.len())?;

            src_buffer.write_from(0, &a)?;
            dst_buffer.write_from(0, &b)?;

            let kernel = program.create_kernel(
                if sub { "sub" } else { "mul" },
                n,
                None,
            );
            call_kernel!(
                kernel,
                &src_buffer,
                &dst_buffer,
                n as u32)?;

            src_buffer.read_into(0, a)?;

            Ok(a)
        }).wait().unwrap()?;
        Ok(())
    }
}
