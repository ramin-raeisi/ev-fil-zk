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
    pub fn radix_fft(a: &mut [E::Fr],
                     omega: &E::Fr,
                     log_n: u32,
    ) -> GPUResult<()> {
        FFTKernel::<E>::ensure_curve()?;

        let mut elems = a.to_vec();
        let omega = *omega;
        let result =
            scheduler::schedule(move |program| -> GPUResult<Vec<E::Fr>> {
                let n = 1 << log_n;
                info!(
                    "Running new radix FFT of {} elements on {}(bus_id: {})...",
                    n,
                    program.device().name(),
                    program.device().bus_id().unwrap()
                );
                let mut src_buffer = program.create_buffer::<E::Fr>(n)?;

                let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);
                let (pq_buffer, omegas_buffer) =
                    FFTKernel::<E>::setup_pq_omegas(program, &omega, n, max_deg)?;

                src_buffer.write_from(0, &elems)?;
                let mut log_p = 0u32;
                while log_p < log_n {
                    let deg = cmp::min(max_deg, log_n - log_p);
                    FFTKernel::<E>::radix_fft_round(
                        program,
                        &src_buffer,
                        &pq_buffer,
                        &omegas_buffer,
                        log_n,
                        log_p,
                        deg,
                        max_deg,
                    )?;
                    log_p += deg;
                }

                src_buffer.read_into(0, &mut elems)?;

                Ok(elems)
            }).wait().unwrap()?;
        a.copy_from_slice(&result);
        Ok(())
    }

        /// Performs inplace FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `lgn` - Specifies log2 of number of elements
    pub fn inplace_fft(a: &mut [E::Fr], omega: &E::Fr, log_n: u32) -> GPUResult<()> {
        FFTKernel::<E>::ensure_curve()?;

        let mut elems = a.to_vec();
        let omega = *omega;
        let result =
            scheduler::schedule(move |program| -> GPUResult<Vec<E::Fr>> {
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

                src_buffer.write_from(0, &elems)?;
                let kernel = program.create_kernel("reverse_bits", n, None);
                call_kernel!(kernel, &src_buffer, log_n)?;

                for log_m in 0..log_n {
                    let kernel = program.create_kernel("inplace_fft", n >> 1, None);
                    call_kernel!(kernel, &src_buffer, &omegas_buffer, log_n, log_m)?;
                }

                src_buffer.read_into(0, &mut elems)?;

                Ok(elems)
            }).wait().unwrap()?;
        a.copy_from_slice(&result);
        Ok(())
    }

    /// Performs inplace FFT on `a`
    /// * `omega` - Special value `omega` is used for FFT over finite-fields
    /// * `lgn` - Specifies log2 of number of elements
    pub fn inplace_fft2(a: &mut [E::Fr], b: &mut [E::Fr], omega: &E::Fr, log_n: u32) -> GPUResult<()> {
        FFTKernel::<E>::ensure_curve()?;

        let mut elems_a = a.to_vec();
        let mut elems_b = b.to_vec();
        let omega = *omega;
        let (result_a, result_b) =
            scheduler::schedule(move |program| -> GPUResult<(Vec<E::Fr>, Vec<E::Fr>)> {
                let n = 1 << log_n;
                info!(
                    "Running inplace 2 FFT of {} elements on {}(bus_id: {})...",
                    n,
                    program.device().name(),
                    program.device().bus_id().unwrap()
                );
                let mut src_buffer_a = program.create_buffer::<E::Fr>(n)?;
                let mut src_buffer_b = program.create_buffer::<E::Fr>(n)?;

                let max_deg = cmp::min(MAX_LOG2_RADIX, log_n);
                let (_pq_buffer, omegas_buffer) =
                    FFTKernel::<E>::setup_pq_omegas(program, &omega, n, max_deg)?;

                src_buffer_a.write_from(0, &elems_a)?;
                src_buffer_b.write_from(0, &elems_b)?;

                let kernel = program.create_kernel("reverse_bits2", n << 1 , None);
                call_kernel!(kernel, &src_buffer_a, &src_buffer_b, log_n).unwrap();

                for log_m in 0..log_n {
                    let kernel = program.create_kernel("inplace_fft2", n, None);
                    call_kernel!(kernel, &src_buffer_a, &src_buffer_b, &omegas_buffer, log_n, log_m).unwrap();
                }
                src_buffer_a.read_into(0, &mut elems_a)?;
                src_buffer_b.read_into(0, &mut elems_b)?;

                Ok((elems_a, elems_b))
            }).wait().unwrap()?;
        a.copy_from_slice(&result_a);
        b.copy_from_slice(&result_b);
        Ok(())
    }

   

    /// Distribute powers of `g` on `a`
    /// * `lgn` - Specifies log2 of number of elements
    pub fn distribute_powers(a: &mut [E::Fr], g: &E::Fr, lgn: u32) -> GPUResult<()> {
        let mut elems: Vec<E::Fr> = a.to_vec();
        let gl = *g;
        let result =
            scheduler::schedule(move |program| -> GPUResult<Vec<E::Fr>> {
                let n = 1u32 << lgn;
                info!(
                    "Running powers distribution of {} elements on {}(bus_id: {})...",
                    n,
                    program.device().name(),
                    program.device().bus_id().unwrap()
                );

                let mut src_buffer = program.create_buffer::<E::Fr>(n as usize)?;
                src_buffer.write_from(0, &elems)?;

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

                src_buffer.read_into(0, &mut elems)?;

                Ok(elems)
            }).wait().unwrap()?;
        a.copy_from_slice(&result);
        Ok(())
    }

    /// Distribute powers of `g` on `a`
    /// * `lgn` - Specifies log2 of number of elements
    pub fn distribute_powers2(a: &mut [E::Fr], b: &mut [E::Fr], g: &E::Fr, lgn: u32) -> GPUResult<()> {
        let mut elems_a: Vec<E::Fr> = a.to_vec();
        let mut elems_b: Vec<E::Fr> = b.to_vec();
        let gl = *g;
        let (result_a, result_b) =
            scheduler::schedule(move |program| -> GPUResult<(Vec<E::Fr>, Vec<E::Fr>)> {
                let n = 1u32 << lgn;
                info!(
                    "Running 2 powers distribution of {} elements on {}(bus_id: {})...",
                    n,
                    program.device().name(),
                    program.device().bus_id().unwrap()
                );

                let mut src_buffer_a = program.create_buffer::<E::Fr>(n as usize)?;
                let mut src_buffer_b = program.create_buffer::<E::Fr>(n as usize)?;
                src_buffer_a.write_from(0, &elems_a)?;
                src_buffer_b.write_from(0, &elems_b)?;
                let g_arg = structs::PrimeFieldStruct::<E::Fr>(gl);

                let kernel = program.create_kernel(
                    "distribute_powers2",
                    //utils::get_core_count(&program.device()),
                    (n << 1) as usize,
                    None,
                );
                call_kernel!(
                    kernel,
                    &src_buffer_a,
                    &src_buffer_b,
                    n,
                    g_arg).unwrap();

                src_buffer_a.read_into(0, &mut elems_a)?;
                src_buffer_b.read_into(0, &mut elems_b)?;
                Ok((elems_a, elems_b))
            }).wait().unwrap()?;
        a.copy_from_slice(&result_a);
        b.copy_from_slice(&result_b);
        Ok(())
    }

    /// Memberwise multiplication/subtraction
    /// * `lgn` - Specifies log2 of number of elements
    /// * `sub` - Set true if you want subtraction instead of multiplication
    pub fn mul_sub(a: &mut [E::Fr], b: &[E::Fr], n: usize, sub: bool) -> GPUResult<()> {
        let mut aelems = a.to_vec();
        let belems = b.to_vec();
        let result =
            scheduler::schedule(move |program| -> GPUResult<Vec<E::Fr>> {
                if sub {
                    info!(
                        "Running sub of {} elements on {} (bus_id: {})...",
                        aelems.len(),
                        program.device().name(),
                        program.device().bus_id().unwrap()
                    );
                } else {
                    info!(
                        "Running mul of {} elements on {} (bus_id: {})...",
                        aelems.len(),
                        program.device().name(),
                        program.device().bus_id().unwrap()
                    );
                }

                let mut src_buffer = program.create_buffer::<E::Fr>(aelems.len())?;
                let mut dst_buffer = program.create_buffer::<E::Fr>(belems.len())?;

                src_buffer.write_from(0, &aelems)?;
                dst_buffer.write_from(0, &belems)?;

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

                src_buffer.read_into(0, &mut aelems)?;

                Ok(aelems)
            }).wait().unwrap()?;
        a.copy_from_slice(&result);
        Ok(())
    }
}
