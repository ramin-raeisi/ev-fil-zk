use std::sync::{mpsc, Arc};
use std::time::Instant;

use crate::bls::Engine;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use rayon::prelude::*;
use crossbeam;

use super::{ParameterGetter, Proof};
use crate::domain::{EvaluationDomain};
use crate::multiexp::{multiexp, multiexp_skipdensity, density_filter};
use crate::{
    Circuit, ConstraintSystem, Index, SynthesisError, Variable,
};
use futures::future::Future;
use log::info;
use crate::gpu::{DEVICE_POOL};
use super::super::prover::ProvingAssignment;

pub fn create_proof_batch_preload<E, C, P: ParameterGetter<E>>(
    circuits: Vec<C>,
    params: P,
) -> Result<Vec<Proof<E>>, SynthesisError>
    where
        E: Engine,
        C: Circuit<E> + Send,
{
    let prover_start = Instant::now();

    let mut provers = circuits
        .into_par_iter()
        .map(|circuit| -> Result<_, SynthesisError> {
            let mut prover = ProvingAssignment::new();

            prover.alloc_input(|| "", || Ok(E::Fr::one()))?;

            circuit.synthesize(&mut prover)?;

            for i in 0..prover.input_assignment.len() {
                prover.enforce(|| "", |lc| lc + Variable(Index::Input(i)), |lc| lc, |lc| lc);
            }

            Ok(prover)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let prover_time = prover_start.elapsed();
    info!("Circuit conversion phase time: {:?}", prover_time);

    // Start fft/multiexp prover timer
    let start = Instant::now();
    info!("starting proof timer");

    let vk = params.get_vk()?;
    let n = provers[0].a.len();

    // Make sure all circuits have the same input len.
    provers.par_iter().for_each(|prover| {
        assert_eq!(
            prover.a.len(),
            n,
            "only equally sized circuits are supported"
        );
    });

    let now = Instant::now();
    let (tx_h, rx_h) = mpsc::channel();
    let (tx_l, rx_l) = mpsc::channel();
    let (tx_a, rx_a) = mpsc::channel();
    let (tx_bg2, rx_bg2) = mpsc::channel();
    let (tx_input_assignments, rx_input_assignments) = mpsc::channel();
    let (tx_aux_assignments, rx_aux_assignments) = mpsc::channel();
    crossbeam::scope(|s| {
        let mut threads = Vec::new();
        let params = &params;
        let provers = &mut provers;
        // h_params
        threads.push(s.spawn(move |_| {
            let h_params = params.get_h().unwrap();
            tx_h.send(h_params).unwrap();
        }));
        // l_params
        threads.push(s.spawn(move |_| {
            let l_params = params.get_l().unwrap();
            tx_l.send(l_params).unwrap();
        }));
        // a_params
        threads.push(s.spawn(move |_| {
            let a_base = params.get_a().unwrap();
            tx_a.send(a_base).unwrap();
        }));
        // bg2_params
        threads.push(s.spawn(move |_| {
            let b_g2_base = params.get_b_g2().unwrap();
            tx_bg2.send(b_g2_base).unwrap();
        }));
        // assignments
        threads.push(s.spawn(move |_| {
            let input_assignments = provers	
                .par_iter_mut()	
                .map(|prover| {	
                    let input_assignment = std::mem::replace(&mut prover.input_assignment, Vec::new());	
                    Arc::new(	
                        input_assignment	
                            .par_iter()	
                            .map(|s| s.into_repr())	
                            .collect::<Vec<_>>(),	
                    )	
                })	
                .collect::<Vec<_>>();
            tx_input_assignments.send(input_assignments).unwrap();

            let aux_assignments = provers	
                .par_iter_mut()	
                .map(|prover| {	
                    let aux_assignment = std::mem::replace(&mut prover.aux_assignment, Vec::new());	
                    Arc::new(	
                        aux_assignment	
                            .par_iter()	
                            .map(|s| s.into_repr())	
                            .collect::<Vec<_>>(),	
                    )	
                })	
                .collect::<Vec<_>>();
            tx_aux_assignments.send(aux_assignments).unwrap();
        }));

        for t in threads {
            t.join().unwrap();
        }
    }).unwrap();
    let h_base = rx_h.recv().unwrap();
    let l_base = rx_l.recv().unwrap();
    let a_base = rx_a.recv().unwrap();
    let b_g2_base = rx_bg2.recv().unwrap();
    let input_assignments = rx_input_assignments.recv().unwrap();
    let aux_assignments = rx_aux_assignments.recv().unwrap();
    info!("params preload time: {:?}", now.elapsed());

    info!("starting fft + multiexp phases in parallel");
    let now = Instant::now();

    let (tx_l_s, rx_l_s) = mpsc::sync_channel(1);
    let (tx_inputs, rx_inputs) = mpsc::sync_channel(1);
    let (tx_h_s, rx_h_s) = mpsc::sync_channel(1);
    let (tx_a_s, rx_a_s) = mpsc::sync_channel(1);

    crossbeam::scope(|s| {
        let mut threads = Vec::new();
        let provers_a = &mut provers;
        // a_s
        threads.push(s.spawn(move |_| {
            let tx_a_s = tx_a_s.clone();
            let a_s = provers_a
            .par_iter_mut()
            .map(|prover| {
                let mut a =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new()))?;
                let mut b =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new()))?;
                let mut c =
                    EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new()))?;

                let mut coeff = vec![&mut a, &mut b, &mut c];

                coeff.par_iter_mut().enumerate().for_each(move |(i, v)| {
                    if i == 2 {
                        v.ifft(Some(&DEVICE_POOL)).unwrap();
                    } else {
                        v.ifft(Some(&DEVICE_POOL)).unwrap();
                        v.coset_fft(Some(&DEVICE_POOL)).unwrap();
                    }
                });

                a.mul_assign(&b, Some(&DEVICE_POOL))?;
                drop(b);
                a.icoset_fft(Some(&DEVICE_POOL))?;
                a.sub_assign(&c, Some(&DEVICE_POOL))?;
                drop(c);
                a.divide_by_z_on_coset();
                let mut a = a.into_coeffs();
                let a_len = a.len() - 1;
                a.truncate(a_len);

                Ok(Arc::new(
                    a.iter().map(|s| s.0.into_repr()).collect::<Vec<_>>(),
                ))
            })
            .collect::<Result<Vec<_>, SynthesisError>>().unwrap();

            tx_a_s.send(a_s).unwrap();
        }));

        // l_s
        threads.push(s.spawn(|_| {
            let l_skip = 0;
            let aux_assignments_arc = aux_assignments.clone();
            let l_s = aux_assignments_arc
                .into_par_iter()
                .map(|aux_assignment| {
                    multiexp_skipdensity(
                        l_base.clone(),
                        l_skip,
                        Arc::clone(&aux_assignment),
                        aux_assignment.len(),
                        Some(&DEVICE_POOL))
                }).collect::<Vec<_>>();
            
            tx_l_s.send(l_s).unwrap();
        }));
        
        for t in threads {
            t.join().unwrap();
        }
    }).unwrap();

    crossbeam::scope(|s| {
        let mut threads = Vec::new();
        // inputs
        threads.push(s.spawn(|_| {
            let inputs = provers
            .par_iter()
            .zip(input_assignments.par_iter())
            .zip(aux_assignments.par_iter())
            .map(|((prover, input_assignment), aux_assignment)| {

                let a_input_skip = 0;
                let a_aux_skip = input_assignment.len();

                let a_inputs = multiexp_skipdensity(
                    a_base.clone(),
                    a_input_skip,
                    input_assignment.clone(),
                    input_assignment.len(),
                    Some(&DEVICE_POOL),
                );

                let (a_aux_exps, a_aux_n) = density_filter(
                    a_base.clone(),
                    Arc::new(prover.a_aux_density.clone()),
                    aux_assignment.clone(),
                );

                let a_aux = multiexp_skipdensity(
                    a_base.clone(),
                    a_aux_skip,
                    a_aux_exps,
                    a_aux_n,
                    Some(&DEVICE_POOL),
                );

                let b_input_density = Arc::new(prover.b_input_density.clone());
                let b_input_density_total = b_input_density.get_total_density();
                let b_aux_density = Arc::new(prover.b_aux_density.clone());
                //let b_aux_density_total = b_aux_density.get_total_density();

                let b_input_skip = 0;
                let b_aux_skip = b_input_density_total;

                let b_g2_inputs = multiexp(
                    b_g2_base.clone(),
                    b_input_skip,
                    b_input_density,
                    input_assignment.clone(),
                    Some(&DEVICE_POOL),
                );

                let (b_g2_aux_exps, b_g2_aux_n) = density_filter(
                    b_g2_base.clone(),
                    b_aux_density.clone(),
                    aux_assignment.clone()
                );

                let b_g2_aux = multiexp_skipdensity(
                    b_g2_base.clone(),
                    b_aux_skip,
                    b_g2_aux_exps,
                    b_g2_aux_n,
                    Some(&DEVICE_POOL),
                );

                Ok((
                    a_inputs,
                    a_aux,
                    b_g2_inputs,
                    b_g2_aux,
                ))
            })
            .collect::<Result<Vec<_>, SynthesisError>>().unwrap();

            tx_inputs.send(inputs).unwrap();
        }));

        // h_s
        threads.push(s.spawn(move |_| {
            let a_s = rx_a_s.recv().unwrap();
            let h_skip = 0;
            let h_s = a_s
                .into_par_iter()
                .map(|a| {
                    multiexp_skipdensity(
                        h_base.clone(),
                        h_skip,
                        Arc::clone(&a),
                        a.len(),
                        Some(&DEVICE_POOL))
                })
                .collect::<Vec<_>>();

            tx_h_s.send(h_s).unwrap();
        }));

        for t in threads {
            t.join().unwrap();
        }
    }).unwrap();

    let l_s = rx_l_s.recv().unwrap();
    let inputs = rx_inputs.recv().unwrap();
    let h_s = rx_h_s.recv().unwrap();

    info!("fft + multiexp phases time: {:?}", now.elapsed());

    let proofs = //h_s_l_s.into_par_iter()
        h_s.into_par_iter()
        .zip(l_s.into_par_iter())
        .zip(inputs.into_par_iter())
        .map(|((h, l), (a_inputs, a_aux, b_g2_inputs, b_g2_aux))| {
            if vk.delta_g1.is_zero() || vk.delta_g2.is_zero() {
                // If this element is zero, someone is trying to perform a
                // subversion-CRS attack.
                return Err(SynthesisError::UnexpectedIdentity);
            }

            let mut g_a = vk.alpha_g1.into_projective();
            let mut g_b = vk.beta_g2.into_projective();
            let mut g_c = E::G1::zero();

            let mut a_answer = a_inputs.wait()?;
            a_answer.add_assign(&a_aux.wait()?);
            g_a.add_assign(&a_answer);

            let mut b2_answer = b_g2_inputs.wait()?;
            b2_answer.add_assign(&b_g2_aux.wait()?);
            g_b.add_assign(&b2_answer);

            g_c.add_assign(&h.wait()?);
            g_c.add_assign(&l.wait()?);

            Ok(Proof {
                a: g_a.into_affine(),
                b: g_b.into_affine(),
                c: g_c.into_affine(),
            })
        }).collect::<Result<Vec<_>, SynthesisError>>()?;

    let proof_time = start.elapsed();
    info!("prover time: {:?}", proof_time);

    Ok(proofs)
}