use std::sync::{Arc};
use std::time::Instant;

use crate::bls::Engine;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use rayon::prelude::*;

use super::{ParameterSource, Proof};
use crate::domain::{EvaluationDomain, Scalar};
use crate::multiexp::{multiexp, DensityTracker, FullDensity};
use crate::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use futures::future::Future;
use log::info;
use crate::gpu::{DEVICE_POOL};

fn eval<E: Engine>(
    lc: &LinearCombination<E>,
    mut input_density: Option<&mut DensityTracker>,
    mut aux_density: Option<&mut DensityTracker>,
    input_assignment: &[E::Fr],
    aux_assignment: &[E::Fr],
) -> E::Fr {
    let mut acc = E::Fr::zero();

    for (&index, &coeff) in lc.0.iter() {
        let mut tmp;

        match index {
            Variable(Index::Input(i)) => {
                tmp = input_assignment[i];
                if let Some(ref mut v) = input_density {
                    v.inc(i);
                }
            }
            Variable(Index::Aux(i)) => {
                tmp = aux_assignment[i];
                if let Some(ref mut v) = aux_density {
                    v.inc(i);
                }
            }
        }

        if coeff == E::Fr::one() {
            acc.add_assign(&tmp);
        } else {
            tmp.mul_assign(&coeff);
            acc.add_assign(&tmp);
        }
    }

    acc
}

struct ProvingAssignment<E: Engine> {
    // Density of queries
    a_aux_density: DensityTracker,
    b_input_density: DensityTracker,
    b_aux_density: DensityTracker,

    // Evaluations of A, B, C polynomials
    a: Vec<Scalar<E>>,
    b: Vec<Scalar<E>>,
    c: Vec<Scalar<E>>,

    // Assignments of variables
    input_assignment: Vec<E::Fr>,
    aux_assignment: Vec<E::Fr>,
}

use std::fmt;

impl<E: Engine> fmt::Debug for ProvingAssignment<E> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        fmt.debug_struct("ProvingAssignment")
            .field("a_aux_density", &self.a_aux_density)
            .field("b_input_density", &self.b_input_density)
            .field("b_aux_density", &self.b_aux_density)
            .field(
                "a",
                &self
                    .a
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field(
                "b",
                &self
                    .b
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field(
                "c",
                &self
                    .c
                    .iter()
                    .map(|v| format!("Fr({:?})", v.0))
                    .collect::<Vec<_>>(),
            )
            .field("input_assignment", &self.input_assignment)
            .field("aux_assignment", &self.aux_assignment)
            .finish()
    }
}

impl<E: Engine> PartialEq for ProvingAssignment<E> {
    fn eq(&self, other: &ProvingAssignment<E>) -> bool {
        self.a_aux_density == other.a_aux_density
            && self.b_input_density == other.b_input_density
            && self.b_aux_density == other.b_aux_density
            && self.a == other.a
            && self.b == other.b
            && self.c == other.c
            && self.input_assignment == other.input_assignment
            && self.aux_assignment == other.aux_assignment
    }
}

impl<E: Engine> ConstraintSystem<E> for ProvingAssignment<E> {
    type Root = Self;

    fn new() -> Self {
        Self {
            a_aux_density: DensityTracker::new(),
            b_input_density: DensityTracker::new(),
            b_aux_density: DensityTracker::new(),
            a: vec![],
            b: vec![],
            c: vec![],
            input_assignment: vec![],
            aux_assignment: vec![],
        }
    }

    fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
        where
            F: FnOnce() -> Result<E::Fr, SynthesisError>,
            A: FnOnce() -> AR,
            AR: Into<String>,
    {
        self.aux_assignment.push(f()?);
        self.a_aux_density.add_element();
        self.b_aux_density.add_element();

        Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
    }

    fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
        where
            F: FnOnce() -> Result<E::Fr, SynthesisError>,
            A: FnOnce() -> AR,
            AR: Into<String>,
    {
        self.input_assignment.push(f()?);
        self.b_input_density.add_element();

        Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, a: LA, b: LB, c: LC)
        where
            A: FnOnce() -> AR,
            AR: Into<String>,
            LA: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
            LB: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
            LC: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
    {
        let a = a(LinearCombination::zero());
        let b = b(LinearCombination::zero());
        let c = c(LinearCombination::zero());

        self.a.push(Scalar(eval(
            &a,
            // Inputs have full density in the A query
            // because there are constraints of the
            // form x * 0 = 0 for each input.
            None,
            Some(&mut self.a_aux_density),
            &self.input_assignment,
            &self.aux_assignment,
        )));
        self.b.push(Scalar(eval(
            &b,
            Some(&mut self.b_input_density),
            Some(&mut self.b_aux_density),
            &self.input_assignment,
            &self.aux_assignment,
        )));
        self.c.push(Scalar(eval(
            &c,
            // There is no C polynomial query,
            // though there is an (beta)A + (alpha)B + C
            // query for all aux variables.
            // However, that query has full density.
            None,
            None,
            &self.input_assignment,
            &self.aux_assignment,
        )));
    }

    fn push_namespace<NR, N>(&mut self, _: N)
        where
            NR: Into<String>,
            N: FnOnce() -> NR,
    {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn pop_namespace(&mut self) {
        // Do nothing; we don't care about namespaces in this context.
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }

    fn is_extensible() -> bool {
        true
    }

    fn extend(&mut self, other: Self) {
        self.a_aux_density.extend(other.a_aux_density, false);
        self.b_input_density.extend(other.b_input_density, true);
        self.b_aux_density.extend(other.b_aux_density, false);

        self.a.extend(other.a);
        self.b.extend(other.b);
        self.c.extend(other.c);

        self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
        self.aux_assignment.extend(other.aux_assignment);
    }
}

pub fn create_proof_batch<E, C, P: ParameterSource<E>>(
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

    let input_len = provers[0].input_assignment.len();
    let vk = params.get_vk(input_len)?;
    let n = provers[0].a.len();

    // Make sure all circuits have the same input len.
    provers.par_iter().for_each(|prover| {
        assert_eq!(
            prover.a.len(),
            n,
            "only equally sized circuits are supported"
        );
    });

    info!("starting FFT phase");
    let fft_start = Instant::now();

    let a_s = provers
        .par_iter_mut()
        .map(|prover| {
            let mut a =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.a, Vec::new()))?;
            let mut b =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.b, Vec::new()))?;
            let mut c =
                EvaluationDomain::from_coeffs(std::mem::replace(&mut prover.c, Vec::new()))?;

            let mut coeff = vec![&mut a, &mut b, &mut c];
            let indices = vec![0, 1, 2];

            indices.par_iter()
                .zip(coeff.par_iter_mut())
                .for_each(|(i, v)| {
                if *i == 2 {
                    v.ifft(Some(&DEVICE_POOL)).unwrap();
                }
                else{
                    v.ifft(Some(&DEVICE_POOL)).unwrap();
                    v.coset_fft(Some(&DEVICE_POOL)).unwrap();
                }
            });

            a.mul_assign(&b, Some(&DEVICE_POOL))?;
            drop(b);
            //a.sub_assign(&c, Some(&DEVICE_POOL))?;
            //drop(c);
            //a.divide_by_z_on_coset();
            a.icoset_fft(Some(&DEVICE_POOL))?;
            a.sub_assign(&c, Some(&DEVICE_POOL))?;
            drop(c);
            a.divide_by_z_on_coset();
            let mut a = a.into_coeffs();
            let a_len = a.len() - 1;
            a.truncate(a_len);

            Ok(Arc::new(
                a.par_iter().map(|s| s.0.into_repr()).collect::<Vec<_>>(),
            ))
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    let fft_time = fft_start.elapsed();
    info!("FFT phase time: {:?}", fft_time);

    info!("starting multiexp phase");
    let multiexp_start = Instant::now();

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

    let h_s_l_s = a_s
        .par_iter()
        .zip(aux_assignments.par_iter())
        .map(|(a, aux_assignment)| {
            (
                multiexp(
                    params.get_h(a.len()).unwrap(),
                    FullDensity,
                    a.clone(),
                    Some(&DEVICE_POOL)),
                multiexp(
                    params.get_l(aux_assignment.len()).unwrap(),
                    FullDensity,
                    aux_assignment.clone(),
                    Some(&DEVICE_POOL))
            )
        }).collect::<Vec<_>>();

    let inputs = provers
        .par_iter()
        .zip(input_assignments.par_iter())
        .zip(aux_assignments.par_iter())
        .map(|((prover, input_assignment), aux_assignment)| {
            let a_aux_density_total = prover.a_aux_density.get_total_density();

            let (a_inputs_source, a_aux_source) =
                params.get_a(input_assignment.len(), a_aux_density_total)?;

            let a_inputs = multiexp(
                a_inputs_source,
                FullDensity,
                input_assignment.clone(),
                Some(&DEVICE_POOL),
            );

            let a_aux = multiexp(
                a_aux_source,
                Arc::new(prover.a_aux_density.clone()),
                aux_assignment.clone(),
                Some(&DEVICE_POOL),
            );

            let b_input_density = Arc::new(prover.b_input_density.clone());
            let b_input_density_total = b_input_density.get_total_density();
            let b_aux_density = Arc::new(prover.b_aux_density.clone());
            let b_aux_density_total = b_aux_density.get_total_density();

            let (b_g2_inputs_source, b_g2_aux_source) =
                params.get_b_g2(b_input_density_total, b_aux_density_total)?;

            let b_g2_inputs = multiexp(
                b_g2_inputs_source,
                b_input_density,
                input_assignment.clone(),
                Some(&DEVICE_POOL),
            );
            let b_g2_aux = multiexp(
                b_g2_aux_source,
                b_aux_density,
                aux_assignment.clone(),
                Some(&DEVICE_POOL),
            );

            Ok((
                a_inputs,
                a_aux,
                b_g2_inputs,
                b_g2_aux,
            ))
        })
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    let multiexp_time = multiexp_start.elapsed();
    info!("multiexp phase time: {:?}", multiexp_time);

    let proofs = h_s_l_s.into_par_iter()
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::bls::{Bls12, Fr};
    use rand::Rng;
    use rand_core::SeedableRng;
    use rand_xorshift::XorShiftRng;

    #[test]
    fn test_proving_assignment_extend() {
        let mut rng = XorShiftRng::from_seed([
            0x59, 0x62, 0xbe, 0x5d, 0x76, 0x3d, 0x31, 0x8d, 0x17, 0xdb, 0x37, 0x32, 0x54, 0x06,
            0xbc, 0xe5,
        ]);

        for k in &[2, 4, 8] {
            for j in &[10, 20, 50] {
                let count: usize = k * j;

                let mut full_assignment = ProvingAssignment::<Bls12>::new();
                full_assignment
                    .alloc_input(|| "one", || Ok(Fr::one()))
                    .unwrap();

                let mut partial_assignments = Vec::with_capacity(count / k);
                for i in 0..count {
                    if i % k == 0 {
                        let mut p = ProvingAssignment::new();
                        p.alloc_input(|| "one", || Ok(Fr::one())).unwrap();
                        partial_assignments.push(p)
                    }

                    let index: usize = i / k;
                    let partial_assignment = &mut partial_assignments[index];

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el.clone()))
                            .unwrap();
                        partial_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    if rng.gen() {
                        let el = Fr::random(&mut rng);
                        full_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el.clone()))
                            .unwrap();
                        partial_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el))
                            .unwrap();
                    }

                    // TODO: LinearCombination
                }

                let mut combined = ProvingAssignment::new();
                combined.alloc_input(|| "one", || Ok(Fr::one())).unwrap();

                for assignment in partial_assignments.into_iter() {
                    combined.extend(assignment);
                }
                assert_eq!(combined, full_assignment);
            }
        }
    }
}
