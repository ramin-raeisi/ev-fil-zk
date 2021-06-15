use std::sync::{Arc};
use std::time::Instant;

use crate::bls::Engine;
use ff::{Field, PrimeField};
use groupy::{CurveAffine, CurveProjective};
use rayon::prelude::*;

use super::{ParameterGetter, Proof};
use crate::domain::{EvaluationDomain, Scalar};
use crate::multiexp::{multiexp, multiexp_skipdensity, density_filter, DensityTracker};
use crate::{
    Circuit, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use futures::future::Future;
use log::{info, error};
use crate::gpu::{DEVICE_POOL};

mod prover_preload;

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

#[derive(Clone)]
pub struct ProvingAssignment<E: Engine> {
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

    fn extend_from_element(&mut self, other: Self, unit: &Self){
        self.b_input_density.extend_from_element(other.b_input_density, &unit.b_input_density);
        self.a_aux_density.extend_from_element(other.a_aux_density, &unit.a_aux_density);
        self.b_aux_density.extend_from_element(other.b_aux_density, &unit.b_aux_density);

        if other.a.len() > unit.a.len() {
            self.a.extend(&other.a[unit.a.len()..]);
        }
        if other.b.len() > unit.b.len() {
            self.b.extend(&other.b[unit.b.len()..]);
        }
        if other.c.len() > unit.c.len() {
            self.c.extend(&other.c[unit.c.len()..]);
        }
        if other.input_assignment.len() > unit.input_assignment.len() {
            self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[unit.input_assignment.len()..]);
        }
        if other.aux_assignment.len() > unit.aux_assignment.len() {
            self.aux_assignment.extend(&other.aux_assignment[unit.aux_assignment.len()..]);
        }

    }

    fn make_vector(&self, size: usize) -> Result<Vec<Self::Root>, SynthesisError> {
        let mut res = Vec::new();
        for _ in 0..size {
            let mut new_cs = Self::new();
            new_cs.alloc_input(|| "", || Ok(E::Fr::one()))?; // each CS has one
            res.push(new_cs);
        }
        Ok(res)
    }

    fn make_vector_copy(&self, size: usize) -> Result<Vec<Self::Root>, SynthesisError> {
        let mut res = Vec::new();
        for _ in 0..size {
            let mut new_cs = Self::new();
            new_cs.a_aux_density = self.a_aux_density.clone();
            new_cs.b_input_density = self.b_input_density.clone();
            new_cs.b_aux_density = self.b_aux_density.clone();
            new_cs.a = self.a.clone();
            new_cs.b = self.b.clone();
            new_cs.c = self.c.clone();
            new_cs.input_assignment = self.input_assignment.clone();
            new_cs.aux_assignment = self.aux_assignment.clone();
            res.push(new_cs);
        }
        Ok(res)
    }
    fn make_copy(&self) -> Result<Self::Root, SynthesisError> {
            let mut new_cs = Self::new();
            new_cs.a_aux_density = self.a_aux_density.clone();
            new_cs.b_input_density = self.b_input_density.clone();
            new_cs.b_aux_density = self.b_aux_density.clone();
            new_cs.a = self.a.clone();
            new_cs.b = self.b.clone();
            new_cs.c = self.c.clone();
            new_cs.input_assignment = self.input_assignment.clone();
            new_cs.aux_assignment = self.aux_assignment.clone();
        Ok(new_cs)
    }


    fn aggregate(&mut self, other: Vec<Self::Root>) {
        for cs in other {
            self.extend(cs);
        }
    }

    fn aggregate_element(&mut self, other: Self::Root) {
        self.extend(other);
    }

    fn part_aggregate_element(&mut self, other: Self::Root, unit: &Self::Root) {
        self.extend_from_element(other, unit);
    }

    fn align_variable(&mut self, v: &mut Variable, input_shift: usize, aux_shift: usize,) {
        match v {
            Variable(Index::Input(_i)) => {
                *v = Variable(Index::Input(input_shift));

            }
            Variable(Index::Aux(_i)) => {
                *v = Variable(Index::Aux(aux_shift));

            }
        }
    }

    fn get_aux_assigment_len(&mut self,) -> usize {
        self.aux_assignment.len()
    }

    fn get_input_assigment_len(&mut self,) -> usize {
        self.input_assignment.len()
    }

    fn get_index(&mut self, v: &mut Variable,) -> usize {
        match v {
            Variable(Index::Input(i)) => {
                *i

            }
            Variable(Index::Aux(i)) => {
                *i

            }
        }
    }

    fn deallocate(&mut self, v: Variable) -> Result<(), SynthesisError> {
        match v {
            Variable(Index::Input(i)) => {
                self.input_assignment.remove(i);
                self.b_input_density.deallocate(i);
            }
            Variable(Index::Aux(i)) => {
                self.aux_assignment.remove(i);
                self.a_aux_density.deallocate(i);
                self.b_aux_density.deallocate(i);
            }
        }

        Ok(())
    }

    fn set_var_density(&mut self, v: Variable, density_value: bool) -> Result<(), SynthesisError> {
        match v {
            Variable(Index::Input(i)) => {
                self.b_input_density.set_var_density(i, density_value);
            }
            Variable(Index::Aux(i)) => {
                self.a_aux_density.set_var_density(i, density_value);
                self.b_aux_density.set_var_density(i, density_value);
            }
        }
        Ok(())
    }
}

pub fn create_proof_batch<E, C, P: ParameterGetter<E>>(
    circuits: Vec<C>,
    params: P,
) -> Result<Vec<Proof<E>>, SynthesisError>
    where
        E: Engine,
        C: Circuit<E> + Send,
{

    use crate::groth16::prover::prover_preload::create_proof_batch_preload;

    let params_preload = std::env::var("FIL_ZK_PARAMS_PRELOAD")
        .and_then(|v| match v.parse() {
            Ok(val) => Ok(val),
            Err(_) => {
                error!("Invalid FIL_ZK_PARAMS_PRELOAD! Defaulting to {}", 0);
                Ok(0)
            }
        })
        .unwrap_or(0);

    if params_preload == 0 {
        return create_proof_batch_inner(circuits, params);
    }
    create_proof_batch_preload(circuits, params)
}

fn create_proof_batch_inner<E, C, P: ParameterGetter<E>>(
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
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    let fft_time = fft_start.elapsed();
    info!("FFT phase time: {:?}", fft_time);

    info!("starting multiexp phase");
    let multiexp_start = Instant::now();

    info!("h_s");
    let h_base = params.get_h().unwrap();
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

    info!("aux_assignments");	
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
    
    info!("l_s");
    let l_base = params.get_l().unwrap();
    let l_skip = 0;
    let aux_assignments_arc = Arc::new(&aux_assignments);
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


    info!("input_assignments");	
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

    info!("inputs");

    let a_base = params.get_a().unwrap();
    let b_g2_base = params.get_b_g2().unwrap();

    let inputs = provers
        .par_iter()
        .zip(input_assignments.par_iter())
        .zip(aux_assignments.par_iter())
        .map(|((prover, input_assignment), aux_assignment)| {
            //let a_aux_density_total = prover.a_aux_density.get_total_density();

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
        .collect::<Result<Vec<_>, SynthesisError>>()?;

    let multiexp_time = multiexp_start.elapsed();
    info!("multiexp phase time: {:?}", multiexp_time);

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

                let mut base_partial = ProvingAssignment::<Bls12>::new();

                let mut base_partial2 = ProvingAssignment::<Bls12>::new();

                let provers = vec![&mut full_assignment, &mut base_partial];
                let x = Fr::from_str("5").unwrap();
                let mut x_sqr = x.clone();
                x_sqr.mul_assign(&x);
                for p in provers {
                    p.alloc_input(|| "one", || Ok(Fr::one()))
                        .unwrap();
                    let x_var = p.alloc_input(|| "x_val", || Ok(x.clone())).unwrap();
                    let x_sqr_var = p.alloc(|| "x_sqr", || Ok(x_sqr.clone())).unwrap();
                    p.enforce(|| "x_squaring", |lc| lc + x_var, |lc| lc + x_var, |lc| lc + x_sqr_var);
                }
                
                let mut partial_assignments = base_partial.make_vector(count / k).unwrap();
                let mut partial_assignments2 = base_partial.make_vector(count / k).unwrap();

                let mut parents = Vec::with_capacity(count);
                for i in 0..count {
                    let index: usize = i / k;
                    let partial_assignment = &mut partial_assignments[index];
                    let partial_assignment2 = &mut partial_assignments2[index];
                    let c = Fr::from_str("7").unwrap();
                    let com = partial_assignment
                            .alloc_input(|| format!("alloc_input comm"), || Ok(c.clone()))
                            .unwrap();
                            let com = partial_assignment2
                            .alloc_input(|| format!("alloc_input comm"), || Ok(c.clone()))
                            .unwrap();

                    // take a random element, dobule it and verify results
                    if rng.gen() {
                        let el = Fr::random(&mut rng);

                        let el_var_ful = full_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el.clone()))
                            .unwrap();
                        let el_var_part = partial_assignment
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el.clone()))
                            .unwrap();
                            let el_var_part = partial_assignment2
                            .alloc_input(|| format!("alloc_input:{},{}", i, k), || Ok(el.clone()))
                            .unwrap();
                        parents.push(el_var_part);

                        let mut el_double = el.clone();
                        el_double.add_assign(&el);

                        let el_double_var_ful = full_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el_double.clone()))
                            .unwrap();
                        let el_double_var_part = partial_assignment
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el_double.clone()))
                            .unwrap();
                            let el_double_var_part = partial_assignment2
                            .alloc(|| format!("alloc:{},{}", i, k), || Ok(el_double.clone()))
                            .unwrap();

                        full_assignment.enforce(|| "el_double", |lc| lc + el_var_ful + el_var_ful, |lc| lc + ProvingAssignment::<Bls12>::one(), |lc| lc + el_double_var_ful);
                        partial_assignment.enforce(|| "el_double", |lc| lc + el_var_part + el_var_part, |lc| lc + ProvingAssignment::<Bls12>::one(), |lc| lc + el_double_var_part);
                        partial_assignment2.enforce(|| "el_double", |lc| lc + el_var_part + el_var_part, |lc| lc + ProvingAssignment::<Bls12>::one(), |lc| lc + el_double_var_part);
                        parents.push(el_double_var_part);
                    }
                    partial_assignment.deallocate(com).unwrap();
                    partial_assignment2.deallocate(com).unwrap();
                }

                let y = Fr::from_str("4").unwrap();
                let y_var_ful = full_assignment.alloc(|| "y", || Ok(y.clone())).unwrap();
                let pa_n = partial_assignments.len();
                let mut y_var_part = partial_assignments[pa_n - 1].alloc(|| "y", || Ok(y.clone())).unwrap();

                let z = Fr::from_str("2").unwrap();
                let z_var_ful = full_assignment.alloc_input(|| "z", || Ok(z.clone())).unwrap();
                let mut z_var_part = partial_assignments[pa_n - 1].alloc_input(|| "z", || Ok(z.clone())).unwrap();

                let last_partial = partial_assignments.split_off(pa_n - 1);

                for (i, other_cs) in partial_assignments.into_iter().enumerate() {
                    base_partial.align_variable(&mut parents[i], 4, 1);
                    base_partial.aggregate(vec![other_cs]); // aggregate all CSs exept the last one
                }

                for (i, other_cs) in partial_assignments2.into_iter().enumerate() {
                    base_partial.align_variable(&mut parents[i], 1, 0);
                    base_partial2.aggregate(vec![other_cs]); // aggregate all CSs exept the last one
                }

                base_partial.align_variable(&mut y_var_part, 1, 0); // align variables form the last CS
                base_partial.align_variable(&mut z_var_part, 1, 0);
                base_partial.aggregate(last_partial);

                full_assignment.enforce(|| "y_enforce", |lc| lc + y_var_ful, |lc| lc + z_var_ful, |lc| lc + (Fr::from_str("8").unwrap(), ProvingAssignment::<Bls12>::one()));
                base_partial.enforce(|| "y_enforce", |lc| lc + y_var_part, |lc| lc + z_var_part, |lc| lc + (Fr::from_str("8").unwrap(), ProvingAssignment::<Bls12>::one()));
                //for j in 0..3 {
                //    full_assignment.enforce(|| "y_enforce", |lc| lc + parents[j] , |lc| lc + parents[count + j], |lc| lc + (Fr::from_str("8").unwrap(), ProvingAssignment::<Bls12>::one()));
                    //base_partial.enforce(|| "y_enforce", |lc| lc + parents[j], |lc| lc + parents[count + j], |lc| lc + (Fr::from_str("8").unwrap(), ProvingAssignment::<Bls12>::one()));
               // } 
                assert_eq!(full_assignment, base_partial);
            }
        }
    }
}
