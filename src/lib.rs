//! `bellperson` is a crate for building zk-SNARK circuits. It provides circuit
//! traits and and primitive structures, as well as basic gadget implementations
//! such as booleans and number abstractions.
//!
//! # Example circuit
//!
//! Say we want to write a circuit that proves we know the preimage to some hash
//! computed using SHA-256d (calling SHA-256 twice). The preimage must have a
//! fixed length known in advance (because the circuit parameters will depend on
//! it), but can otherwise have any value. We take the following strategy:
//!
//! - Witness each bit of the preimage.
//! - Compute `hash = SHA-256d(preimage)` inside the circuit.
//! - Expose `hash` as a public input using multiscalar packing.
//!
//! ```
//! use bellperson::{
//!     gadgets::{
//!         boolean::{AllocatedBit, Boolean},
//!         multipack,
//!         sha256::sha256,
//!     },
//!     groth16, Circuit, ConstraintSystem, SynthesisError,
//!     bls::{Bls12, Engine}
//! };
//! use rand::rngs::OsRng;
//! use sha2::{Digest, Sha256};
//!
//! /// Our own SHA-256d gadget. Input and output are in little-endian bit order.
//! fn sha256d<E: Engine, CS: ConstraintSystem<E>>(
//!     mut cs: CS,
//!     data: &[Boolean],
//! ) -> Result<Vec<Boolean>, SynthesisError> {
//!     // Flip endianness of each input byte
//!     let input: Vec<_> = data
//!         .chunks(8)
//!         .map(|c| c.iter().rev())
//!         .flatten()
//!         .cloned()
//!         .collect();
//!
//!     let mid = sha256(cs.namespace(|| "SHA-256(input)"), &input)?;
//!     let res = sha256(cs.namespace(|| "SHA-256(mid)"), &mid)?;
//!
//!     // Flip endianness of each output byte
//!     Ok(res
//!         .chunks(8)
//!         .map(|c| c.iter().rev())
//!         .flatten()
//!         .cloned()
//!         .collect())
//! }
//!
//! struct MyCircuit {
//!     /// The input to SHA-256d we are proving that we know. Set to `None` when we
//!     /// are verifying a proof (and do not have the witness data).
//!     preimage: Option<[u8; 80]>,
//! }
//!
//! impl<E: Engine> Circuit<E> for MyCircuit {
//!     fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
//!         // Compute the values for the bits of the preimage. If we are verifying a proof,
//!         // we still need to create the same constraints, so we return an equivalent-size
//!         // Vec of None (indicating that the value of each bit is unknown).
//!         let bit_values = if let Some(preimage) = self.preimage {
//!             preimage
//!                 .iter()
//!                 .map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
//!                 .flatten()
//!                 .map(|b| Some(b))
//!                 .collect()
//!         } else {
//!             vec![None; 80 * 8]
//!         };
//!         assert_eq!(bit_values.len(), 80 * 8);
//!
//!         // Witness the bits of the preimage.
//!         let preimage_bits = bit_values
//!             .into_iter()
//!             .enumerate()
//!             // Allocate each bit.
//!             .map(|(i, b)| {
//!                 AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {}", i)), b)
//!             })
//!             // Convert the AllocatedBits into Booleans (required for the sha256 gadget).
//!             .map(|b| b.map(Boolean::from))
//!             .collect::<Result<Vec<_>, _>>()?;
//!
//!         // Compute hash = SHA-256d(preimage).
//!         let hash = sha256d(cs.namespace(|| "SHA-256d(preimage)"), &preimage_bits)?;
//!
//!         // Expose the vector of 32 boolean variables as compact public inputs.
//!         multipack::pack_into_inputs(cs.namespace(|| "pack hash"), &hash)
//!     }
//! }
//!
//! // Create parameters for our circuit. In a production deployment these would
//! // be generated securely using a multiparty computation.
//! let params = {
//!     let c = MyCircuit { preimage: None };
//!     groth16::generate_random_parameters::<Bls12, _, _>(c, &mut OsRng).unwrap()
//! };
//!
//! // Prepare the verification key (for proof verification).
//! let pvk = groth16::prepare_verifying_key(&params.vk);
//!
//! // Pick a preimage and compute its hash.
//! let preimage = [42; 80];
//! let hash = Sha256::digest(&Sha256::digest(&preimage));
//!
//! // Create an instance of our circuit (with the preimage as a witness).
//! let c = MyCircuit {
//!     preimage: Some(preimage),
//! };
//!
//! // Create a Groth16 proof with our parameters.
//! let proof = groth16::create_random_proof(c, &params, &mut OsRng).unwrap();
//!
//! // Pack the hash as inputs for proof verification.
//! let hash_bits = multipack::bytes_to_bits_le(&hash);
//! let inputs = multipack::compute_multipacking::<Bls12>(&hash_bits);
//!
//! // Check the proof!
//! assert!(groth16::verify_proof(&pvk, &proof, &inputs).unwrap());
//! ```
//!
//! # Roadmap
//!
//! `bellperson` is being refactored into a generic proving library. Currently it
//! is pairing-specific, and different types of proving systems need to be
//! implemented as sub-modules. After the refactor, `bellperson` will be generic
//! using the [`ff`] and [`group`] crates, while specific proving systems will
//! be separate crates that pull in the dependencies they require.

// Requires nightly for aarch64
#![cfg_attr(target_arch = "aarch64", feature(stdsimd))]

#[cfg(test)]
#[macro_use]
extern crate hex_literal;

pub mod bls;
pub mod domain;
pub mod gadgets;
pub mod gpu;
#[cfg(feature = "groth16")]
pub mod groth16;
pub mod multiexp;
pub mod settings;

pub mod util_cs;
use ff::{Field, ScalarEngine};

use rustc_hash::FxHashMap as HashMap;
use std::io;
use std::marker::PhantomData;
use std::ops::{Add, Sub};

/// Computations are expressed in terms of arithmetic circuits, in particular
/// rank-1 quadratic constraint systems. The `Circuit` trait represents a
/// circuit that can be synthesized. The `synthesize` method is called during
/// CRS generation and during proving.
pub trait Circuit<E: ScalarEngine> {
    /// Synthesize the circuit into a rank-1 quadratic constraint system.
    fn synthesize<CS: ConstraintSystem<E>>(self, cs: &mut CS) -> Result<(), SynthesisError>;
}

/// Represents a variable in our constraint system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Variable(Index);

impl Variable {
    /// This constructs a variable with an arbitrary index.
    /// Circuit implementations are not recommended to use this.
    pub fn new_unchecked(idx: Index) -> Variable {
        Variable(idx)
    }

    /// This returns the index underlying the variable.
    /// Circuit implementations are not recommended to use this.
    pub fn get_unchecked(&self) -> Index {
        self.0
    }
}

/// Represents the index of either an input variable or
/// auxiliary variable.
#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
pub enum Index {
    Input(usize),
    Aux(usize),
}

/// This represents a linear combination of some variables, with coefficients
/// in the scalar field of a pairing-friendly elliptic curve group.
#[derive(Clone)]
pub struct LinearCombination<E: ScalarEngine>(HashMap<Variable, E::Fr>);
impl<E: ScalarEngine> Default for LinearCombination<E> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<E: ScalarEngine> LinearCombination<E> {
    pub fn zero() -> LinearCombination<E> {
        LinearCombination(HashMap::default())
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Variable, &E::Fr)> + '_ {
        self.0.iter()
    }

    pub fn add_unsimplified(mut self, (coeff, var): (E::Fr, Variable)) -> LinearCombination<E> {
        self.0
            .entry(var)
            .or_insert(E::Fr::zero())
            .add_assign(&coeff);

        self
    }
}

impl<E: ScalarEngine> Add<(E::Fr, Variable)> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn add(mut self, (coeff, var): (E::Fr, Variable)) -> LinearCombination<E> {
        self.0
            .entry(var)
            .or_insert(E::Fr::zero())
            .add_assign(&coeff);

        self
    }
}

impl<E: ScalarEngine> Sub<(E::Fr, Variable)> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn sub(self, (mut coeff, var): (E::Fr, Variable)) -> LinearCombination<E> {
        coeff.negate();

        self + (coeff, var)
    }
}

impl<E: ScalarEngine> Add<Variable> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn add(self, other: Variable) -> LinearCombination<E> {
        self + (E::Fr::one(), other)
    }
}

impl<E: ScalarEngine> Sub<Variable> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn sub(self, other: Variable) -> LinearCombination<E> {
        self - (E::Fr::one(), other)
    }
}

impl<'a, E: ScalarEngine> Add<&'a LinearCombination<E>> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn add(mut self, other: &'a LinearCombination<E>) -> LinearCombination<E> {
        for (var, val) in &other.0 {
            self.0.entry(*var).or_insert(E::Fr::zero()).add_assign(val);
        }

        self
    }
}

impl<'a, E: ScalarEngine> Sub<&'a LinearCombination<E>> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn sub(mut self, other: &'a LinearCombination<E>) -> LinearCombination<E> {
        for (var, val) in &other.0 {
            self = self - (*val, *var);
        }

        self
    }
}

impl<'a, E: ScalarEngine> Add<(E::Fr, &'a LinearCombination<E>)> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn add(mut self, (coeff, other): (E::Fr, &'a LinearCombination<E>)) -> LinearCombination<E> {
        for s in &other.0 {
            let mut tmp = *s.1;
            tmp.mul_assign(&coeff);
            self = self + (tmp, *s.0);
        }

        self
    }
}

impl<'a, E: ScalarEngine> Sub<(E::Fr, &'a LinearCombination<E>)> for LinearCombination<E> {
    type Output = LinearCombination<E>;

    fn sub(mut self, (coeff, other): (E::Fr, &'a LinearCombination<E>)) -> LinearCombination<E> {
        for s in &other.0 {
            let mut tmp = *s.1;
            tmp.mul_assign(&coeff);
            self = self - (tmp, *s.0);
        }

        self
    }
}

/// This is an error that could occur during circuit synthesis contexts,
/// such as CRS generation, proving or verification.
#[derive(thiserror::Error, Debug)]
pub enum SynthesisError {
    /// During synthesis, we lacked knowledge of a variable assignment.
    #[error("an assignment for a variable could not be computed")]
    AssignmentMissing,
    /// During synthesis, we divided by zero.
    #[error("division by zero")]
    DivisionByZero,
    /// During synthesis, we constructed an unsatisfiable constraint system.
    #[error("unsatisfiable constraint system")]
    Unsatisfiable,
    /// During synthesis, our polynomials ended up being too high of degree
    #[error("polynomial degree is too large")]
    PolynomialDegreeTooLarge,
    /// During proof generation, we encountered an identity in the CRS
    #[error("encountered an identity element in the CRS")]
    UnexpectedIdentity,
    /// During proof generation, we encountered an I/O error with the CRS
    #[error("encountered an I/O error: {0}")]
    IoError(#[from] io::Error),
    /// During verification, our verifying key was malformed.
    #[error("malformed verifying key")]
    MalformedVerifyingKey,
    /// During CRS generation, we observed an unconstrained auxiliary variable
    #[error("auxiliary variable was unconstrained")]
    UnconstrainedVariable,
    /// During GPU multiexp/fft, some GPU related error happened
    #[error("encountered a GPU error: {0}")]
    GPUError(#[from] gpu::GPUError),
}

/// Represents a constraint system which can have new variables
/// allocated and constrains between them formed.
pub trait ConstraintSystem<E: ScalarEngine>: Sized + Send {
    /// Represents the type of the "root" of this constraint system
    /// so that nested namespaces can minimize indirection.
    type Root: ConstraintSystem<E>;

    fn new() -> Self {
        unimplemented!(
            "ConstraintSystem::new must be implemented for extensible types implementing ConstraintSystem"
        );
    }

    fn clone(&self) -> Self{
        panic!("parallel functional (fn clone) in not implemented for {}", std::any::type_name::<Self>())
    }

    /// Return the "one" input variable
    fn one() -> Variable {
        Variable::new_unchecked(Index::Input(0))
    }

    /// Allocate a private variable in the constraint system. The provided function is used to
    /// determine the assignment of the variable. The given `annotation` function is invoked
    /// in testing contexts in order to derive a unique name for this variable in the current
    /// namespace.
    fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>;

    /// Allocate a public variable in the constraint system. The provided function is used to
    /// determine the assignment of the variable.
    fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>;

    /// Enforce that `A` * `B` = `C`. The `annotation` function is invoked in testing contexts
    /// in order to derive a unique name for the constraint in the current namespace.
    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LB: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LC: FnOnce(LinearCombination<E>) -> LinearCombination<E>;

    /// Create a new (sub)namespace and enter into it. Not intended
    /// for downstream use; use `namespace` instead.
    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR;

    /// Exit out of the existing namespace. Not intended for
    /// downstream use; use `namespace` instead.
    fn pop_namespace(&mut self);

    /// Gets the "root" constraint system, bypassing the namespacing.
    /// Not intended for downstream use; use `namespace` instead.
    fn get_root(&mut self) -> &mut Self::Root;

    /// Begin a namespace for this constraint system.
    fn namespace<NR, N>(&mut self, name_fn: N) -> Namespace<'_, E, Self::Root>
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.get_root().push_namespace(name_fn);

        Namespace(self.get_root(), Default::default())
    }

    /// Most implementations of ConstraintSystem are not 'extensible': they won't implement a specialized
    /// version of `extend` and should therefore also keep the default implementation of `is_extensible`
    /// so callers which optionally make use of `extend` can know to avoid relying on it when unimplemented.
    fn is_extensible() -> bool {
        false
    }

    /// Extend concatenates thew  `other` constraint systems to the receiver, modifying the receiver, whose
    /// inputs, allocated variables, and constraints will precede those of the `other` constraint system.
    /// The primary use case for this is parallel synthesis of circuits which can be decomposed into
    /// entirely independent sub-circuits. Each can be synthesized in its own thread, then the
    /// original `ConstraintSystem` can be extended with each, in the same order they would have
    /// been synthesized sequentially.
    fn extend(&mut self, _other: Self) {
        unimplemented!(
            "ConstraintSystem::extend must be implemented for types implementing ConstraintSystem"
        );
    }

    fn extend_without_inputs(&mut self, _other: Self) {
        unimplemented!(
            "ConstraintSystem::extend_without_inputs must be implemented for types implementing ConstraintSystem"
        );
    }

    fn extend_from_element(&mut self, mut _other: Self, _unit: &Self) {
        unimplemented!(
            "ConstraintSystem::extend_from_element must be implemented for types implementing ConstraintSystem"
        );
    }

    // Create a vector of CSs with the same porperties
    // Later these CSs can be aggregated to the one
    // It allows to calculate synthesize-functions in parallel for several copies of the CS and later aggregate them
    fn make_vector(&self, _size: usize) -> Result<Vec<Self::Root>, SynthesisError> {
        panic!("parallel functional (fn make_vector) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn make_vector_copy(&self, _size: usize) -> Result<Vec<Self::Root>, SynthesisError> {
        panic!("parallel functional (fn make_vector) in not implemented for {}", std::any::type_name::<Self>())
    }

    // Aggregate all data from other to self
    fn aggregate(&mut self, _other: Vec<Self::Root>) {
        panic!("parallel functional (fn aggregate) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn aggregate_element(&mut self, _other: Self::Root) {
        panic!("parallel functional (fn aggregate_element) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn part_aggregate_element(&mut self, mut _other: Self::Root, unit: &Self::Root) {
        panic!("parallel functional (fn part_aggregate_element) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn aggregate_without_inputs(&mut self, _other: Vec<Self::Root>) {
        panic!("parallel functional (fn aggregate_without_inputs) in not implemented for {}", std::any::type_name::<Self>())
    }

    /*fn part_aggregate(&mut self, mut _other: Vec<Self::Root>, _unit:Vec<Self::Root>) {
        panic!("parallel functional (fn part_aggregate) in not implemented for {}", std::any::type_name::<Self>())
    }*/

    fn align_variable(&mut self, _v: &mut Variable, _input_shift: usize, _aux_shift: usize) {
        panic!("parallel functional (fn align_variable) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn get_aux_assigment_len(&mut self,) -> usize{
        panic!("parallel functional (fn get_aux_assigment_len) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn get_input_assigment_len(&mut self,) -> usize{
        panic!("parallel functional (fn get_input_assigment_len) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn get_index(&mut self, v: &mut Variable,) -> usize{
        panic!("parallel functional (fn get_index) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn print_index(&mut self, _v: &mut Variable) {
        panic!("parallel functional (fn print_index) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn deallocate(&mut self, _v: Variable) -> Result<(), SynthesisError> {
        panic!("parallel functional (fn deallocate) in not implemented for {}", std::any::type_name::<Self>())
    }

    fn set_var_density(&mut self, _v: Variable, _density_value: bool) -> Result<(), SynthesisError> {
        panic!("parallel functional (fn set_var_density) in not implemented for {}", std::any::type_name::<Self>())
    }
}

/// This is a "namespaced" constraint system which borrows a constraint system (pushing
/// a namespace context) and, when dropped, pops out of the namespace context.
pub struct Namespace<'a, E: ScalarEngine, CS: ConstraintSystem<E>>(&'a mut CS, SendMarker<E>);

struct SendMarker<E: ScalarEngine>(PhantomData<E>);

impl<E: ScalarEngine> Default for SendMarker<E> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

// Safety: ScalarEngine is static and this is only a marker
unsafe impl<E: ScalarEngine> Send for SendMarker<E> {}

impl<'cs, E: ScalarEngine, CS: ConstraintSystem<E>> ConstraintSystem<E> for Namespace<'cs, E, CS> {
    type Root = CS::Root;

    fn one() -> Variable {
        CS::one()
    }

    fn clone(&self) -> Self {
        self.clone()
    }

    fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.0.alloc(annotation, f)
    }

    fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.0.alloc_input(annotation, f)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LB: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LC: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
    {
        self.0.enforce(annotation, a, b, c)
    }

    // Downstream users who use `namespace` will never interact with these
    // functions and they will never be invoked because the namespace is
    // never a root constraint system.

    fn push_namespace<NR, N>(&mut self, _: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        panic!("only the root's push_namespace should be called");
    }

    fn pop_namespace(&mut self) {
        panic!("only the root's pop_namespace should be called");
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self.0.get_root()
    }

    fn make_vector(&self, size: usize) -> Result<Vec<Self::Root>, SynthesisError> {
        self.0.make_vector(size)
    }

    fn make_vector_copy(&self, size: usize) -> Result<Vec<Self::Root>, SynthesisError> {
        self.0.make_vector_copy(size)
    }

    // Aggregate all data from other to self
    fn aggregate(&mut self, other: Vec<Self::Root>) {
        self.0.aggregate(other)
    }

    /*fn part_aggregate(&mut self, mut other: Vec<Self::Root>, unit: Vec<Self::Root>) {
        self.0.part_aggregate(other, unit)
    }*/

    fn aggregate_element(&mut self, other: Self::Root) {
        self.0.aggregate_element(other)
    }

    fn part_aggregate_element(&mut self, mut other: Self::Root, unit: &Self::Root) {
        self.0.part_aggregate_element(other, unit)
    }

    fn aggregate_without_inputs(&mut self, other: Vec<Self::Root>) {
        self.0.aggregate_without_inputs(other)
    }

    fn align_variable(&mut self, v: &mut Variable, input_shift: usize, aux_shift: usize) {
        self.0.align_variable(v, input_shift, aux_shift);
    }

    fn get_aux_assigment_len(&mut self) -> usize {
        self.0.get_aux_assigment_len()
    }

    fn get_input_assigment_len(&mut self) -> usize {
        self.0.get_input_assigment_len()
    }

    fn get_index(&mut self, v: &mut Variable,) -> usize {
        self.0.get_index(v)
    }

    fn print_index(&mut self, v: &mut Variable) {
        self.0.print_index(v);
    }

    fn deallocate(&mut self, v: Variable) -> Result<(), SynthesisError> {
        self.0.deallocate(v)
    }

    fn set_var_density(&mut self, v: Variable, density_value: bool) -> Result<(), SynthesisError> {
        self.0.set_var_density(v, density_value)
    }
}

impl<'a, E: ScalarEngine, CS: ConstraintSystem<E>> Drop for Namespace<'a, E, CS> {
    fn drop(&mut self) {
        self.get_root().pop_namespace()
    }
}

impl<'a, E: ScalarEngine, CS: ConstraintSystem<E>> PartialEq for Namespace<'a, E, CS> {
    fn eq(&self, other: &Namespace<'a, E, CS>) -> bool {
        self.eq(other)
    }
}

/// Convenience implementation of ConstraintSystem<E> for mutable references to
/// constraint systems.
impl<'cs, E: ScalarEngine, CS: ConstraintSystem<E>> ConstraintSystem<E> for &'cs mut CS {
    type Root = CS::Root;

    fn one() -> Variable {
        CS::one()
    }

    fn clone(&self) -> Self{
        (**&self).clone()
    }

    fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        (**self).alloc(annotation, f)
    }

    fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
    where
        F: FnOnce() -> Result<E::Fr, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        (**self).alloc_input(annotation, f)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LB: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
        LC: FnOnce(LinearCombination<E>) -> LinearCombination<E>,
    {
        (**self).enforce(annotation, a, b, c)
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        (**self).push_namespace(name_fn)
    }

    fn pop_namespace(&mut self) {
        (**self).pop_namespace()
    }

    fn get_root(&mut self) -> &mut Self::Root {
        (**self).get_root()
    }

    fn make_vector(&self, size: usize) -> Result<Vec<Self::Root>, SynthesisError> {
        (**self).make_vector(size)
    }

    fn make_vector_copy(&self, size: usize) -> Result<Vec<Self::Root>, SynthesisError> {
        (**self).make_vector_copy(size)
    }

    // Aggregate all data from other to self
    fn aggregate(&mut self, other: Vec<Self::Root>) {
        (**self).aggregate(other)
    }

    /*fn part_aggregate(&mut self, mut other: Vec<Self::Root>, unit: Vec<Self::Root>) {
        (**self).part_aggregate(other, unit)
    }*/

    fn aggregate_element(&mut self, other: Self::Root) {
        (**self).aggregate_element(other)
    }

    fn part_aggregate_element(&mut self, mut other: Self::Root, unit: &Self::Root) {
        (**self).part_aggregate_element(other, unit)
    }

    fn aggregate_without_inputs(&mut self, other: Vec<Self::Root>) {
        (**self).aggregate_without_inputs(other)
    }

    fn align_variable(&mut self, v: &mut Variable, input_shift: usize, aux_shift: usize) {
        (**self).align_variable(v, input_shift, aux_shift);
    }

    fn get_aux_assigment_len(&mut self) -> usize {
        (**self).get_aux_assigment_len()
    }

    fn get_input_assigment_len(&mut self) -> usize {
        (**self).get_input_assigment_len()
    }

    fn get_index(&mut self,  v: &mut Variable) -> usize {
        (**self).get_index(v)
    }

    fn print_index(&mut self, v: &mut Variable) {
        (**self).print_index(v);
    }

    fn deallocate(&mut self, v: Variable) -> Result<(), SynthesisError> {
        (**self).deallocate(v)
    }

    fn set_var_density(&mut self, v: Variable, density_value: bool) -> Result<(), SynthesisError> {
        (**self).set_var_density(v, density_value)
    }
}

#[cfg(all(test, feature = "groth16"))]
mod tests {
    use super::*;

    #[test]
    fn test_add_simplify() {
        use crate::bls::Bls12;

        let n = 5;

        let mut lc = LinearCombination::<Bls12>::zero();

        let mut expected_sums = vec![<Bls12 as ScalarEngine>::Fr::zero(); n];
        let mut total_additions = 0;
        for i in 0..n {
            for _ in 0..i + 1 {
                let coeff = <Bls12 as ScalarEngine>::Fr::one();
                lc = lc + (coeff, Variable::new_unchecked(Index::Aux(i)));
                let mut tmp = expected_sums[i];
                tmp.add_assign(&coeff);
                expected_sums[i] = tmp;
                total_additions += 1;
            }
        }

        // There are only as many terms as distinct variable Indexes — not one per addition operation.
        assert_eq!(n, lc.0.len());
        assert!(lc.0.len() != total_additions);

        // Each variable has the expected coefficient, the sume of those added by its Index.
        lc.0.iter().for_each(|(var, coeff)| match var.0 {
            Index::Aux(i) => assert_eq!(expected_sums[i], *coeff),
            _ => panic!("unexpected variable type"),
        });
    }
}
