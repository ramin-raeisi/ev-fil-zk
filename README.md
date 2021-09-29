# bellperson [![Crates.io](https://img.shields.io/crates/v/bellperson.svg)](https://crates.io/crates/bellperson)

> This is a fork of the great [bellman](https://github.com/zkcrypto/bellman) library.

`bellman` is a crate for building zk-SNARK circuits. It provides circuit traits
and primitive structures, as well as basic gadget implementations such as
booleans and number abstractions.

## Backend

There are currently two backends available for the implementation of Bls12 381:
- [`blstrs`](https://github.com/filecoin-project/blstrs) - optimized with hand tuned assembly, using [blst](https://github.com/supranational/blst)
- [`paired`](https://github.com/filecoin-project/paired) - pure Rust implementation

They can be selected at compile time with the mutually exclusive features `blst` and `pairing`. Specifying one of them is enough for a working library, no additional features need to be set.

The default for now is `blst`, as the secure and audited choice.  Note that `pairing` is deprecated and may be removed in the future.

## GPU

This fork contains GPU parallel acceleration to the FFT and Multiexponentation algorithms in the groth16 prover codebase under the compilation feature `gpu`, it can be used in combination with `blst` or `pairing`.

### Requirements
- NVIDIA or AMD GPU Graphics Driver
- OpenCL

( For AMD devices we recommend [ROCm](https://rocm-documentation.readthedocs.io/en/latest/Installation_Guide/Installation-Guide.html) )

### Environment variables

The gpu extension contains some env vars that may be set externally to this library. 

- `FIL_ZK_CUSTOM_GPU`

    Allows adding a GPU that is not in the tested list. This requires providing the name of the GPU device and the number of cores in the format ["name:cores"].

    ```rust
    // Example
    env::set_var("FIL_ZK_CUSTOM_GPU", "GeForce RTX 2080 Ti:4352, GeForce GTX 1060:1280");
    ```

- `FIL_ZK_CPU_UTILIZATION`

    - Possible values: `[0, 1]` (float)
    - Default value: `0`

    The proportion of the multi-exponentiation calculation (commit2 phase) that will be moved to CPU in parallel to the GPU. For example, 0.1 = 10% of the calculations are proceeding on CPU. It allows keeping all hardware occupied and decreases C2 timings. To get the best performance, the value of the variable must be higher for configuration with good CPU and weak GPU and vice versa. 

    ```rust
    // Example
    env::set_var("FIL_ZK_CPU_UTILIZATION", "0.05");
    ```

- `FIL_ZK_DISABLE_FFT_GPU`

    - Possible values: `0, 1`
    - Default value: `0`

    Defines is GPUs are used during FFT (commit2 phase) or not. FIL_ZK_DISABLE_FFT_GPU=1 uses pure CPU calculations for FFT that increases the overall commit2 phase time. 

    ```rust
    // Example
    env::set_var("FIL_ZK_DISABLE_FFT_GPU", "1");
    ```

- `FIL_ZK_GPU_MEMORY_PADDING`

    - Possible values: `[0, 1]` (float)
    - Default value: `0.1`

    Determines the proportion of free memory, e.g. 0.1 ≅ 10% (not exactly, but close) of free GPU memory during commit2 phase.

    __Important note:__ commit2 phase contains two different algorithms that use GPU: FFT and multi-exponentiation. Currently, `FIL_ZK_GPU_MEMORY_PADDING` restricts only multi-exponentiation algorithm. However, FFT uses less GPU memory than mutli-exponentiation so the variable may be used to make the GPU memory consumption eqaul between these algorithms. 

    ```rust
    // Example
    env::set_var("FIL_ZK_GPU_MEMORY_PADDING", "0.35");
    ```

- `FIL_ZK_P2_GPU_REUSE`

    - Possible values: `[1, 20]`
    - Default value: `1`

    How many instances of P2 can use the same GPU in parallel settings. 

    ```rust
    // Example
    env::set_var("FIL_ZK_P2_GPU_REUSE", "2");
    ```

- `FIL_ZK_PARAMS_PRELOAD`

    - Possible values: `0, 1`
    - Default value: `0`

    Defines the implementation of Groth's SNARK proof that is used in commit2-phase. 1 use the implementation with preloaded data for all SNARK protocol. It increases the amount of used RAM but decreases the proof time. The time-bonus obtained from the preloaded data depends on the hardware and should be tested in practice. 

    ```rust
    // Example
    env::set_var("FIL_ZK_PARAMS_PRELOAD", "1");
    ```

- `FIL_ZK_MAX_WINDOW`

    - Possible values: `[5, 17]` (integer)
    - Default value: `10`
    

Defines the window size for the multi-exponentiation algorithm. A higher value means more serial work in each parallel thread (thus fewer parallel GPU threads in general). 
    
**Important note:** The variable defines the *maximum* window size. The algorithm uses a smaller window size if it's enough for optimal performance.
    
```rust
    // Example
    env::set_var("FIL_ZK_MAX_WINDOW", "12");
```

- `FIL_ZK_WORK_SIZE_MULTIPLIER`

    - Possible values: `[0.1, 10]` (float)
    - Default value: `2`

    Defines the multiplier for GPU load as the number of simultaneous threads. The higher values of the variable - the more threads. 

    ```rust
    // Example
    env::set_var("FIL_ZK_WORK_SIZE_MULTIPLIER", "1.2");
    ```

- `FIL_ZK_CHUNK_SIZE_MULTIPLIER`

    - Possible values: `[1, 10]` (float)
    - Default value: `2`

    Defines the multiplier for GPU memory usage. The higher values of the variable - the more GPU memory is occupied by each thread of the multi-exponentiation algorithm of the commit2 phase.
    Allows controlling the amount of data proceeded by each thread without changing the overall amount of threads (unlike `FIL_ZK_MAX_WINDOW` and `FIL_ZK_WORK_SIZE_MULTIPLIER`). 

    ```rust
    // Example
    env::set_var("FIL_ZK_CHUNK_SIZE_MULTIPLIER", "2.5");
    ```

    

#### Supported / Tested Cards

Depending on the size of the proof being passed to the gpu for work, certain cards will not be able to allocate enough memory to either the FFT or Multiexp kernel. Below are a list of devices that work for small sets. In the future we will add the cuttoff point at which a given card will not be able to allocate enough memory to utilize the GPU.

| Device Name            | Cores | Comments       |
|------------------------|-------|----------------|
| Quadro RTX 6000        | 4608  |                |
| TITAN RTX              | 4608  |                |
| Tesla V100             | 5120  |                |
| Tesla P100             | 3584  |                |
| Tesla T4               | 2560  |                |
| Quadro M5000           | 2048  |                |
| GeForce RTX 3090       |10496  |                |
| GeForce RTX 3080       | 8704  |                |
| GeForce RTX 3070       | 5888  |                |
| GeForce RTX 2080 Ti    | 4352  |                |
| GeForce RTX 2080 SUPER | 3072  |                |
| GeForce RTX 2080       | 2944  |                |
| GeForce RTX 2070 SUPER | 2560  |                |
| GeForce GTX 1080 Ti    | 3584  |                |
| GeForce GTX 1080       | 2560  |                |
| GeForce GTX 2060       | 1920  |                |
| GeForce GTX 1660 Ti    | 1536  |                |
| GeForce GTX 1060       | 1280  |                |
| GeForce GTX 1650 SUPER | 1280  |                |
| GeForce GTX 1650       |  896  |                |
|                        |       |                |
| gfx1010                | 2560  | AMD RX 5700 XT |
| gfx906                 | 7400  | AMD RADEON VII |
|------------------------|-------|----------------|

### Running Tests

To run using the `pairing` backend, you can use:

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release --all --no-default-features --features pairing
```

To run using both the `gpu` and `blst` backend, you can use:

```bash
RUSTFLAGS="-C target-cpu=native" cargo test --release --all --no-default-features --features gpu,blst
```

To run the multiexp_consistency test you can use:

```bash
RUST_LOG=info cargo test --features gpu -- --exact multiexp::gpu_multiexp_consistency --nocapture
```

### Considerations

Bellperson uses `rust-gpu-tools` as its OpenCL backend, therefore you may see a
directory named `~/.rust-gpu-tools` in your home folder, which contains the
compiled binaries of OpenCL kernels used in this repository.

## License

Licensed under either of

- Apache License, Version 2.0, |[LICENSE-APACHE](LICENSE-APACHE) or
   http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the Apache-2.0
license, shall be dual licensed as above, without any additional terms or
conditions.
