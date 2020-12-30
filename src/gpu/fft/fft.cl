uint bitreverse(uint n, uint bits) {
  uint r = 0;
  for(int i = 0; i < bits; i++) {
    r = (r << 1) | (n & 1);
    n >>= 1;
  }
  return r;
}

/*
 * FFT algorithm is inspired from: http://www.bealto.com/gpu-fft_group-1.html
 */
__kernel void radix_fft(
    __global FIELD *x,      // Source buffer
    __global FIELD *y,      // Destination buffer
    __global FIELD *pq,     // Precalculated twiddle factors
    __global FIELD *omegas, // [omega, omega^2, omega^4, ...]
    __local FIELD *u,       // Local buffer to store intermediary values
    uint n,                 // Number of elements
    uint lgp,               // Log2 of `p` (Read more in the link above)
    uint deg,               // 1=>radix2, 2=>radix4, 3=>radix8, ...
    uint max_deg) // Maximum degree supported, according to `pq` and `omegas`
{
  uint lid = get_local_id(0);
  uint lsize = get_local_size(0);
  uint index = get_group_id(0);
  uint t = n >> deg;
  uint p = 1 << lgp;
  uint k = index & (p - 1);

  x += index;
  y += ((index - k) << deg) + k;

  uint count = 1 << deg;    // 2^deg
  uint counth = count >> 1; // Half of count

  uint counts = count / lsize * lid;
  uint counte = counts + count / lsize;

  // Compute powers of twiddle
  const FIELD twiddle = FIELD_pow_lookup(omegas, (n >> lgp >> deg) * k);
  FIELD tmp = FIELD_pow(twiddle, counts);
  for (uint i = counts; i < counte; i++) {
    u[i] = FIELD_mul(tmp, x[i * t]);
    tmp = FIELD_mul(tmp, twiddle);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  const uint pqshift = max_deg - deg;
  for (uint rnd = 0; rnd < deg; rnd++) {
    const uint bit = counth >> rnd;
    for (uint i = counts >> 1; i < (counte >> 1); i++) {
      const uint di = i & (bit - 1);
      const uint i0 = (i << 1) - di;
      const uint i1 = i0 + bit;
      tmp = u[i0];
      u[i0] = FIELD_add(u[i0], u[i1]);
      u[i1] = FIELD_sub(tmp, u[i1]);
      if (di != 0)
        u[i1] = FIELD_mul(pq[di << rnd << pqshift], u[i1]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
  }

  for (uint i = counts >> 1; i < (counte >> 1); i++) {
    y[i * p] = u[bitreverse(i, deg)];
    y[(i + counth) * p] = u[bitreverse(i + counth, deg)];
  }
}

/*
 * Bitreverse elements before doing inplace FFT
 */
__kernel void reverse_bits(__global FIELD *a, // Source buffer
                           uint lgn)          // Log2 of n
{
  uint k = get_global_id(0);
  uint rk = bitreverse(k, lgn);
  if (k < rk) {
    FIELD old = a[rk];
    a[rk] = a[k];
    a[k] = old;
  }
}

/*
 * Inplace FFT algorithm, uses 1/2 less memory than radix-fft
 * Inspired from original bellman FFT implementation
 */
__kernel void
inplace_fft(__global FIELD *a,      // Source buffer
            __global FIELD *omegas, // [omega, omega^2, omega^4, ...]
            uint lgn,
            uint lgm) // Log2 of n
{
  uint gid = get_global_id(0);
  uint n = 1 << lgn;
  uint m = 1 << lgm;
  uint j = gid & (m - 1);
  uint k = 2 * m * (gid >> lgm);
  FIELD w = FIELD_pow_lookup(omegas, j << (lgn - 1 - lgm));
  FIELD t = FIELD_mul(a[k + j + m], w);
  FIELD tmp = a[k + j];
  tmp = FIELD_sub(tmp, t);
  a[k + j + m] = tmp;
  a[k + j] = FIELD_add(a[k + j], t);
}

/// Distribute powers
/// E.g.
/// [elements[0]*g^0, elements[1]*g^1, ..., elements[n]*g^n]
__kernel void distribute_powers(__global FIELD *elements, uint n, FIELD g) {
  uint gid = get_global_id(0);
  uint gsize = get_global_size(0);

  uint len = (uint)ceil(n / (float)gsize);
  uint start = len * gid;
  uint end = min(start + len, n);

  FIELD field = FIELD_pow(g, start);
  for (uint i = start; i < end; i++) {
    elements[i] = FIELD_mul(elements[i], field);
    field = FIELD_mul(field, g);
  }
}

/// Memberwise multiplication
__kernel void mul(__global FIELD *a, __global FIELD *b, uint n) {
  uint gid = get_global_id(0);
  a[gid] = FIELD_mul(a[gid], b[gid]);
}

/// Memberwise subtraction
__kernel void sub(__global FIELD *a, __global FIELD *b, uint n) {
  uint gid = get_global_id(0);
  a[gid] = FIELD_sub(a[gid], b[gid]);
}
