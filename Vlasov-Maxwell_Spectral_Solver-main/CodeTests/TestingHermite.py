import jax.numpy as jnp
import jax
import time
from jax.scipy.special import factorial
from orthax import hermite

jax.config.update("jax_enable_x64", True)


# Reccurence Method
@jax.jit
def hermite_recurrence(n, x):
    x = jnp.asarray(x, dtype=jnp.float64)

    def base_case_0(_):
        return jnp.ones_like(x)

    def base_case_1(_):
        return 2 * x

    def recurrence_case(n):
        H_n_minus_2 = jnp.ones_like(x)
        H_n_minus_1 = 2 * x

        def body_fn(i, carry):
            H_n_minus_2, H_n_minus_1 = carry
            H_n = 2 * x * H_n_minus_1 - 2 * (i - 1) * H_n_minus_2
            return (H_n_minus_1, H_n)

        _, H_n = jax.lax.fori_loop(2, n + 1, body_fn, (H_n_minus_2, H_n_minus_1))
        return H_n

    return jax.lax.switch(n, [base_case_0, base_case_1, recurrence_case], n)


# Factorial Method
@jax.jit
def hermite_factorial(n, x):
    """
    Computes value for any hermite polynomial
    n - mode number of hermite polynomial
    x - function input value
    """
    n = n.astype(int)

    def add_Hermite_term(m, partial_sum):
        return partial_sum + ((-1) ** m / (factorial(m) * factorial(n - 2 * m))) * (2 * x) ** (n - 2 * m)

    return factorial(n) * jax.lax.fori_loop(0, (n // 2) + 1, add_Hermite_term, jnp.zeros_like(x))


# Orthax Method
def hermite_orthax(n, x):
    x = jnp.asarray(x, dtype=jnp.float64)

    H_0 = hermite.hermzero(x)
    H_1 = hermite.hermone(x)

    if n == 0:
        return H_0
    elif n == 1:
        return H_1

    H_n_minus_2 = H_0
    H_n_minus_1 = H_1

    def recurrence_step(i, carry):
        H_n_minus_2, H_n_minus_1 = carry
        H_n = hermite.hermmulx(H_n_minus_1) - 2 * (i - 1) * H_n_minus_2
        return (H_n_minus_1, H_n)

    _, H_n = jax.lax.fori_loop(2, n + 1, recurrence_step, (H_n_minus_2, H_n_minus_1))

    return H_n


# Benchmarking
def benchmark_hermite(n, num_points=1000):
    x = jnp.linspace(-5, 5, num_points, dtype=jnp.float64)

    # Warm-up JAX functions to avoid compilation time
    hermite_recurrence(n, x)
    hermite_factorial(n, x)
    hermite_orthax(n, x)

    # Timing for Recurrence Method
    start_time = time.time()
    recurrence_result = hermite_recurrence(n, x).block_until_ready()
    recurrence_time = time.time() - start_time

    # Timing for Factorial Method
    start_time = time.time()
    factorial_result = hermite_factorial(n, x).block_until_ready()
    factorial_time = time.time() - start_time

    # Timing for Orthax Method
    start_time = time.time()
    orthax_result = hermite_orthax(n, x).block_until_ready()
    orthax_time = time.time() - start_time

    # Check if results match
    match = jnp.allclose(recurrence_result, factorial_result, atol=1e-6, rtol=1e-4)
    speedup = "Infinity" if recurrence_time == 0 else f"{factorial_time / recurrence_time:.2f}x Faster"

    # Print the results
    print(f"Hermite Polynomial Order: {n}")
    print(f"Recurrence Method Time: {recurrence_time:.6f} s")
    print(f"Factorial Method Time: {factorial_time:.6f} s")
    print(f"Orthax Method Time: {orthax_time:.6f} s")
    print(f"Results Match: {'Yes' if match else 'No'}\n")
    print("Sample inputs:", x[:10])
    print("Orthax outputs:", orthax_result[:10])
    print("Recurrence Output:", recurrence_result[:10])
    print("Factorial Output:", factorial_result[:10])


# Running the benchmark
benchmark_hermite(50)
