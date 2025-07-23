import jax
import jax.numpy as jnp
from jax import random
import time

key = random.PRNGKey(0)
size = 5000
x = random.normal(key, (size, size), dtype=jnp.float32)

start = time.time()
result = jnp.dot(x, x.T).block_until_ready()
print(result)
end = time.time()
print(f"{end - start:.4f} seconds")
