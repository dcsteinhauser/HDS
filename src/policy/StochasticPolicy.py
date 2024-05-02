import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

class StochasticPolicy(nn.Module):

    observation_size: int
    action_size: int

    def setup(self):
        self.dense1 = nn.Dense(32)
        self.dense2 = nn.Dense(64)

        self.dense5 = nn.Dense(10)
        self.dense6 = nn.Dense(self.action_size*2)

    def __call__(self, x,rng_key):
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        x = jax.nn.relu(x)
        x = self.dense5(x)
        x = jax.nn.tanh(x)
        x = self.dense6(x)

        mean, logvar = jnp.split(x, 2, axis=-1)
        std = jnp.exp(0.5 * logvar)
        eps = random.normal(rng_key, mean.shape)
        x = mean + std * eps
        
        return x

