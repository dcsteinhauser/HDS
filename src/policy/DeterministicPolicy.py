import jax
from flax import linen as nn

class DeterministicPolicy(nn.module):
    def setup(self, observation_size, action_size):
        self.dense1 = nn.Dense(observation_size/2)
        self.dense2 = nn.Dense(action_size)

    def __call__(self, x):
        x = self.dense1(x)
        x = jax.nn.tanh(x)
        x = self.dense2(x)
        return x
