from flax import linen as nn
import jax

class DynamicsModel(nn.Module):

    output_size: int
    input_size: int

    def setup (self):
        self.dense1 = nn.Dense(8)
        self.dense2 = nn.Dense(16)
        self.dense3 = nn.Dense(64)
        self.dense4 = nn.Dense(16)
        self.dense5 = nn.Dense(8)
        self.dense6 = nn.Dense(self.output_size)
        self.layer_norm = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()

    def __call__ (self, x):
        x = self.dense1(x)
        x = self.layer_norm(x)
        x = jax.nn.leaky_relu(x)
        x = self.dense2(x)
        x = jax.nn.leaky_relu(x)
        x = self.dense3(x)
        x = jax.nn.leaky_relu(x)
        x = self.dense4(x)
        x = jax.nn.leaky_relu(x)
        x = self.layer_norm2(x)
        x = self.dense5(x)
        x = jax.nn.leaky_relu(x)
        x = self.dense6(x)
        return x


    


