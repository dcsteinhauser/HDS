from flax import linen as nn
import jax

class model(nn.Module):

    output_size: int
    input_size: int

    def setup (self):
        self.dense1 = nn.Dense(8)
        self.dense2 = nn.Dense(16)
        self.dense3 = nn.Dense(8)
        self.dense4 = nn.Dense(self.output_size)

    def __call__ (self, x):
        x = self.dense1(x)
        x = jax.nn.relu(x)
        x = self.dense2(x)
        x = jax.nn.relu(x)
        x = self.dense3(x)
        x = self.dense4(x)
        return x


    


