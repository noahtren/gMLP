import jax.numpy as np
import jax.random as random

import flax.linen as nn


class SpatialGatingUnit(nn.Module):
  @nn.compact
  def __call__(self, x):
    u, v = np.split(x, 2, axis=-1)
    v = nn.normalization.LayerNorm()(v)
    dims = v.shape[1:-1]
    axes = np.arange(v.ndim)[1:-1]
    v = np.moveaxis(v, [0, v.ndim - 1, *axes], np.arange(v.ndim))
    v = nn.DenseGeneral(
        features=dims,
        axis=np.arange(v.ndim)[2:],
        batch_dims=[0, 1],
        kernel_init=nn.initializers.variance_scaling(
            0.1, 'fan_in', 'truncated_normal'),  # near-zero projection matrix
        bias_init=nn.initializers.ones)(v)
    v = np.moveaxis(v, np.arange(v.ndim), [0, v.ndim - 1, *axes])
    return u * v


class gMLPBlock(nn.Module):
  """Gating MLP block
  """
  ffn_dim: int
  model_dim: int

  @nn.compact
  def __call__(self, x):
    # if temporal, x is [b, l, d]
    # if spatial, x is  [b, h, w, d]
    shortcut = x
    x = nn.normalization.LayerNorm()(x)
    x = nn.Dense(features=self.ffn_dim)(x)
    x = nn.gelu(x)
    x = SpatialGatingUnit()(x)
    x = nn.Dense(features=self.model_dim)(x)
    x = x + shortcut
    return x


if __name__ == "__main__":
  key = random.PRNGKey(2)
  x = random.normal(key, shape=[8, 12, 18, 100])
  model_state = SpatialGatingUnit().init(key, x)
