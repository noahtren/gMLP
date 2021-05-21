import functools
import operator

import jax
import jax.numpy as np
import jax.random as random

import flax.linen as nn
import flax


class SpatialGatingUnit(nn.Module):
  @nn.compact
  def __call__(self, x):
    u, v = np.split(x, 2, axis=-1)
    v = nn.normalization.LayerNorm()(v)
    dims = v.shape[1:-1]
    axes = np.arange(v.ndim)[1:-1]
    v = np.moveaxis(v, [0, v.ndim - 1, *axes], np.arange(v.ndim))
    general = nn.DenseGeneral(
        features=dims,
        axis=tuple(np.arange(v.ndim)[2:]),
        kernel_init=nn.initializers.variance_scaling(
            0.1, 'fan_in', 'truncated_normal'),  # near-zero projection matrix
        bias_init=nn.initializers.ones)
    v = general(v)
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


class gMLPModel(nn.Module):
  ffn_dim: int
  model_dim: int
  num_blocks: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(name='embedding', features=self.model_dim)(x)
    for i in range(self.num_blocks):
      x = gMLPBlock(ffn_dim=self.ffn_dim, model_dim=self.model_dim)(x)
    return x


tiny_settings = {'ffn_dim': 768, 'model_dim': 128, 'num_blocks': 30}

if __name__ == "__main__":
  key = random.PRNGKey(2)
  x = random.normal(key, shape=[8, 12, 18, 100])
  model = gMLPModel(**tiny_settings)
  model_state = model.init(key, x)
  y = model.apply(model_state, x)
  print(
      json.dumps(jax.tree_map(np.shape,
                              flax.core.unfreeze(model_state['params'])),
                 indent=2))
  num_params = functools.reduce(
      operator.add, map(np.size, jax.tree_leaves(model_state['params'])))
