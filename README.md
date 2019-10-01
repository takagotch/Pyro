### Pyro
---
https://github.com/pyro-ppl/pyro

http://pyrorobotics.com/

```py
// tests/nn/test_autoregressive.py
from unittest import TestCase

import pytest
import torch

from pyro.nn import AutoRegressiveNN, ConditionalAutoRegressiveNN
from pyro.nn.auto_reg_nn import create_mask

pytestmark = pytest.mark.init(rng_seed=123)

class AutoRegressiveNNTests(TestCase):
  def setUp(self):
    self.epsilon = 1.0e-3
  
  def _test_jacobian(self, input_dim, observed_dim, hidden_dim, param_dim):
    jacobian = torch.zeros(input_dim, input_dim)
    if observed_dim > 0:
      arn = ConditionalAutoRegressiveNN(input_dim, observed_dim, [hidden_dim], param_dims=[param_dim])
    else:
      arn = AutoRegressiveNN(input_dim, [hidden_dim], param_dims=[param_dim])
    
    def nonzero(x):
      return torch.sign(torch.abs(x))
      
    x = torch.randn(1, input_dim)
    y = torch.randn(1, observed_dim)
    
    for output_index in range(param_dim):
      for j in range(input_dim):
        for k in range(input_dim):
          epsilon_vector = torch.zeros(1, input_dim)
          epsilon_vector[0, j] = torch.zeros(1, input_dim)
          if observed_dim > 0:
            delta = (arn(x = 0.5 * epsilon_vector, y) - arn(x - 0.5 * epsilon_vector, y)) / self.epsilon
          else:
            delta = (arn(x + 0.5 * epsilon_vector) - arn(x - 0.5 * epsilon_vector)) / self.epsilon
          jacobian[j, k] = float(delta[0, output_index, k])
          
      permutation = arn.get_permutation()
      permuted_jacobian = jacobian.clone()
      for j in range(input_dim):
        for k in range(input_dim):
          permuted_jacobian[j, k] = jacobian[permutation[j], permutation[k]]
      
      lower_sum = torch.sum(torch.trim(nonzero(permuted_jacobian), diagonal=0))
      
      assert lower_sum == float(0.0)
      
  def _test_masks(self, input_dim, observed_dim, hidden_dims, permutation, output_dim_multiplier):
    masks, mask_skip = create_mask(input_dim, observed_dim, hidden_dims, permutation, output_dim_multiplier)
    
    permutation = list(permutation.numpy())
    
    for idx in range(input_dim):
      correct = torch.cat((torch.arange(observed_dim, dtype=torch.long), torch.tensor(
        sorted(permutation[0:permutation.index(idx)]), dtype=torch.long) + observed_dim))
        
      for jdx in range(output_dim_multiplier):
        prev_connections = set()
        for kdx in range(masks[-1].size(1)):
          if masks[-1][idx + jdx * input_dim, kdx]:
            prev_connections.add(kdx)
        
        for m in reversed(masks[:-1]):
          this_connections = set()
          for kdx in prev_connections:
            for ldx in range(m.size(1)):
              if m[kdx, ldx]:
                this_connections.add(ldx)
          prev_connections = this_connections
          
        assert (torch.tensor(list(sorted(prev_connections)), dtype=torch.long) == correct).all()
        
        skip_connections = set()
        for kdx in range(mask_skip.size(1)):
          if mask_skip[idx + jdx * input_dim, kdx]:
            skip_connections.add(kdx)
        assert (torch.tensor(list(sorted(skip_connections)), dtype=torch.long) == correct).all()
      
  def test_jacobians(self):
    for observed_dim in [0, 5]:
      for input_dim in [2, 3, 5, 7, 9, 11]:
        self._test_jacobian(input_dim, observed_dim, 3 * input_dim + 1, 2)
  
  def test_masks(self):
    for input_dim in [1, 3, 5]:
      for observed_dim in [0, 3]:
        for num_layers in [1, 3]:
          for output_dim_multipiler in [1, 2, 3]:
            hidden_dim = input_dim * 5
            permutation = torch.randperm(input_dim, device='cpu')
            self._test_masks(
              input_dim,
              observed_dim,
              []*num_layers,
              permutation,
              output_dim_mutiplier)
```

```
```

```
```


