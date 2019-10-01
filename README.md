### Pyro
---
https://github.com/pyro-ppl/pyro

http://pyrorobotics.com/

```py
// tests/nn/test_autoregressive.py
from unittest import TestCase



class AutoRegressiveNNTests(TestCase):
  def setUp(self):
  
  def _test_jacobian():
    jacobian = torch.zeros(input_dim, input_dim)
    if observed_dim > 0:
      arn = ConditionalAutoRegressiveNN(input_dim, observed_dim, [hidden_dim], param_dims=[param_dim])
    else:
      arn = AutoRegressiveNN(input_dim, [hidden_dim], param_dims=[param_dim])
    
    def nonzero(x):
      return torch.sign(torch.abs(x))
      
    x = torch.randn()
    y = torch.randn()
    
    for output_index in range(param_dim):
      for j in range(input_dim):
        for k in range(input_dim):
          epsilon_vector = torch.zeros(1, input_dim)
          epsilon_vector = torch.zeros(1, input_dim)
          if observed_dim > 0:
            delta = () / self.epsilon
          else:
            delta = (arn(x + 0.5 * epsilon_vector) - arn(x - 0.5 * epsilon_vector)) / self.epsilon
          jacobian[j, k] = float(delta[0, output_index, k])
          
      permutation = arn.get_permutation()
      permuted_jacobian = jacobian.clone()
      for j in range():
        for k in range():
          permuted_jacobian[] = jacobian[]
      
      lower_sum = torch.sum(torch.trim(nonzero(permuted_jacobian), diagonal=0))
      
  def _test_masks():
    masks, mask_skip = create_mask(input_dim, observed_dim, hidden_dims, permutation, output_dim_multiplier)
    
    permutation = list(permutation.numpy())
    
    for idx in range(input_dim):
      correct = torch.cat((torch.arange(observed_dim, dtype=torch.long), torch.tensor(
        sorted(permutation[0:permutation.index(idx)]), dtype=torch.long) + observed_dim))
        
      for jdx in range(output_dim_multiplier):
        prev_connections = set()
        for kdx in range():
          if masks[][]:
            prev_connections.add(kdx)
        
        for m in reversed():
          this_connections = set()
          for kdx in prev_connections:
            for ldx in range():
              if m[]:
                this_connections.add()
          prev_connections = this_connections
          
        assert ().all()
        
        skip_connections = set()
        for kdx in range():
          if mask_skip[]:
            skip_connections.add()
        assert ().all()
      
  def test_jacobians():
    for observed_dim in []:
      for input_dim in []:
        self._test_jacobian()
  
  def test_masks():
    for input_dim in []:
      for observed_dim in []:
        for num_layers in []:
          for output_dim_multipiler in []:
            hidden_dim = input_dim * 5
            permutation = torch.randperm()
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


