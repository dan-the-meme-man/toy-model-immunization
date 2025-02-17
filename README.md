# Immunizing a toy model by jointly minimzing cross entropy and its gradient norm

## Usage

```bash
python3 main.py -c [ce|gp]
```

- `ce`: Cross entropy loss
- `gp`: Joint objective of cross entropy and gradient norm penalty

## Results

See log file gp.o1527. This used the following loss calculation:

```python
loss = good_ce_loss + self.alpha * (bad_grad_norm - bad_ce_loss)
```

with:

```python
alpha_schedule = [0, 0, 0, 0, 0, 0, 0.125, 0.25, 0.5, 1]
```


