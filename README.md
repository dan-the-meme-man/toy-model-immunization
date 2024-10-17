# Immunizing a toy model by jointly minimzing cross entropy and its gradient norm

## Usage

```bash
python3 main.py -c [ce|gp]
```

- `ce`: Cross entropy loss
- `gp`: Joint objective of cross entropy and gradient norm penalty

## Results

See the log files ce.o1194 and gp.o1993. Both models can learn approximately equally well. I use a scheduler to slowly increase the gradient norm penalty. If this is not done, the model will learn more slowly.

An earlier experiment with a constant `alpha=0.1` took twice as many epochs to converge to the same accuracy as the ordinary cross entropy loss.

`alpha_schedule` in `main.py` can be adjusted to change how the penalty is increased over the epochs.

It remains to be seen whether the toy model can be easily fine-tuned.
