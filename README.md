# Probabilistic Linear Integer Arithmetic (PLIA<sub>t</sub>)

---

This is the code repository for the paper 
"A Fast Convoluted Story: Scaling Probabilistic Inference for Integer Arithmetic".

## Installation

---

To install the package, run the following command:

```bash
cd probabilistic-arithmetic
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Experiments

---

### Inference

---

To run the inference experiment, the following command will initiate a single run of each of the four
inference tasks. Available options are which device to run on and the maximum bitwidth/sequence length to test.
For example, the following command will run the experiments on the GPU and test probabilistic integers
of domains up to 2 to the power 24.

```bash
python experiments/expectation/run.py --device gpu --max_bitwidth 24
```

### Learning

---

The neurosymbolic learning experiments can be run using the following commands.
Note that the datasets will be automatically downloaded when first running the experiments.

The MNIST experiment for 2 digits for each of the two numbers training for 10 epochs using a learning
rate of 0.001 can be run using the following command.

```bash
python experiments/addition/run.py --digits_per_number 2 --N_epochs 10 --learning_rate 0.001
```

Finally, optimising a 4x4 visual sudoku task can be run with the command.

```bash
python experiments/visudo/run.py --grid_size 4
```


