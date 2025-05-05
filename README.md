## Overview

This notebook demonstrates how to apply and compare common regularization techniques to improve a binary classification model on the HIGGS dataset using TensorFlow and Keras. You will learn how to load and preprocess the data, define models of varying capacity, apply L1/L2 regularization and dropout, and visualize training with TensorBoard.

## Prerequisites

* **Python 3.7+**
* **TensorFlow 2.x** installed via `pip install tensorflow` (includes TensorBoard).
* **tensorflow-docs** for helper plotting utilities: `pip install git+https://github.com/tensorflow/docs`.

## Directory Structure

```
.
├── regularization_notebook.ipynb  
├── tensorboard_logs/              
└── README.md                      
```

## Setup
Open the notebook (**main.ipynb**) and run cells in order.

## Learning Rate Schedule and Callbacks

* Use an **InverseTimeDecay** schedule that halves the learning rate every 1,000 epochs (base 0.001) to improve convergence.
* Define `get_callbacks(name)` to include:

  * `EpochDots` for concise epoch logging
  * `EarlyStopping(monitor='val_binary_crossentropy', patience=200)` to halt when validation loss stalls
  * `TensorBoard(log_dir=logdir/name)` to record scalars, graphs, and histograms

## Model Definitions and Training

1. **Tiny model**: one dense layer (16 units) to establish a baseline.
2. **Small model**: two dense layers (16 units each) to test modest capacity.
3. **Medium model**: three dense layers (64 units each) for increased power.
4. **Large model**: four dense layers (512 units each) to observe overfitting.

Each model is compiled with `BinaryCrossentropy(from_logits=True)` and trained via `compile_and_fit(...)` for up to 10,000 epochs, subject to early stopping.

## Regularization Techniques

After establishing capacity-based behavior, the notebook applies:

* **L2 weight regularization** (`kernel_regularizer=regularizers.l2(0.001)`) on each hidden layer to penalize large weights and improve generalization.
* **Dropout** layers (rate 0.5) interleaved between hidden layers to prevent co-adaptation of neurons and reduce overfitting.

Each variant’s training and validation losses are plotted side by side for direct comparison.


## Conclusion

This notebook highlights how model capacity, weight decay, and dropout interact to control overfitting. You can extend it by logging custom metrics or embeddings and experimenting with other schedules or regularizers.
