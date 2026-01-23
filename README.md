This is a personal project for the initiation to real life applications of machine learning. Why not have some fun and create something meaningful?

## Project Structure

```
├── data/                  # Dataset
│   └── breast-cancer.csv
├── src/                   # Core neural network code
│   ├── model.py           # Main entry point
│   ├── neuralnetwork.py   # Neural network layers and functions
│   ├── load_data.py       # Data loading utilities
│   ├── Saving_Weights.py  # Save weights to CSV
│   ├── loading_weights.py # Load weights from CSV
│   └── test.py            # Model evaluation
├── Model_Weights/         # Trained weights (CSV format)
├── notebooks/             # Jupyter notebooks for exploration
└── DataBase/              # Optional MySQL storage (not required)
```

## Usage

### Running the Model

From the `src/` directory:

```bash
cd src
python model.py
```

This will load the pre-trained weights from `Model_Weights/` and evaluate the model on the test set, printing accuracy, precision, and recall.

### Training from Scratch

In `src/model.py`, uncomment the training lines:

```python
train(network, 1e-5, 2000, bce, bce_prime, X, Y, True)
save(network)
```

Then comment out `load_weights(network)` to avoid overwriting with old weights before training.

### How Weights are Stored

Weights are saved as CSV files in `Model_Weights/`:

- `{layer_name}_weights.csv` - weight matrices
- `{layer_name}_bias.csv` - bias vectors

This keeps things simple and human-readable. No need for pickle or database storage for a model this size.

## Requirements

- numpy
- pandas
- scikit-learn
- mysql-connector-python (only if using database features -- I ran a local MySQL DB on my computer so get your own server :|)
