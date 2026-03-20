# Federated Kidney Disease Prediction
A federated learning system for predicting chronic kidney disease severity across multiple hospitals without sharing patient data.

## Features
- Autoencoder-based feature compression (100+ features → 16D)
- Cluster-aware federated aggregation
- Handles extreme class imbalance (84% healthy, 2% severe)
- Differential privacy protection

## Dataset
19,502 kidney disease patients across 5 severity levels:
```
No Disease:     15,625 (84.1%)
Low Risk:        1,943 (10.5%)
Moderate Risk:     775 (4.2%)
High Risk:         772 (4.2%)
Severe Disease:    387 (2.1%)
```

## Installation
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

## How it works
1. Split data into 4 clients (simulated hospitals)
2. Each client trains local autoencoder
3. Cluster clients by latent space similarity
4. Federated training up to 25 rounds with cluster-aware aggregation
5. Global model evaluation

## Project structure
```
notebooks/          Jupyter notebooks for each pipeline step
src/                Python source code
models/             Saved models and encoders
data/               Dataset files
requirements.txt    Dependencies
```

## Tech stack
- Python 3.8+
- TensorFlow 2.10
- scikit-learn
- pandas, numpy

## License
MIT
