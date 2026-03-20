# Federated Kidney Disease Prediction
A federated learning system for predicting chronic kidney disease across multiple hospitals without sharing patient data.

## Features
- Autoencoder-based feature compression (15 features → 10D latent space)
- Cluster-aware federated aggregation
- Handles extreme class imbalance (97% CKD, 3% No CKD)
- Differential privacy protection

## Dataset
4,000 patients with binary CKD classification:
```
CKD:      3,875 (96.9%)
No CKD:     125  (3.1%)
```

Split into train/test:
```
Training: 3,200 samples
Test:       800 samples
```

## Program Requirements
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib streamlit
```

## Usage
```bash
python federated_train.py
```

## Demo
A Streamlit web app for interactive CKD prediction:
```bash
streamlit run demo.py
```
Enter patient vitals (creatinine, GFR, BUN, blood pressure, etc.) and get an instant CKD risk prediction with identified risk factors.

## How it works
1. Split data into 4 clients (simulated hospitals):
   - Client 1: 800 samples (770 CKD, 30 No CKD)
   - Client 2: 800 samples (778 CKD, 22 No CKD)
   - Client 3: 800 samples (770 CKD, 30 No CKD)
   - Client 4: 800 samples (782 CKD, 18 No CKD)
2. Each client trains local autoencoder
3. Cluster clients by latent space similarity
4. Federated training up to 25 rounds with cluster-aware aggregation
5. Global model evaluated on held-out test set

## Project structure
```
notebooks/          Jupyter notebooks for each pipeline step
src/                Python source code
models/             Saved models and encoders
data/               Dataset files
results/            Evaluation outputs
demo.py             Streamlit prediction app
requirements.txt    Dependencies
```

## Tech stack
- Python 3.8+
- TensorFlow 2.10
- scikit-learn
- pandas, numpy
- Streamlit

## License
MIT
