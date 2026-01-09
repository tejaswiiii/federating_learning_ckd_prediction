# Extract Scaler for Streamlit Demo
# This script recreates the scaler from the data preparation process

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Recreating preprocessing pipeline to extract scaler...")
print("="*70)

# Load original data (same as notebook 1)
data = pd.read_csv('../data/updated_ckd_dataset_with_stages.csv')
target_col = 'ckd_pred'

# Drop unnecessary columns
columns_to_drop = ['ckd_stage', 'cluster']
for col in columns_to_drop:
    if col in data.columns:
        data = data.drop(col, axis=1)

# Binary conversion (same as notebook 1)
def convert_to_binary(y):
    unique_classes = y.unique()
    if len(unique_classes) == 2:
        y_binary = (y == unique_classes[1]).astype(int)
        return y_binary
    if y.dtype == 'object':
        healthy_labels = ['Non-CKD', 'no', 'NO', 'Non CKD', 'non-ckd', 'notckd']
        healthy_label = None
        for label in unique_classes:
            if str(label).strip() in healthy_labels:
                healthy_label = label
                break
        if healthy_label is not None:
            y_binary = (y != healthy_label).astype(int)
        else:
            y_binary = (y != unique_classes[0]).astype(int)
        return y_binary
    y_binary = (y != min(unique_classes)).astype(int)
    return y_binary

y_binary = convert_to_binary(data[target_col])
data['Target'] = y_binary
data = data.drop(target_col, axis=1)

# Feature selection (same as notebook 1)
x = data.drop('Target', axis=1)
y = data['Target']

num_cols = x.select_dtypes(include=np.number).columns.tolist()
cat_cols = x.select_dtypes(include='object').columns.tolist()

x_imputed = x.copy()

if len(num_cols) > 0:
    imputer_num = SimpleImputer(strategy='median')
    x_imputed[num_cols] = imputer_num.fit_transform(x[num_cols])

if len(cat_cols) > 0:
    imputer_cat = SimpleImputer(strategy='most_frequent')
    x_imputed[cat_cols] = imputer_cat.fit_transform(x[cat_cols])
    
    for col in cat_cols:
        le = LabelEncoder()
        x_imputed[col] = le.fit_transform(x_imputed[col].astype(str))

k_features = min(15, x_imputed.shape[1])
selector = SelectKBest(mutual_info_classif, k=k_features)
x_selected = selector.fit_transform(x_imputed, y)

selected_features = x.columns[selector.get_support()].tolist()
x_reduced = x[selected_features].copy()

print(f"Selected features: {selected_features}")
print()

# Now create the preprocessing pipeline for the 15 selected features
print("Creating preprocessing pipeline for selected features...")

# Prepare the preprocessing objects
num_cols_selected = x_reduced.select_dtypes(include=np.number).columns.tolist()
cat_cols_selected = x_reduced.select_dtypes(include='object').columns.tolist()

# Create imputers
imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

# Fit imputers on the reduced feature set
x_for_fitting = x_reduced.copy()

if len(num_cols_selected) > 0:
    x_for_fitting[num_cols_selected] = imputer_num.fit_transform(x_reduced[num_cols_selected])

# Create label encoders for categorical features
label_encoders = {}
if len(cat_cols_selected) > 0:
    x_for_fitting[cat_cols_selected] = imputer_cat.fit_transform(x_reduced[cat_cols_selected])
    
    for col in cat_cols_selected:
        le = LabelEncoder()
        x_for_fitting[col] = le.fit_transform(x_for_fitting[col].astype(str))
        label_encoders[col] = le

# Convert to array and fit scaler
x_array = x_for_fitting.values
scaler = StandardScaler()
scaler.fit(x_array)

print(f"Scaler fitted on shape: {x_array.shape}")
print(f"Scaler mean: {scaler.mean_[:5]}... (first 5)")
print(f"Scaler std: {scaler.scale_[:5]}... (first 5)")
print()

# Save the preprocessing pipeline
preprocessing_pipeline = {
    'selected_features': selected_features,
    'num_cols': num_cols_selected,
    'cat_cols': cat_cols_selected,
    'imputer_num': imputer_num,
    'imputer_cat': imputer_cat,
    'label_encoders': label_encoders,
    'scaler': scaler,
    'feature_order': selected_features  # Exact order matters!
}

import os
os.makedirs('../data', exist_ok=True)

with open('../data/preprocessing_pipeline.pkl', 'wb') as f:
    pickle.dump(preprocessing_pipeline, f)

print("✅ Saved preprocessing pipeline to: ../data/preprocessing_pipeline.pkl")
print()
print("Pipeline contains:")
print(f"  - Selected features: {len(selected_features)}")
print(f"  - Numerical columns: {len(num_cols_selected)}")
print(f"  - Categorical columns: {len(cat_cols_selected)}")
print(f"  - Label encoders: {len(label_encoders)}")
print(f"  - StandardScaler fitted")
print()

# Test the pipeline with a sample
print("Testing pipeline with healthy patient example:")
test_input = pd.DataFrame({
    selected_features[0]: [0.9],   # serum_creatinine
    selected_features[1]: [98],    # gfr
    selected_features[2]: [14],    # bun
    selected_features[3]: [9.3],   # serum_calcium
    selected_features[4]: [0],     # ana
    selected_features[5]: [115],   # c3_c4
    selected_features[6]: [0],     # hematuria
    selected_features[7]: [2.1],   # oxalate
    selected_features[8]: [6.5],   # urine_ph
    selected_features[9]: [118],   # blood_pressure
    selected_features[10]: ['balanced'],  # diet
    selected_features[11]: [2.8],  # water_intake
    selected_features[12]: ['no'], # painkiller
    selected_features[13]: ['no'], # family_history
    selected_features[14]: ['stable']  # weight_changes
})

# Apply preprocessing
test_processed = test_input.copy()
if len(num_cols_selected) > 0:
    test_processed[num_cols_selected] = imputer_num.transform(test_input[num_cols_selected])
if len(cat_cols_selected) > 0:
    test_processed[cat_cols_selected] = imputer_cat.transform(test_input[cat_cols_selected])
    for col in cat_cols_selected:
        test_processed[col] = label_encoders[col].transform(test_processed[col].astype(str))

test_array = test_processed.values
test_scaled = scaler.transform(test_array)

print(f"Raw input: {test_array[0][:5]}... (first 5)")
print(f"Scaled input: {test_scaled[0][:5]}... (first 5)")
print()
print("✅ Pipeline test successful!")
print("="*70)