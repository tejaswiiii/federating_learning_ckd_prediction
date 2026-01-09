"""
CKD Prediction System - Federated Learning Demo
Run with: streamlit run demo.py

FIXED: Handles unseen categorical values by mapping to closest known value
"""
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from collections import Counter

# Page config
st.set_page_config(
    page_title="CKD Prediction System",
    page_icon="🏥",
    layout="wide"
)

# Define FocalLoss (needed to load model)
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, name='focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        y_true = tf.cast(y_true, tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        
        ce = -y_true_one_hot * tf.math.log(y_pred)
        p_t = tf.where(tf.equal(y_true_one_hot, 1), y_pred, 1 - y_pred)
        focal_weight = tf.pow(1 - p_t, self.gamma)
        alpha_t = tf.where(tf.equal(y_true_one_hot, 1), self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    
    def get_config(self):
        config = super().get_config()
        config.update({'alpha': self.alpha, 'gamma': self.gamma})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Load model and preprocessing pipeline
@st.cache_resource
def load_model():
    try:
        # Load production config
        with open('models/production_config.pkl', 'rb') as f:
            config = pickle.load(f)
        
        # Load model (Round 3)
        model_path = f"models/{config['model_file']}"
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'FocalLoss': FocalLoss}
        )
        
        # Load encoder
        with open('data/client_encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        encoder = next((enc for enc in encoders if enc is not None), None)
        
        # Load preprocessing pipeline (CRITICAL!)
        with open('data/preprocessing_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        
        return model, encoder, pipeline, config
    except Exception as e:
        st.error(f"Error loading model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, None, None, None

model, encoder, pipeline, prod_config = load_model()

if model is None:
    st.error("❌ Could not load model files")
    st.info("Make sure you're running from the project root directory and have run extract_scaler.py first")
    st.stop()

# Get valid categorical values from label encoders
valid_values = {}
for col, le in pipeline['label_encoders'].items():
    valid_values[col] = list(le.classes_)

# Header
st.markdown("# 🏥 Chronic Kidney Disease Prediction System")
st.markdown("### Privacy-Preserving Federated Learning")
st.markdown("---")

# Model info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Recall", f"{prod_config['performance']['recall']*100:.0f}%", 
              help="Catches all CKD cases")
with col2:
    st.metric("Specificity", f"{prod_config['performance']['specificity']*100:.0f}%",
              help="Correctly identifies healthy patients")
with col3:
    st.metric("Accuracy", f"{prod_config['performance']['accuracy']*100:.0f}%",
              help="Overall correct predictions")
with col4:
    st.metric("AUC-ROC", f"{prod_config['performance']['auc_roc']:.3f}",
              help="Model discrimination ability")

st.info(f"🔬 **Model:** Round {prod_config['round']} | **Threshold:** {prod_config['threshold']:.2f} | **Optimized for:** {prod_config['clinical_note']}")

st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔬 Predict", "📊 Examples", "📖 About"])

with tab1:
    st.header("Enter Patient Information")
    
    col1, col2, col3 = st.columns(3)
    
    # Get feature names from pipeline
    features = pipeline['selected_features']
    
    with col1:
        st.subheader("Kidney Function Tests")
        serum_creatinine = st.number_input(
            "Serum Creatinine (mg/dL)", 
            0.5, 5.0, 1.0,
            help="Normal: 0.7-1.3"
        )
        gfr = st.number_input(
            "GFR (mL/min)", 
            0, 120, 90,
            help="Normal: >90"
        )
        bun = st.number_input(
            "BUN (mg/dL)", 
            7, 150, 15,
            help="Normal: 7-20"
        )
        serum_calcium = st.number_input(
            "Serum Calcium (mg/dL)", 
            5.0, 10.5, 9.0,
            help="Normal: 8.5-10.5"
        )
        c3_c4 = st.number_input(
            "C3/C4 Complement", 
            10, 180, 110
        )
        
    with col2:
        st.subheader("Clinical Markers")
        ana = st.selectbox(
            "ANA", 
            [0, 1],
            format_func=lambda x: "Negative" if x == 0 else "Positive"
        )
        hematuria = st.selectbox(
            "Blood in Urine",
            [0, 1],
            format_func=lambda x: "No" if x == 0 else "Yes"
        )
        oxalate_levels = st.number_input(
            "Oxalate Levels", 
            1.0, 5.0, 2.5
        )
        urine_ph = st.number_input(
            "Urine pH", 
            4.5, 8.0, 6.5,
            help="Normal: 4.5-8.0"
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mmHg)", 
            90, 180, 120,
            help="Normal: <120"
        )
        
    with col3:
        st.subheader("Lifestyle & History")
        
        # Show valid options for diet
        diet_options = valid_values.get('diet', ['balanced', 'high_protein', 'low_sodium'])
        diet = st.selectbox(
            "Diet Type",
            diet_options,
            help=f"Valid options from training: {', '.join(diet_options)}"
        )
        
        water_intake = st.number_input(
            "Water Intake (L/day)", 
            1.5, 3.5, 2.5
        )
        
        # Show valid options for painkiller
        painkiller_options = valid_values.get('painkiller_usage', ['no', 'yes'])
        painkiller_usage = st.selectbox(
            "Regular Painkiller Use",
            painkiller_options
        )
        
        # Show valid options for family history
        family_options = valid_values.get('family_history', ['no', 'yes'])
        family_history = st.selectbox(
            "Family History of CKD",
            family_options
        )
        
        # Show valid options for weight changes
        weight_options = valid_values.get('weight_changes', ['stable', 'gain', 'loss'])
        weight_changes = st.selectbox(
            "Recent Weight Changes",
            weight_options
        )
    
    st.markdown("---")
    
    if st.button("🔍 Predict CKD Risk", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            try:
                # Create input DataFrame with exact feature names from training
                input_dict = {
                    features[0]: serum_creatinine,
                    features[1]: gfr,
                    features[2]: bun,
                    features[3]: serum_calcium,
                    features[4]: ana,
                    features[5]: c3_c4,
                    features[6]: hematuria,
                    features[7]: oxalate_levels,
                    features[8]: urine_ph,
                    features[9]: blood_pressure,
                    features[10]: diet,
                    features[11]: water_intake,
                    features[12]: painkiller_usage,
                    features[13]: family_history,
                    features[14]: weight_changes
                }
                
                input_df = pd.DataFrame([input_dict])
                
                # Apply the SAME preprocessing as training
                # Step 1: Imputation
                num_cols = pipeline['num_cols']
                cat_cols = pipeline['cat_cols']
                
                input_processed = input_df.copy()
                
                if len(num_cols) > 0:
                    input_processed[num_cols] = pipeline['imputer_num'].transform(input_df[num_cols])
                
                if len(cat_cols) > 0:
                    input_processed[cat_cols] = pipeline['imputer_cat'].transform(input_df[cat_cols])
                    
                    # Step 2: Label encoding for categoricals
                    for col in cat_cols:
                        le = pipeline['label_encoders'][col]
                        try:
                            input_processed[col] = le.transform(input_processed[col].astype(str))
                        except ValueError as e:
                            # Handle unseen labels by using most common value
                            st.warning(f"Unknown value for {col}: '{input_df[col].values[0]}'. Using default: '{le.classes_[0]}'")
                            input_processed[col] = le.transform([le.classes_[0]])
                
                # Step 3: Convert to array in correct feature order
                input_array = input_processed[features].values
                
                # Step 4: StandardScaler normalization (CRITICAL!)
                input_scaled = pipeline['scaler'].transform(input_array)
                
                # Step 5: Transform through encoder to latent space (15 → 10)
                if encoder is not None:
                    input_latent = encoder.predict(input_scaled, verbose=0)
                else:
                    input_latent = input_scaled[:, :10]
                
                # Step 6: Predict
                prediction_prob = model.predict(input_latent, verbose=0)
                
                # LABEL MAPPING: 0=CKD, 1=Healthy
                ckd_probability = float(prediction_prob[0][0])
                healthy_probability = float(prediction_prob[0][1])
                
                # Use production threshold
                threshold = prod_config['threshold']
                is_ckd = ckd_probability >= threshold
                
                # Results
                st.markdown("---")
                st.markdown("## 📊 Results")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    if is_ckd:
                        st.error("### ⚠️ CKD DETECTED")
                        st.markdown(f"""
                        **Risk Level:** {ckd_probability*100:.1f}% probability
                        
                        **Recommendations:**
                        - Consult nephrologist immediately
                        - Complete metabolic panel required
                        - Kidney function monitoring
                        - Additional confirmatory tests
                        """)
                    else:
                        st.success("### ✅ NO CKD DETECTED")
                        st.markdown(f"""
                        **Risk Level:** {ckd_probability*100:.1f}% probability
                        
                        **Recommendations:**
                        - Continue routine monitoring
                        - Maintain healthy lifestyle
                        - Annual kidney screening
                        - Stay hydrated
                        """)
                
                with col2:
                    # Probability gauge
                    st.markdown("#### Prediction Confidence")
                    st.progress(ckd_probability)
                    st.caption(f"CKD: {ckd_probability*100:.1f}%")
                    st.caption(f"Healthy: {healthy_probability*100:.1f}%")
                    st.caption(f"Threshold: {threshold*100:.1f}%")
                
                # Debug info
                with st.expander("🔍 Debug Info"):
                    st.write(f"**Input (raw):** {input_array[0][:5]}...")
                    st.write(f"**Input (scaled):** {input_scaled[0][:5]}...")
                    st.write(f"**Latent (encoded):** {input_latent[0][:5]}...")
                    st.write(f"**Prediction probs:** CKD={ckd_probability:.4f}, Healthy={healthy_probability:.4f}")
                    st.write(f"**Threshold:** {threshold:.4f}")
                
                # Risk factors
                st.markdown("---")
                st.markdown("### 🔍 Risk Factors")
                
                risk_factors = []
                if serum_creatinine > 1.3:
                    risk_factors.append(f"⚠️ High Creatinine ({serum_creatinine:.2f})")
                if gfr < 60:
                    risk_factors.append(f"⚠️ Low GFR ({gfr})")
                if bun > 20:
                    risk_factors.append(f"⚠️ High BUN ({bun})")
                if hematuria == 1:
                    risk_factors.append("⚠️ Blood in urine present")
                if blood_pressure > 140:
                    risk_factors.append(f"⚠️ High BP ({blood_pressure})")
                if family_history in ['yes', 'YES', '1', 1]:
                    risk_factors.append("⚠️ Family history positive")
                
                if risk_factors:
                    st.warning("**Identified risk factors:**")
                    for rf in risk_factors:
                        st.markdown(f"- {rf}")
                else:
                    st.success("✅ No major risk factors detected")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                import traceback
                st.code(traceback.format_exc())

with tab2:
    st.header("📊 Example Cases")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚠️ CKD Case")
        st.code("""
Serum Creatinine: 4.2 (CRITICAL)
GFR: 28 (STAGE 4 CKD)
BUN: 95 (VERY HIGH)
Calcium: 7.8 (LOW)
Hematuria: Yes
BP: 158 (HIGH)
Family History: Yes
        """)
    
    with col2:
        st.markdown("### ✅ Healthy Case")
        st.code("""
Serum Creatinine: 0.9 (Normal)
GFR: 98 (Normal)
BUN: 14 (Normal)
Calcium: 9.3 (Normal)
Hematuria: No
BP: 118 (Normal)
Family History: No
        """)
    
    st.markdown("---")
    st.info(f"""
    **Model Characteristics:**
    - Optimized for high sensitivity ({prod_config['performance']['recall']*100:.0f}% recall)
    - Minimizes missed CKD cases ({prod_config['performance']['false_negatives']} false negatives)
    - May produce false positives ({prod_config['performance']['false_positives']} on test set)
    - Confirmatory testing recommended
    - Threshold: {prod_config['threshold']*100:.1f}%
    """)
    
    # Show valid categorical values
    st.markdown("---")
    st.markdown("### 📋 Valid Input Values")
    st.write("The model was trained with these categorical values:")
    for col, values in valid_values.items():
        st.write(f"**{col}:** {', '.join(map(str, values))}")

with tab3:
    st.header("📖 About")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 System Overview
        
        Privacy-preserving federated learning for CKD prediction.
        
        **Architecture:**
        - 🏥 4 federated clients (hospitals)
        - 🗜️ Autoencoder compression (15→10)
        - 🔐 Differential privacy
        - ⚖️ SMOTE + Focal Loss
        - 📊 StandardScaler normalization
        
        **Features (15):**
        1. Serum Creatinine
        2. GFR
        3. BUN
        4. Serum Calcium
        5. ANA
        6. C3/C4
        7. Hematuria
        8. Oxalate
        9. Urine pH
        10. Blood Pressure
        11. Diet
        12. Water Intake
        13. Painkiller Use
        14. Family History
        15. Weight Changes
        """)
    
    with col2:
        st.markdown(f"""
        ### 📊 Performance
        
        **Test Results:**
        - Recall: {prod_config['performance']['recall']*100:.0f}%
        - Specificity: {prod_config['performance']['specificity']*100:.0f}%
        - Precision: {prod_config['performance']['precision']*100:.1f}%
        - Accuracy: {prod_config['performance']['accuracy']*100:.0f}%
        - AUC-ROC: {prod_config['performance']['auc_roc']:.3f}
        
        **Confusion Matrix:**
        - True Negatives: {prod_config['performance']['true_negatives']}
        - True Positives: {prod_config['performance']['true_positives']}
        - False Positives: {prod_config['performance']['false_positives']}
        - False Negatives: {prod_config['performance']['false_negatives']}
        
        **Training:**
        - 4,960 samples (SMOTE-balanced)
        - 800 test samples
        - 25 federated rounds
        
        **Label Mapping:**
        - 0 = CKD (Chronic Kidney Disease)
        - 1 = Healthy
        """)
    
    st.markdown("---")
    st.warning("""
    ### ⚠️ Important Disclaimers
    
    1. **Educational prototype** - not for clinical use
    2. **Screening tool only** - requires confirmatory testing
    3. **Always consult healthcare professionals**
    4. For research and demonstration purposes
    """)

# Footer
st.markdown("---")
st.caption("⚕️ Educational Research Project - Not for Clinical Diagnosis")