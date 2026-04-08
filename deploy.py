import streamlit as st
import pandas as pd
import joblib
from streamlit_lottie import st_lottie
import json
import time

# 1. Page Configuration
st.set_page_config(page_title="NextTrip AI | Airbnb Finder", page_icon="🌍", layout="wide")

# 2. Asset Loading (Updated for .pkl)
@st.cache_resource
def load_assets():
    try:
        # Load the Preprocessor and the Model using joblib
        preprocessor = joblib.load("airbnb_preprocessor.joblib")
        model = joblib.load("final_xgboost_smote_model.joblib") # Updated to use your .pkl file
        return preprocessor, model
    except Exception as e:
        st.error(f"⚠️ Error loading model files: {e}")
        return None, None

preprocessor, model = load_assets()

def load_lottie_file(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except: return None

welcome_animation = load_lottie_file("Start.json")
predict_animation = load_lottie_file("Predict.json")

# 3. UI Styling
st.markdown("""
    <style>
        .stApp { background: rgb(5, 50, 90); font-family: 'Arial', sans-serif; }
        .title { 
            font-size: 3.5em; color: #faf0e6; text-align: center; font-weight: bold; 
            text-shadow: 4px 4px 10px rgba(0, 0, 0, 0.8); margin-bottom: 0px;
        }
        .subtitle { font-size: 1.5em; color: #faf0e6; text-align: center; margin-bottom: 2em; }
        .card { 
            background: rgba(250, 240, 230, 0.1); border-radius: 20px; padding: 25px; 
            border: 1px solid rgba(250, 240, 230, 0.2); color: white; margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        label { color: #faf0e6 !important; font-size: 1.1em !important; }
    </style>
""", unsafe_allow_html=True)

# 4. Header
col_logo, col_text = st.columns([1, 3])
with col_logo:
    if welcome_animation: st_lottie(welcome_animation, height=200)
with col_text:
    st.markdown('<h1 class="title">NextTrip AI 🌍</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Personalized Airbnb Destination Insights</p>', unsafe_allow_html=True)

# 5. Input Form
with st.form("user_profile_form"):
    st.markdown('<h3 style="color:white;">👤 Traveler Profile</h3>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: age = st.number_input("Age", 18, 100, 28)
    with c2: gender = st.selectbox("Gender", ["MALE", "FEMALE", "OTHER", "UNKNOWN"])
    with c3: signup_method = st.selectbox("Signup", ["basic", "facebook", "google"])

    c4, c5 = st.columns(2)
    with c4: device = st.selectbox("Device", ["Mac Desktop", "Windows Desktop", "iPhone", "Android Phone", "iPad"])
    with c5: browser = st.selectbox("Browser", ["Chrome", "Safari", "Firefox", "IE", "Mobile Safari"])

    st.markdown('<h3 style="color:white;">🕒 Session Stats</h3>', unsafe_allow_html=True)
    c6, c7 = st.columns(2)
    with c6: total_seconds = st.slider("Browsing Time (Sec)", 1, 100000, 5000)
    with c7: unique_action_types = st.slider("Actions", 1, 50, 10)

    submit = st.form_submit_button("✨ Predict Destination")

# 6. Prediction Logic
if submit:
    if model and preprocessor:
        with st.spinner("Calculating Destination Match..."):
            time.sleep(1)
            
            raw_data = pd.DataFrame([{
                'age': age, 'gender': gender, 'signup_method': signup_method,
                'first_device_type': device, 'first_browser': browser,
                'total_seconds': total_seconds, 'unique_action_types': unique_action_types,
                'unique_actions': unique_action_types, 'total_actions': 50, 
                'avg_seconds_per_action': (total_seconds / unique_action_types),
                'unique_devices': 1, 'most_used_device': device, 'booking_intent_count': 0,
                'account_year': 2026, 'account_month': 4, 'account_day': 5,
                'first_active_year': 2026, 'first_active_month': 4, 'first_active_day': 5,
                'days_to_signup': 0, 'device_group': 'Desktop' if 'Desktop' in device else 'Mobile',
                'signup_flow': 0, 'language': 'en', 'affiliate_channel': 'direct',
                'affiliate_provider': 'direct', 'first_affiliate_tracked': 'untracked', 'signup_app': 'Web'
            }])

            try:
                processed_input = preprocessor.transform(raw_data)
                
                # Note: model.predict_proba depends on your classifier type
                probs = model.predict_proba(processed_input)[0]
                classes = model.classes_
                top_3_idx = probs.argsort()[-3:][::-1] 

                # MAPPING: Numbers to Names (Verify with your LabelEncoder classes)
                country_name_mapping = {
                    10: 'United States 🇺🇸', 7: 'Netherlands 🇳🇱', 11: 'Other Destinations ✈️',
                    4: 'France 🇫🇷', 6: 'Italy 🇮🇹', 5: 'Great Britain 🇬🇧',
                    3: 'Spain 🇪🇸', 2: 'Germany 🇩🇪', 1: 'Canada 🇨🇦', 0: 'Australia 🇦🇺',
                    8: 'Portugal 🇵🇹'
                }

                st.markdown('<h2 style="color:white; text-align:center;">🎯 Top Recommendations</h2>', unsafe_allow_html=True)
                res_cols = st.columns(3)
                
                for i, idx in enumerate(top_3_idx):
                    with res_cols[i]:
                        label_id = classes[idx] 
                        display_name = country_name_mapping.get(label_id, f"Code {label_id}")
                        conf = probs[idx] * 100
                        
                        st.markdown(f"""
                            <div class="card" style="text-align:center; border-top: 5px solid #4CAF50;">
                                <h2 style="margin:0; font-size:1.4em;">{display_name}</h2>
                                <p style="color:#4CAF50; font-weight:bold;">{conf:.1f}% Match</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                if predict_animation: st_lottie(predict_animation, height=200)
                if probs[top_3_idx[0]] > 0.6: st.balloons()

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    else:
        st.error("❌ Critical Error: model.pkl or Preprocessor files are missing.")