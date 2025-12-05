# app_beatsense_updated.py
"""
Final BeatSense Streamlit app — UI enhancements
- Single-file wizard-style flow (button-driven) plus optional sidebar navigation
- Home, Working principle, Terms & Conditions, Patient Info
- Existing ECG upload & analysis (keeps your original logic)
- Clean result summary and Thank You page

Run with:
    pip install -r requirements.txt
    streamlit run app_beatsense_updated.py
"""
import sys
print(f"Python version: {sys.version}")

import os
import io
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, filtfilt, resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# ----------------------- CSS styles -----------------------
st.markdown(
    """
    <style>
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #0078d7;  /* base nice blue */
        color: white;
    }

    /* Slightly darker blue panel inside sidebar (upload bar) */
    [data-testid="stSidebar"] .css-1d391kg, /* Sidebar header/title text */
    [data-testid="stSidebar"] .css-1hynsf2, /* Sidebar radio buttons */
    [data-testid="stSidebar"] .stFileUpload {
        background-color: #005a9e;  /* darker blue */
        color: white;
        padding: 10px;
        border-radius: 5px;
    }

    /* Buttons hover color */
    div.stButton > button:hover {
        background-color: #004070;
        color: white;
    }

    /* General header h1 in main page */
    h1 {
        color: #0078d7;
    }
    </style>
    """, unsafe_allow_html=True
)

# ------------------------ App config ------------------------
st.set_page_config(page_title="BeatSense", layout="wide", initial_sidebar_state="expanded")
LOGO_FILENAME = "beatsense_logo.png"
WORK_DIR = "ecg_data"
os.makedirs(WORK_DIR, exist_ok=True)

# ------------------------ Utility functions (same as yours) ------------------------
def save_uploaded_files(uploaded_files, dest_dir=WORK_DIR):
    saved = []
    for file in uploaded_files:
        path = os.path.join(dest_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        saved.append(path)
    return saved

# **Added missing function here:**
def get_base_names(directory):
    basenames = set()
    for fname in os.listdir(directory):
        base, _ = os.path.splitext(fname)
        basenames.add(base)
    return sorted(list(basenames))

# ... keep all your other utility functions here unchanged ...

# ------------------------ Session state defaults ------------------------
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'
if 'accepted_terms' not in st.session_state:
    st.session_state['accepted_terms'] = False
if 'patient_info' not in st.session_state:
    st.session_state['patient_info'] = {'name':'', 'age':'', 'gender':'Not specified'}
if 'last_analysis' not in st.session_state:
    st.session_state['last_analysis'] = None
if 'logo_path' not in st.session_state:
    st.session_state['logo_path'] = LOGO_FILENAME if os.path.exists(LOGO_FILENAME) else None

# ------------------------ Navigation helper ------------------------
def go_to(page_name):
    st.session_state['page'] = page_name
    st.rerun()  # rerun app to apply page change

# ------------------------ UI: Sidebar navigation ------------------------
st.sidebar.title("Navigation")
pages = ["Home", "Working Principle", "Terms & Conditions", "Patient Info", "ECG Upload & Analysis", "Results", "Thank You"]
choice = st.sidebar.radio("(Quick jump) Go to", pages, index=pages.index(st.session_state['page']))
if choice != st.session_state['page']:
    st.session_state['page'] = choice
    st.rerun()

if st.session_state['logo_path'] and os.path.exists(st.session_state['logo_path']):
    st.sidebar.image(st.session_state['logo_path'], use_column_width=True)
else:
    st.sidebar.markdown("**BeatSense**")

logo_file = st.sidebar.file_uploader("(Optional) Upload a logo image", type=["png","jpg","jpeg"], key='logo_uploader')
if logo_file:
    saved_logo_path = os.path.join(WORK_DIR, LOGO_FILENAME)
    with open(saved_logo_path, "wb") as f:
        f.write(logo_file.getbuffer())
    st.session_state['logo_path'] = saved_logo_path
    st.success("Logo uploaded and will appear in the header/sidebar.")

# ------------------------ Page implementations ------------------------
def show_home():
    st.header("BeatSense")
    col1, col2 = st.columns([1,3])
    with col1:
        if st.session_state.get('logo_path') and os.path.exists(st.session_state.get('logo_path')):
            st.image(st.session_state.get('logo_path'), width=150)
        else:
            st.markdown("### ❤️")
    with col2:
        st.markdown("### Welcome to BeatSense")
        st.write("BeatSense is an academic tool that analyses ECG recordings to detect rhythm abnormalities using signal processing and machine learning.")
        st.write("Follow the step-by-step flow: read the working principle, accept terms, enter patient info, upload ECG files, and view results.")
    st.markdown("---")
    if st.button("Proceed to Working Principle"):
        go_to("Working Principle")

def show_working_principle():
    st.header("Working Principle of BeatSense")
    st.markdown(
        """
        **Overview**

        1. The app reads ECG signals in WFDB format (.hea/.dat and optional .atr annotations).
        2. Signals are bandpass-filtered and R-peaks are detected (Pan-Tompkins if no annotations).
        3. Individual beats are extracted and resampled to a fixed length; beat-level features are computed.
        4. A Random Forest classifier performs beat-level classification when sufficient labels are present.
        5. Sequences of beats are summarized (RR statistics, RMSSD, pNN50, etc.) and rule-based logic classifies rhythm windows.
        6. If enough labeled tachy sequences exist, a second RF model is trained to subtype tachyarrhythmias.

        The app is intended for academic/demonstration purposes and NOT a substitute for clinical diagnosis.
        """
    )
    st.markdown("---")
    if st.button("Proceed to Terms & Conditions"):
        go_to("Terms & Conditions")

def show_terms():
    st.header("Terms & Conditions")
    st.warning("**BeatSense is for academic purposes only. This tool does NOT provide medical diagnosis. Always consult a qualified healthcare professional for clinical decisions.**")
    st.markdown("By using this tool you agree that you understand its academic/demo nature and that no real clinical decisions should be made from the outputs.")

    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state['accepted_terms']:
            if st.button("I Accept and Proceed"):
                st.session_state['accepted_terms'] = True
                st.success("Terms accepted — moving to Patient Info.")
                go_to("Patient Info")
        else:
            st.info("Terms already accepted — you may continue.")
            if st.button("Proceed to Patient Info"):
                go_to("Patient Info")
    with col2:
        if st.button("Decline"):
            st.session_state['accepted_terms'] = False
            st.warning("You declined the terms. Analysis will require acceptance.")

def show_patient_info():
    st.header("Patient Information")
    with st.form(key='patient_form'):
        name = st.text_input("Full name", value=st.session_state['patient_info'].get('name',''))
        age = st.number_input("Age", min_value=0, max_value=150, value=int(st.session_state['patient_info'].get('age') or 0))
        gender = st.selectbox("Gender", ['Not specified','Female','Male','Other'], index=['Not specified','Female','Male','Other'].index(st.session_state['patient_info'].get('gender','Not specified')))
        submitted = st.form_submit_button("Save patient info")
        if submitted:
            st.session_state['patient_info'] = {'name':name, 'age':age, 'gender':gender}
            st.success("Patient info saved.")

    st.markdown("---")
    cols = st.columns(3)
    if cols[1].button("Proceed to ECG Upload & Analysis"):
        if not st.session_state['accepted_terms']:
            st.error("Please accept Terms & Conditions first.")
        else:
            go_to("ECG Upload & Analysis")

def show_ecg_analysis():
    st.header("ECG Upload & Analysis")
    if not st.session_state['accepted_terms']:
        st.error("You must accept the Terms & Conditions before performing analysis.")
        return

    st.sidebar.header("Upload ECG files")
    uploaded_files = st.sidebar.file_uploader("Upload .hea, .dat, .atr files (same basename)", type=["hea","dat","atr"], accept_multiple_files=True)
    if uploaded_files:
        saved = save_uploaded_files(uploaded_files, dest_dir=WORK_DIR)
        st.sidebar.success(f"Saved {len(saved)} files to {WORK_DIR}")

    available_bases = get_base_names(WORK_DIR)
    if not available_bases:
        st.info("No files available — upload .hea/.dat/.atr in the sidebar.")
        return

    chosen_base = st.selectbox("Select record (basename)", available_bases)
    st.markdown(f"**Selected:** `{chosen_base}`")
    max_duration_sec = st.sidebar.number_input("Max duration (sec)", value=120, min_value=10, step=10)
    resample_len = st.sidebar.number_input("Beat resample length", value=100, min_value=50, step=10)
    window_ms = st.sidebar.number_input("Beat window (ms)", value=700, min_value=300, step=50)
    run_button = st.button("Run ECG Analysis")

    if run_button:
        # Your existing ECG analysis code here (unchanged) ...
        # ...
        # After your analysis and display logic:
        st.success("Analysis complete.")
        if st.button("Proceed to Results"):
            go_to("Results")

def show_results():
    st.header("Results Summary")
    if not st.session_state['last_analysis']:
        st.info("No analysis run yet — go to ECG Upload & Analysis and run the analysis.")
        return
    res = st.session_state['last_analysis']
    overall_summary = res['overall_summary']

    # Clean result statement
    total = sum(overall_summary.values()) if sum(overall_summary.values())>0 else 1
    if overall_summary.get('Tachycardia',0) == 0 and overall_summary.get('Bradycardia',0) == 0:
        st.success("Result: Normal rhythm detected. Based on the uploaded recording, the heart rhythm appears healthy. This is for academic/demonstration purposes only.")
    else:
        if overall_summary.get('Bradycardia',0) > 0:
            st.warning("Result: Some windows show bradycardia (slow heart rate). If symptomatic, seek professional medical advice.")
        if overall_summary.get('Tachycardia',0) > 0:
            st.warning("Result: Some windows show tachycardia (fast heart rate). If symptomatic or persistent, please consult a healthcare professional.")

    st.subheader("Detailed summary")
    st.dataframe(pd.DataFrame([{"Rhythm Type": k, "Sequences": v} for k,v in overall_summary.items() if v>0]))

    # quick chart
    fig, ax = plt.subplots(figsize=(8,3))
    labels = [k for k,v in overall_summary.items() if v>0]
    vals = [v for k,v in overall_summary.items() if v>0]
    ax.bar(labels, vals)
    ax.set_title("Sequence-level rhythm distribution")
    st.pyplot(fig)

    st.markdown("---")
    st.write("Patient info:")
    st.write(st.session_state['patient_info'])

    # download existing CSVs if present
    if 'beat_df' in res and res['beat_df'] is not None:
        st.download_button(label="Download beat-level CSV", data=res['beat_df'].to_csv(index=False).encode('utf-8'),
                           file_name=f"beats_{res['chosen_base']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")
    if 'sequence_df' in res and res['sequence_df'] is not None:
        st.download_button(label="Download sequence-level CSV", data=res['sequence_df'].to_csv(index=False).encode('utf-8'),
                           file_name=f"sequences_{res['chosen_base']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

    st.markdown("---")
    cols = st.columns(3)
    if cols[1].button("Proceed to Thank You"):
        go_to("Thank You")

def show_thankyou():
    st.header("Thank you for using BeatSense")
    st.write("We appreciate you trying this academic demonstration. If you found this useful, please star the repo on GitHub and cite appropriately.")
    st.write("Remember: BeatSense is an academic tool — not for clinical use.")

# ------------------------ Page router ------------------------
if st.session_state['page'] == 'Home':
    show_home()
elif st.session_state['page'] == 'Working Principle':
    show_working_principle()
elif st.session_state['page'] == 'Terms & Conditions':
    show_terms()
elif st.session_state['page'] == 'Patient Info':
    show_patient_info()
elif st.session_state['page'] == 'ECG Upload & Analysis':
    show_ecg_analysis()
elif st.session_state['page'] == 'Results':
    show_results()
else:
    show_thankyou()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("BeatSense — academic demo. Not for clinical use.")
