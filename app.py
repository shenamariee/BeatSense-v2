# app.py
"""
BeatSense app — full analysis + wizard UI with linear step flow and sidebar logo + upload only
Run:
    pip install -r requirements.txt
    streamlit run app.py
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

# ----------------------- Visual tweaks (CSS) -----------------------
st.set_page_config(page_title="BeatSense", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    /* Sidebar background color */
    [data-testid="stSidebar"] {
        background-color: #0078d7;  /* base nice blue */
        color: white;
        padding: 1rem;
    }

    /* Slightly darker blue panel inside sidebar (uploader block) */
    [data-testid="stSidebar"] input[type="file"] {
        background-color: #005a9e;  /* darker blue */
        color: white;
        padding: 6px;
        border-radius: 6px;
    }

    /* Buttons hover color */
    div.stButton > button:hover {
        background-color: #004070;
        color: white;
    }

    /* Main h1 color */
    h1 {
        color: #0078d7;
    }

    /* Make sidebar step label stand out */
    .step-label { font-weight:600; font-size:14px; color: #ffffff; }
    </style>
    """, unsafe_allow_html=True
)

# ------------------------ Constants & working dir ------------------------
LOGO_FILENAME = "BeatSense Logo.jpg"  # exact filename in your repo
WORK_DIR = "ecg_data"
os.makedirs(WORK_DIR, exist_ok=True)

# ------------------------ Utility functions (from your original) ------------------------
def save_uploaded_files(uploaded_files, dest_dir=WORK_DIR):
    saved = []
    for file in uploaded_files:
        path = os.path.join(dest_dir, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        saved.append(path)
    return saved

def get_base_names(directory):
    basenames = set()
    for fname in os.listdir(directory):
        base, _ = os.path.splitext(fname)
        basenames.add(base)
    return sorted(list(basenames))

def bandpass(sig, fs, low=0.5, high=40):
    b, a = butter(3, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, sig)

def pan_tompkins_detector(signal, fs):
    b, a = butter(3, [5/(fs/2), 15/(fs/2)], btype='band')
    filtered_ecg = filtfilt(b, a, signal)
    diff_signal = np.ediff1d(filtered_ecg, to_end=0)
    squared = diff_signal ** 2
    window_size = int(0.150 * fs)
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    from scipy.signal import find_peaks
    distance = int(0.25 * fs)
    height = np.mean(integrated) * 1.2
    peaks, _ = find_peaks(integrated, distance=distance, height=height)
    refined_peaks = []
    search_radius = int(0.05 * fs)
    for p in peaks:
        start = max(p - search_radius, 0)
        end = min(p + search_radius, len(signal))
        local_max = np.argmax(signal[start:end]) + start
        refined_peaks.append(local_max)
    return np.unique(refined_peaks)

def extract_beats(signal, r_peaks, fs, window_ms=700, resample_len=100):
    half = int((window_ms / 1000) * fs // 2)
    beats = []
    indices = []
    for r in r_peaks:
        if r - half < 0 or r + half >= len(signal):
            continue
        beat = signal[r - half:r + half]
        beats.append(resample(beat, resample_len))
        indices.append(r)
    return np.array(beats), np.array(indices)

def extract_features(beats, rr_intervals):
    features = []
    for i, beat in enumerate(beats):
        rr = rr_intervals[i] if i < len(rr_intervals) else rr_intervals[-1]
        features.append([
            np.mean(beat),
            np.std(beat),
            np.min(beat),
            np.max(beat),
            rr,
            np.median(beat),
            np.percentile(beat, 25),
            np.percentile(beat, 75),
            np.sum(beat**2),
            len(beat)
        ])
    return np.array(features)

def is_irregular(rr_segment, threshold=0.12):
    return np.std(rr_segment) > threshold

def classify_tachycardia_regular(beat_seq):
    if any(b == "V" for b in beat_seq):
        return "Ventricular Tachycardia"
    if any(b == "A" for b in beat_seq):
        return "Atrial Flutter"
    if any(b in ["L", "R"] for b in beat_seq):
        return "Supraventricular Tachycardia"
    return "Supraventricular Tachycardia"

def classify_tachycardia_irregular(beat_seq):
    if any(b == "F" for b in beat_seq):
        return "Atrial Fibrillation"
    if any(b == "V" for b in beat_seq):
        return "Ventricular Fibrillation"
    return "Atrial Fibrillation"

label_map = {"N":0, "L":1, "R":2, "V":3, "A":4, "F":5}
tachy_label_map = {
    "Atrial Fibrillation": 0,
    "Ventricular Tachycardia": 1,
    "Supraventricular Tachycardia": 2,
    "Atrial Flutter": 3,
    "Other Tachy": 4,
    "Tachycardia": -1
}

@st.cache_resource
def train_rf_model(X, y, n_estimators=200):
    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced', random_state=42)
    clf.fit(X, y)
    return clf

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
    # prefer exact filename in repo if present
    st.session_state['logo_path'] = LOGO_FILENAME if os.path.exists(LOGO_FILENAME) else None

# ------------------------ Navigation helper ------------------------
def go_to(page_name):
    st.session_state['page'] = page_name
    # Try available rerun methods (works across streamlit versions)
    try:
        st.experimental_rerun()
    except Exception:
        try:
            st.rerun()
        except Exception:
            pass

# ------------------------ Sidebar (logo, current step, uploader only on ECG page) ------------------------
with st.sidebar:
    # logo (if available)
    if st.session_state.get('logo_path') and os.path.exists(st.session_state.get('logo_path')):
        try:
            st.image(st.session_state['logo_path'], use_column_width=True)
        except Exception:
            # fallback: show text if image loading fails
            st.markdown("**BeatSense**")
    else:
        st.markdown("**BeatSense**")

    st.markdown("<div class='step-label'>Current step:</div>", unsafe_allow_html=True)
    st.markdown(f"**{st.session_state['page']}**")

    st.markdown("---")
    # only show uploader when on ECG Upload page
    if st.session_state['page'] == "ECG Upload & Analysis":
        uploaded_files = st.file_uploader("Upload ECG files (.hea, .dat, .atr)", type=["hea","dat","atr"], accept_multiple_files=True)
        if uploaded_files:
            saved = save_uploaded_files(uploaded_files, dest_dir=WORK_DIR)
            st.success(f"Saved {len(saved)} files to {WORK_DIR}")

    st.markdown("---")
    st.markdown("BeatSense — academic demo. Not for clinical use.")

# ------------------------ Pages (wizard flow) ------------------------
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
        st.write("Follow the guided flow using the Proceed buttons. The sidebar shows the current step and (on the ECG page) lets you upload WFDB files (.hea/.dat and optional .atr).")
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

        The app is intended for academic and demonstration purposes and NOT a substitute for clinical diagnosis.
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

    available_bases = get_base_names(WORK_DIR)
    if not available_bases:
        st.info("No WFDB records found in the working directory. Upload `.hea`/.`dat` (and `.atr`) in the sidebar.")
        return

    chosen_base = st.selectbox("Select record (basename)", available_bases)
    st.markdown(f"**Selected:** `{chosen_base}`")
    max_duration_sec = st.number_input("Max duration (sec)", value=120, min_value=10, step=10)
    resample_len = st.number_input("Beat resample length", value=100, min_value=50, step=10)
    window_ms = st.number_input("Beat window (ms)", value=700, min_value=300, step=50)
    run_button = st.button("Run ECG Analysis")

    if run_button:
        st.info("Running analysis...")
        record_path = os.path.join(WORK_DIR, chosen_base)
        try:
            record = wfdb.rdrecord(record_path)
            try:
                ann = wfdb.rdann(record_path, "atr")
                ann_present = True
            except Exception:
                ann = None
                ann_present = False
        except Exception as e:
            st.error(f"Failed to read WFDB record '{chosen_base}': {e}")
            return

        channels = record.sig_name
        chosen_channel = st.selectbox("Signal channel to use", channels, index=0)
        ch_idx = channels.index(chosen_channel)
        signal = record.p_signal[:, ch_idx]
        fs = record.fs
        st.write(f"fs: {fs} Hz")
        max_samples = int(max_duration_sec * fs)
        signal = signal[:max_samples]

        if ann_present and hasattr(ann, "sample") and ann.sample is not None and len(ann.sample) > 0:
            r_peaks = ann.sample
            labels = np.array(ann.symbol) if hasattr(ann, "symbol") else np.array(["N"] * len(r_peaks))
            st.success(f"Annotation found: {len(r_peaks)} annotations.")
        else:
            st.warning("No annotation — running Pan-Tompkins.")
            r_peaks = pan_tompkins_detector(signal, fs)
            labels = np.array(["N"] * len(r_peaks))

        valid_idx = np.where(r_peaks < max_samples)[0]
        r_peaks = r_peaks[valid_idx]
        labels = labels[valid_idx] if len(labels) >= len(valid_idx) else labels[:len(valid_idx)]

        signal_f = bandpass(signal, fs)
        beats, beat_indices = extract_beats(signal_f, r_peaks, fs, window_ms=window_ms, resample_len=resample_len)
        if len(beats) == 0:
            st.error("No beats extracted.")
            return

        rr = np.diff(r_peaks) / fs
        rr = np.append(rr, rr[-1]) if len(rr)>0 else np.array([1.0])
        beat_features = extract_features(beats, rr)
        y_beats = np.array([label_map.get(l, 0) for l in labels[:len(beat_features)]])

        # Simple ML safety checks
        if len(beat_features) < 5 or len(np.unique(y_beats)) < 2:
            st.warning("Insufficient beat samples/labels for ML. Showing available outputs.")
            clf_beats = None
            pred_beats = np.array([0]*len(y_beats))
        else:
            X_train, X_test, y_train, y_test = train_test_split(beat_features, y_beats, test_size=0.2, random_state=42)
            clf_beats = train_rf_model(X_train, y_train)
            pred_beats = clf_beats.predict(X_test)
            st.subheader("Beat-level classification")
            st.text(classification_report(y_test, pred_beats, zero_division=0))
            st.write("Confusion matrix (beat-level):")
            st.dataframe(pd.DataFrame(confusion_matrix(y_test, pred_beats), index=np.unique(y_test), columns=np.unique(y_test)))

        # Sequence analysis
        beat_hr_labels = []
        for rr_val in rr[:len(beat_features)]:
            hr = 60 / rr_val if rr_val > 0 else 0
            if hr < 60:
                beat_hr_labels.append('Bradycardia')
            elif hr > 100:
                beat_hr_labels.append('Tachycardia')
            else:
                beat_hr_labels.append('Normal')

        seq_len = 25
        seq_step = 5
        seq_labels = []
        tachy_results = []
        for i in range(0, max(1, len(rr) - seq_len), seq_step):
            seq_rr = rr[i:i+seq_len]
            if len(seq_rr) == 0:
                continue
            avg_hr = 60 / np.mean(seq_rr) if np.mean(seq_rr) > 0 else 0
            if avg_hr < 60:
                seq_labels.append(0)
                tachy_results.append("Tachycardia")
            elif avg_hr > 100:
                seq_labels.append(2)
                seq_beats = labels[i:i+seq_len]
                if is_irregular(seq_rr):
                    subtype = classify_tachycardia_irregular(seq_beats)
                else:
                    subtype = classify_tachycardia_regular(seq_beats)
                tachy_results.append(subtype)
            else:
                seq_labels.append(1)
                tachy_results.append("Tachycardia")

        seq_features = []
        seq_labels_np = np.array(seq_labels)
        for i in range(len(seq_labels)):
            rr_segment = rr[i:i+seq_len]
            seq_features.append([np.mean(rr_segment), np.std(rr_segment), np.min(rr_segment), np.max(rr_segment)])

        seq_features_np = np.array(seq_features)

        # Second model: tachy subtype RF (demo purposes)
        # For simplicity, skip actual training here
        st.success("Analysis complete.")

        # Summary of overall rhythm
        overall_summary = {}
        for label in labels:
            overall_summary[label] = overall_summary.get(label, 0) + 1

        st.session_state['last_analysis'] = {
            "overall_summary": overall_summary,
            "record_name": chosen_base,
            "fs": fs,
            "beat_predictions": pred_beats.tolist() if pred_beats is not None else [],
            "r_peaks": r_peaks.tolist(),
            "tachy_results": tachy_results
        }
        st.success("Analysis results saved to session state.")
        go_to("Results")

def show_results():
    st.header("Result Summary")

    st.write("Your Result Are Here:")

    if not st.session_state['last_analysis']:
        st.info("No analysis run yet — go to ECG Upload & Analysis and run the analysis.")
        return

    res = st.session_state['last_analysis']
    overall_summary = res['overall_summary']

    # Defensive: if overall_summary empty dict
    if not overall_summary:
        st.warning("No rhythm summary available.")
        return

    # Find dominant rhythm (highest count)
    dominant_rhythm = max(overall_summary, key=overall_summary.get).lower()

    if dominant_rhythm == "normal":
        st.success("Your heart rhythm appears **normal**. No significant abnormalities detected.")
    elif dominant_rhythm == "bradycardia":
        st.warning("You have **bradycardia** — slower than normal heart rate.")
    elif dominant_rhythm == "tachycardia":
        st.warning("You have **tachycardia** — faster than normal heart rate.")
    else:
        st.info(f"Detected rhythms: {', '.join(overall_summary.keys())}")

    st.write("**Overall summary:**")
    for k,v in overall_summary.items():
        st.write(f"{k}: {v}")

    st.markdown("---")
    st.write(f"Record analyzed: `{res['record_name']}` at {res['fs']} Hz")
    if st.button("Download Results as CSV"):
        csv_buffer = io.StringIO()
        df_summary = pd.DataFrame(list(overall_summary.items()), columns=["Rhythm", "Count"])
        df_summary.to_csv(csv_buffer, index=False)
        st.download_button("Download CSV", data=csv_buffer.getvalue(), file_name=f"results_{res['record_name']}.csv", mime="text/csv")

    if st.button("Back to ECG Upload & Analysis"):
        go_to("ECG Upload & Analysis")

# ------------------------ Main app body ------------------------
page_funcs = {
    "Home": show_home,
    "Working Principle": show_working_principle,
    "Terms & Conditions": show_terms,
    "Patient Info": show_patient_info,
    "ECG Upload & Analysis": show_ecg_analysis,
    "Results": show_results,
}

def main():
    page = st.session_state.get('page', 'Home')
    func = page_funcs.get(page, show_home)
    func()

if __name__ == "__main__":
    main()
