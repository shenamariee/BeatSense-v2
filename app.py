import streamlit as st

st.set_page_config(page_title="BeatSense", layout="wide")

# --------------------------
# NAVIGATION SIDEBAR
# --------------------------
with st.sidebar:
    st.image("BeatSense logo.jpg", use_column_width=True)
    st.markdown("### Navigation")
    page = st.radio(
        "Go to:",
        ["Home", "Working Principle", "Terms & Conditions", "Patient Info", 
         "ECG Analysis", "Results", "Thank You"]
    )

# Store session state
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {}
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "ecg_summary" not in st.session_state:
    st.session_state.ecg_summary = ""


# --------------------------
# PAGE 1: HOME
# --------------------------
if page == "Home":
    st.title("Welcome to **BeatSense**")
    st.markdown("""
        BeatSense is an academic ECG interpretation tool designed to help users 
        understand basic cardiac rhythm patterns.

        Use the sidebar to navigate through the app.
    """)

    st.markdown("### Proceed to the Working Principle")
    if st.button("➡ Continue"):
        st.session_state['go_to_page'] = "Working Principle"

    if 'go_to_page' in st.session_state:
        page = st.session_state['go_to_page']


# --------------------------
# PAGE 2: WORKING PRINCIPLE
# --------------------------
elif page == "Working Principle":
    st.title("How BeatSense Works")

    st.markdown("""
    BeatSense operates by processing uploaded ECG signals and analyzing the waveform 
    characteristics.

    **Core steps:**
    - Preprocessing of the raw ECG (filtering, denoising)
    - Feature extraction (peak detection, RR intervals, variability)
    - Pattern classification using rule-based or ML-assisted algorithms
    - Output summary indicating rhythm characteristics
    """)

    if st.button("➡ Continue to Terms & Conditions"):
        st.session_state['go_to_page'] = "Terms & Conditions"


# --------------------------
# PAGE 3: TERMS & CONDITIONS
# --------------------------
elif page == "Terms & Conditions":
    st.title("Terms & Conditions")

    st.markdown("""
    **Please read before continuing.**

    BeatSense is **strictly for academic purposes only**.  
    It is **not a medical device** and **does not provide real clinical diagnosis**.

    All outputs are algorithm-generated summaries and should not be used as 
    a substitute for professional medical evaluation.
    """)

    agree = st.checkbox("I understand and agree")

    if agree and st.button("Proceed to Patient Info"):
        st.session_state['go_to_page'] = "Patient Info"


# --------------------------
# PAGE 4: PATIENT INFO
# --------------------------
elif page == "Patient Info":
    st.title("Patient Information")

    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

    if st.button("Save & Continue to ECG Analysis"):
        st.session_state.patient_info = {
            "name": name,
            "age": age,
            "gender": gender
        }
        st.session_state['go_to_page'] = "ECG Analysis"


# --------------------------
# PAGE 5: ECG ANALYSIS
# --------------------------
elif page == "ECG Analysis":
    st.title("Upload ECG Signal")

    st.markdown("Upload the .dat, .hea, .atr, or .csv file here.")

    uploaded = st.file_uploader("Select ECG File", type=["dat", "hea", "atr", "csv"])

    if uploaded:
        st.success("File uploaded successfully!")

        # -------------------------------
        # INSERT YOUR ORIGINAL WORKING ECG CODE HERE
        # (Preprocessing, feature extraction, classification)
        # You will assign the summary result like:
        # st.session_state.ecg_summary = "Normal Sinus Rhythm" or other result
        # -------------------------------

        # TEMP SAMPLE OUTPUT (replace with real output)
        st.session_state.ecg_summary = "Normal Sinus Rhythm (Sample Output)"
        st.session_state.analysis_done = True

        if st.button("View Results"):
            st.session_state['go_to_page'] = "Results"


# --------------------------
# PAGE 6: RESULTS
# --------------------------
elif page == "Results":
    st.title("ECG Analysis Summary")

    if not st.session_state.analysis_done:
        st.warning("No ECG analysis done yet.")
    else:
        st.subheader("Overall Interpretation:")
        st.success(st.session_state.ecg_summary)

        if "normal" in st.session_state.ecg_summary.lower():
            st.info("Your heart rhythm appears healthy based on this analysis.")
        else:
            st.error("Please consult a medical professional for proper evaluation.")

    if st.button("Finish"):
        st.session_state['go_to_page'] = "Thank You"


# --------------------------
# PAGE 7: THANK YOU
# --------------------------
elif page == "Thank You":
    st.title("Thank You for Using BeatSense!")
    st.markdown("""
    We appreciate your time and hope BeatSense helped you understand ECG signals better.

    **Remember:** This tool is for academic use only.
    """)

    st.markdown("You may return to Home at any time using the sidebar.")


# --------------------------
# HANDLE PAGE REDIRECTION
# --------------------------
if "go_to_page" in st.session_state:
    st.session_state.page = st.session_state.go_to_page
