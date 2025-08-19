import base64

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Model Extraction Attack Demo", layout="wide")
st.title("🛡️ Model Extraction Attack Simulation")

st.markdown("""
This app demonstrates a **model extraction attack** on a facial recognition system. An attacker queries a black-box target model, collects input-output pairs, and trains a surrogate (stolen) model to mimic the target. 
""")

with st.expander("ℹ️ What is a Model Extraction Attack?", expanded=True):
    st.markdown("""
    - **Black-box access**: Attacker can only query the model, not see its internals.
    - **Goal**: Steal the model's functionality by training a surrogate using collected queries.
    - **Steps**:
        1. Query the target model with inputs (embeddings).
        2. Collect outputs (labels/predictions).
        3. Train a surrogate model on the stolen dataset.
        4. Compare surrogate vs. target accuracy and fidelity.
    """)

st.header("1️⃣ Experiment Results: Surrogate Model Training")

# Show experiment results table
csv_path = "results/surrogate_experiment_metrics.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("Experiment results CSV not found.")




# Carousel-style graph viewer with arrow buttons
st.header("2️⃣ Experiment Graphs Carousel")
plot_files = [
    ("results/acc_vs_fid_line.png", "Accuracy vs Fidelity"),
    ("results/acc_by_arch_noise.png", "Accuracy by Architecture & Noise"),
    ("results/accuracy_vs_querysize.png", "Accuracy vs Query Size"),
    ("results/fidelity_vs_querysize.png", "Fidelity vs Query Size")
]

if "carousel_idx" not in st.session_state:
    st.session_state.carousel_idx = 0


# Wider center column for bigger image
cols = st.columns([1, 10, 1])
with cols[0]:
    if st.button("←", key="left"):
        st.session_state.carousel_idx = max(0, st.session_state.carousel_idx - 1)
with cols[2]:
    if st.button("→", key="right"):
        st.session_state.carousel_idx = min(len(plot_files)-1, st.session_state.carousel_idx + 1)
with cols[1]:
    plot_path, caption = plot_files[st.session_state.carousel_idx]
    if os.path.exists(plot_path):
        with open(plot_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()
        st.markdown(f"""
        <div style='display: flex; flex-direction: column; align-items: center;'>
            <img src='data:image/png;base64,{img_base64}' width='650'/>
            <div style='margin-top: 8px; color: #888;'>{caption}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning(f"{caption} plot not found.")

st.header("3️⃣ Simulate Model Extraction Attack")
st.markdown("""
You can run the attack simulation below. This will:
- Query the target model to steal outputs
- Train a surrogate (stolen) model
- Show the test accuracy of the stolen model
""")

import subprocess

if st.button("Run Attack Simulation (Query Target Model)"):
    result = subprocess.run(["python", "src/simulate_attack.py"], capture_output=True, text=True)
    st.code(result.stdout)
    if result.stderr:
        st.error(result.stderr)

if st.button("Train Stolen Surrogate Model"):
    result = subprocess.run(["python", "src/train_stolen_surrogate.py"], capture_output=True, text=True)
    st.code(result.stdout)
    if result.stderr:
        st.error(result.stderr)

st.header("4️⃣ Compare Target vs Stolen Model")
st.markdown("""
You can compare the predictions and fidelity of the target and stolen models using the experiment results and plots above.
""")


# ------------------ Streamlit UI ------------------
