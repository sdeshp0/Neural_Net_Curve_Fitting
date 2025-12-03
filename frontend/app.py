import os
import requests
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="Neural Net Curve Fitting", layout="wide")

# --- Initialize session state keys ---
if "sim_on" not in st.session_state:
    st.session_state.sim_on = False

# Backend URLs (injected via env or defaults)
DATA_URL = os.getenv("DATA_URL", "http://localhost:8001")
MODEL_URL = os.getenv("MODEL_URL", "http://localhost:8000")

st.title("Neural Network Curve Fitting – Microservices")
st.caption("Frontend: Streamlit | Backends: data_service + model_service")

# Sidebar hyperparameters
st.sidebar.header("Model hyperparameters")
num_layers = st.sidebar.number_input("Hidden layers", min_value=1, max_value=6, value=2, step=1, key="layers")
units = st.sidebar.number_input("Units per hidden layer", min_value=1, max_value=128, value=16, step=1, key="units")
activation = st.sidebar.selectbox("Activation", ["ReLU", "Tanh", "Sigmoid"], index=0, key="activation")
lr = st.sidebar.number_input("Learning rate", min_value=0.0001, max_value=0.1, value=0.01, step=0.001, format="%.4f", key="lr")

# --- Reset dataset button ---
if st.sidebar.button("Reset dataset", key="reset_btn"):
    try:
        requests.post(f"{MODEL_URL}/reset_data", timeout=5)
        st.success("Dataset reset successfully.")
    except Exception as e:
        st.error(f"Reset error: {e}")

status = st.empty()

# --- Helper functions ---
def get_raw():
    try:
        r = requests.get(f"{DATA_URL}/rawdata", timeout=5)
        return r.json()
    except Exception as e:
        status.error(f"Raw data error: {e}")
        return {"x": [], "y_true": []}

def get_curvefit():
    try:
        r = requests.get(f"{MODEL_URL}/curvefit", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_architecture():
    try:
        r = requests.get(f"{MODEL_URL}/architecture", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def get_summary():
    try:
        r = requests.get(f"{MODEL_URL}/summary", timeout=5)
        return r.json()
    except Exception as e:
        return {"epochs": 0, "final_loss": None, "losses": []}

def draw_curve(fig_area, x, y_true, y_pred=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y_true, mode="markers", name="Raw/True", opacity=0.6, marker=dict(size=6)))
    if y_pred is not None:
        fig.add_trace(go.Scatter(x=x, y=y_pred, mode="lines", name="Predicted", line=dict(width=2)))
        fig.update_layout(title="Curve fit", xaxis_title="x", yaxis_title="y")
    else:
        fig.update_layout(title="Raw data (init + train to fit)", xaxis_title="x", yaxis_title="y")
    fig_area.plotly_chart(fig, use_container_width=True)

def draw_nn(fig_area, cfg):
    nl = cfg.get("num_layers", 2)
    u = cfg.get("units", 16)
    act = cfg.get("activation", "ReLU")
    layer_sizes = [1] + [u] * nl + [1]
    x_spacing, y_spacing = 1.5, 0.5
    nodes_x, nodes_y, text = [], [], []
    edges_x, edges_y = [], []

    for i, size in enumerate(layer_sizes):
        x = i * x_spacing
        for j in range(size):
            y = j * y_spacing - (size - 1) * y_spacing / 2
            nodes_x.append(x)
            nodes_y.append(y)
            label = "Input" if i == 0 else ("Output" if i == len(layer_sizes)-1 else f"Hidden {i} • {act}")
            text.append(label)
            if i > 0:
                for k in range(layer_sizes[i - 1]):
                    y_prev = k * y_spacing - (layer_sizes[i - 1] - 1) * y_spacing / 2
                    edges_x += [x - x_spacing, x, None]
                    edges_y += [y_prev, y, None]

    edge_trace = go.Scatter(x=edges_x, y=edges_y, line=dict(width=0.5, color='gray'), hoverinfo='none', mode='lines')
    node_trace = go.Scatter(x=nodes_x, y=nodes_y, mode='markers',
                            marker=dict(size=12, color='skyblue', line=dict(width=1, color='black')),
                            text=text, hoverinfo='text')
    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=f"Architecture ({act})", showlegend=False,
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=350)
    fig_area.plotly_chart(fig, use_container_width=True)

def draw_loss_curve(fig_area, losses):
    if not losses:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(losses)+1)), y=losses,
                             mode="lines+markers", name="Loss", line=dict(color="red")))
    fig.update_layout(title="Training Loss vs Epochs", xaxis_title="Epoch", yaxis_title="Loss")
    fig_area.plotly_chart(fig, use_container_width=True)

# --- Buttons ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    init = st.button("Init model", key="init_btn")
with col2:
    train_one = st.button("Train one epoch", key="train_btn")
with col3:
    start_sim = st.button("Start simulation", key="start_btn")
with col4:
    stop_sim = st.button("Stop simulation", key="stop_btn")

# --- Actions ---
if init:
    try:
        r = requests.post(f"{MODEL_URL}/init",
                          params={"num_layers": num_layers, "units": units,
                                  "activation": activation, "lr": lr}, timeout=10)
        status.success(f"Model initialized: {num_layers} layers × {units} units, {activation}, lr={lr}")
    except Exception as e:
        status.error(f"Init error: {e}")

if train_one:
    try:
        r = requests.post(f"{MODEL_URL}/train_one_epoch", timeout=10).json()
        if "loss" in r:
            status.info(f"Epoch {r.get('epoch')}, loss {r['loss']:.4f}")
        else:
            status.warning(r)
    except Exception as e:
        status.error(f"Train error: {e}")

if start_sim:
    try:
        requests.post(f"{MODEL_URL}/start_sim", timeout=5)
        st.session_state.sim_on = True
        status.info("Simulation started.")
    except Exception as e:
        status.error(f"Start sim error: {e}")

if stop_sim:
    try:
        requests.post(f"{MODEL_URL}/stop_sim", timeout=5)
        st.session_state.sim_on = False
        status.info("Simulation stopped.")
    except Exception as e:
        status.error(f"Stop sim error: {e}")

# --- Auto-refresh while simulation is running ---
if st.session_state.sim_on:
    st_autorefresh(interval=2000, key="sim_refresh")

# --- Layout ---
left, right = st.columns(2)
curve = get_curvefit()
if "error" in curve:
    raw = get_raw()
    draw_curve(left, raw.get("x", []), raw.get("y_true", []), None)
else:
    draw_curve(left, curve.get("x", []), curve.get("y_true", []), curve.get("y_pred", []))

arch = get_architecture()
draw_nn(right, arch if "error" not in arch else {"num_layers": num_layers, "units": units, "activation": activation})

summ = get_summary()
st.markdown(f"**Epochs:** {summ.get('epochs', 0)} | **Final loss:** {summ.get('final_loss', 'N/A')}")

# --- Loss curve plot ---
draw_loss_curve(st, summ.get("losses", []))