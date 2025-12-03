import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

# 📦 Setup
st.set_page_config(page_title="Neural Network Visualizer", layout="wide")
st.title("🧠 Neural Network Training Visualizer")

# 🧠 Session State Initialization
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.losses = []
    st.session_state.epoch = 0
    st.session_state.x_train = None
    st.session_state.y_train = None
    st.session_state.y_pred = None
    st.session_state.auto_training = False

if "activations" not in st.session_state:
    st.session_state.activations = {}

if "gradient_magnitudes" not in st.session_state:
    st.session_state.gradient_magnitudes = {}


# 🧪 Simple Data Generator
def generate_data():
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = torch.sin(3 * x) + 0.3 * torch.randn_like(x)
    return x, y


# 🧪 Complex Data Generator
def generate_complex_data():
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    x_np = x.numpy().flatten()
    noise = 0.3 * np.random.randn(len(x_np))

    funcs = [
        lambda x: np.sin(3 * x),
        lambda x: np.cos(5 * x),
        lambda x: np.exp(-x ** 2),
        lambda x: x ** 2,
        lambda x: np.sign(x) * np.sqrt(np.abs(x)),
        lambda x: np.sin(x) * np.cos(2 * x),
        lambda x: np.log(np.abs(x) + 1),
        lambda x: np.sin(x ** 2),
    ]

    selected_funcs = np.random.choice(funcs, size=np.random.randint(2, 4), replace=False)
    y_np = sum(f(x_np) for f in selected_funcs) + noise
    y = torch.tensor(y_np, dtype=torch.float32).unsqueeze(1)
    return x, y


# 🧠 Model Builder
def build_model(input_dim, output_dim, num_layers, units, activation):
    layers = [nn.Linear(input_dim, units)]
    act_fn = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "Sigmoid": nn.Sigmoid()}[activation]
    for _ in range(num_layers - 1):
        layers += [act_fn, nn.Linear(units, units)]
    layers += [act_fn, nn.Linear(units, output_dim)]
    return nn.Sequential(*layers)


# 📉 Loss Plot
def plot_loss_curve(losses, threshold=None, epoch=None):
    fig, ax = plt.subplots()
    if losses:
        current_loss = losses[-1]
        ax.plot(losses, label=f"Loss (current: {current_loss:.4f})")
        if threshold:
            ax.axhline(threshold, color="red", linestyle="--", label=f"Threshold: {threshold:.4f}")
        ax.set_title(f"Loss Curve — Epoch {epoch}")
    else:
        ax.text(0.5, 0.5, "No training yet", ha="center", va="center", fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig


# 📈 Prediction Plot
def plot_predictions(x, y_true, y_pred):
    fig, ax = plt.subplots()
    if y_pred is not None:
        ax.scatter(x.numpy(), y_true.numpy(), label="True", alpha=0.5)
        ax.scatter(x.numpy(), y_pred.detach().numpy(), label="Predicted", alpha=0.5)
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No predictions yet", ha="center", va="center", fontsize=12)
    return fig

# 📈 Activation Heat Map Grid

def plot_activation_grid(activations_dict):
    layer_names = list(activations_dict.keys())
    n = len(layer_names)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.5, 2.5))

    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, layer_names):
        act = activations_dict[name]
        if act.dim() == 2:
            act = act[0]  # batch x neurons
        elif act.dim() == 4:
            act = act[0].mean(dim=(1, 2))  # conv: batch x channels x H x W → channels

        sns.heatmap(act.unsqueeze(0).detach().cpu().numpy(), cmap="viridis", ax=ax, cbar=False)
        ax.set_title(name, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_activation_histograms(activations_dict):
    fig, axes = plt.subplots(1, len(activations_dict), figsize=(len(activations_dict) * 3, 2.5))
    for ax, (name, act) in zip(axes, activations_dict.items()):
        flat = act.flatten().numpy()
        ax.hist(flat, bins=30, color='skyblue')
        ax.set_title(name, fontsize=8)
    plt.tight_layout()
    return fig


# 📈 Gradient Magnitudes

def plot_gradient_magnitudes():
    if not st.session_state.gradient_magnitudes:
        st.warning("No gradients recorded yet.")
        return

    names = list(st.session_state.gradient_magnitudes.keys())
    values = list(st.session_state.gradient_magnitudes.values())

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(names, values)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel("Mean Gradient Magnitude")
    ax.set_title("Gradient Flow per Layer")
    st.pyplot(fig)

# 📋 Summary
def show_summary(model, losses, epoch):
    final_loss = f"{losses[-1]:.4f}" if losses else "N/A"
    st.markdown(f"""
### 📋 Training Summary
- Epochs Trained: **{epoch}**
- Final Loss: **{final_loss}**  
- Model Layers: **{len(list(model.children()))}**
""")

# 🧩 Architecture Diagram
def draw_model_architecture(num_layers, units, activation):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.axis('off')

    layer_sizes = [1] + [units] * num_layers + [1]
    layer_labels = ['Input'] + [f'Hidden {i + 1}' for i in range(num_layers)] + ['Output']

    x_spacing = 1.2
    y_spacing = 0.5

    for i, (size, label) in enumerate(zip(layer_sizes, layer_labels)):
        x = i * x_spacing
        for j in range(size):
            y = j * y_spacing - (size - 1) * y_spacing / 2
            circle = plt.Circle((x, y), 0.15, color='skyblue', ec='black')
            ax.add_patch(circle)
            if i > 0:
                for k in range(layer_sizes[i - 1]):
                    y_prev = k * y_spacing - (layer_sizes[i - 1] - 1) * y_spacing / 2
                    ax.plot([x - x_spacing, x], [y_prev, y], color='gray', linewidth=0.5)

        ax.text(x, max(size * y_spacing, 1.2), label, ha='center', fontsize=9)

    ax.set_xlim(-0.5, x_spacing * (len(layer_sizes)))
    ax.set_ylim(-2, 2)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

# 🧠 Neural Network Activations

def get_activation(name):
    def hook(model, input, output):
        st.session_state.activations[name] = output.detach().cpu()
    return hook

def register_activation_hooks(model):
    for name, layer in model.named_modules():
        if isinstance(layer, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Linear)):
            layer.register_forward_hook(get_activation(name))

# 🧠 Neural Network Gradient Tracking

def track_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            st.session_state.gradient_magnitudes[name] = param.grad.abs().mean().item()


# 🧠 Training Function
def train_one_epoch_fn():
    model = st.session_state.model
    x = st.session_state.x_train
    y = st.session_state.y_train
    optimizer = st.session_state.optimizer
    criterion = nn.MSELoss()

    # 🔄 Clear previous activations
    st.session_state.activations.clear()

    # 🧠 Forward pass
    model.train()
    optimizer.zero_grad()
    y_pred = model(x)  # 🔥 This triggers hooks

    # 📉 Loss and backward
    loss = criterion(y_pred, y)
    loss.backward()

    # 📊 Track gradients
    track_gradients(model)

    # 🔁 Update weights
    optimizer.step()

    # 📈 Update session state
    st.session_state.losses.append(loss.item())
    st.session_state.epoch += 1
    st.session_state.y_pred = y_pred


# 🚀 Main App Logic
def main():
    # 🎛 Sidebar Configuration
    st.sidebar.header("🔧 Model Configuration")
    num_layers = st.sidebar.slider("Number of Hidden Layers", 1, 5, 2)
    units = st.sidebar.slider("Units per Layer", 1, 20, 8)
    activation = st.sidebar.selectbox("Activation Function", ["ReLU", "Tanh", "Sigmoid"])

    st.sidebar.header("🧪 Data Settings")
    data_mode = st.sidebar.selectbox("Data Complexity", ["Simple", "Complex"])
    regenerate_data = st.sidebar.button("🔁 Regenerate Data")

    st.sidebar.header("🧪 Training Parameters")
    min_loss = st.sidebar.number_input("Auto-Stop Loss Threshold", min_value=0.001, max_value=10.0, value=0.05,
                                       step=0.001, format="%.4f")
    delay = st.sidebar.number_input("Auto-Training Delay (sec)", min_value=0.0, max_value=5.0, value=0.1, step=0.05,
                                    format="%.2f")
    learning_rate = st.sidebar.number_input("Learning Rate", min_value=0.0001, max_value=1.0, value=0.01, step=0.001,
                                            format="%.4f")

    # 🧪 Data
    if st.session_state.x_train is None or regenerate_data:
        if data_mode == "Simple":
            x, y = generate_data()
        else:
            x, y = generate_complex_data()
        st.session_state.x_train = x
        st.session_state.y_train = y
        st.session_state.losses = []
        st.session_state.epoch = 0
        st.session_state.y_pred = None

    # 📈 Raw Data Preview (Compact)
    with st.expander("📈 Raw Data Preview", expanded=False):
        fig, ax = plt.subplots(figsize=(4, 3))  # Smaller figure size
        ax.scatter(st.session_state.x_train.numpy(), st.session_state.y_train.numpy(), alpha=0.6, s=10)
        ax.set_title(f"{data_mode} Data Sample", fontsize=10)
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)
        ax.tick_params(labelsize=8)
        st.pyplot(fig, use_container_width=False)

    # 🧩 Visualize Model Architecture
    with st.expander("🧩 Visualize Model Architecture", expanded=False):
        st.markdown(f"""
        ### 🧩 Layer-wise Topology

        This schematic shows the feedforward structure of your model.  
        Each node represents a neuron, and each edge a learnable weight.

        - Input layer: 1 feature  
        - Hidden layers: {num_layers} × {units} units  
        - Output layer: 1 regression output

        The diagram updates dynamically based on your configuration.
                """)
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        draw_model_architecture(num_layers, units, activation)
        st.markdown("</div>", unsafe_allow_html=True)

    # 🧠 Model
    if st.session_state.model is None:
        model = build_model(1, 1, num_layers, units, activation)
        register_activation_hooks(model)
        st.session_state.model = model
        st.session_state.optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # 🎛 Training Controls
    st.subheader("🧪 Training Controls")
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    with col_btn1:
        train_one_epoch = st.button("Train One Epoch")
    with col_btn2:
        if st.button("Start Auto Training"):
            st.session_state.auto_training = True
    with col_btn3:
        if st.button("Stop Auto Training"):
            st.session_state.auto_training = False
    with col_btn4:
        if st.button("Reset Model"):
            st.session_state.model = None
            st.session_state.losses = []
            st.session_state.epoch = 0
            st.session_state.y_pred = None
            st.session_state.x_train = None
            st.session_state.y_train = None
            st.rerun()

    # 🔀 Toggle for activation view
    activation_view = st.radio(
        "Activation View",
        ["Heatmap", "Histogram"],
        index=0,
        horizontal=True,
        key="activation_view"
    )

    # 📊 Persistent Side-by-Side Chart Containers
    col1, col2, col3 = st.columns(3)
    loss_chart = col1.empty()
    pred_chart = col2.empty()
    activation_chart = col3.empty()

    if train_one_epoch:
        train_one_epoch_fn()

    if st.session_state.auto_training:
        while st.session_state.auto_training and (
                len(st.session_state.losses) == 0 or st.session_state.losses[-1] > min_loss
        ) and st.session_state.epoch < 1000:
            train_one_epoch_fn()
            loss_chart.pyplot(
                plot_loss_curve(st.session_state.losses, threshold=min_loss, epoch=st.session_state.epoch))
            pred_chart.pyplot(
                plot_predictions(st.session_state.x_train, st.session_state.y_train, st.session_state.y_pred))
            if st.session_state.activations:
                if st.session_state.activation_view == "Heatmap":
                    fig = plot_activation_grid(st.session_state.activations)
                else:
                    fig = plot_activation_histograms(st.session_state.activations)
                activation_chart.pyplot(fig)

            time.sleep(delay)
    else:
        loss_chart.pyplot(plot_loss_curve(st.session_state.losses, threshold=min_loss, epoch=st.session_state.epoch))
        pred_chart.pyplot(plot_predictions(st.session_state.x_train, st.session_state.y_train, st.session_state.y_pred))
        if st.session_state.activations:
            if st.session_state.activation_view == "Heatmap":
                fig = plot_activation_grid(st.session_state.activations)
            else:
                fig = plot_activation_histograms(st.session_state.activations)
            activation_chart.pyplot(fig)

    # 📋 Summary
    show_summary(st.session_state.model, st.session_state.losses, st.session_state.epoch)

    # 🔍 Model Training Deep Dive

    st.subheader("📉 Gradient Magnitudes")
    if st.button("Show Gradient Magnitudes"):
        plot_gradient_magnitudes()

    # 📚 Technical Explanation
    with st.expander("🧠 What Happens During Training?", expanded=False):
        st.markdown(r"""
        ### 🧠 Training Workflow: Forward & Backward Pass

        Each training iteration involves a full pass through the model and a gradient-based update:

        **1. Forward Pass**  
        Inputs are propagated through each layer using affine transformations followed by non-linear activations.  
        The output is the model’s current prediction:  
        $$ \hat{y} = f(x; \theta) $$

        **2. Loss Computation**  
        The prediction is compared to the ground truth using a differentiable loss function (MSE in this case):  
        $$ \mathcal{L} = \frac{1}{n} \sum (y - \hat{y})^2 $$

        **3. Backward Pass (Backpropagation)**  
        Using automatic differentiation, gradients of the loss with respect to each parameter are computed via the chain rule:  
        $$ \frac{\partial \mathcal{L}}{\partial \theta_i} $$

        **4. Parameter Update**  
        An optimizer (Adam here) updates weights using gradients and internal momentum buffers:  
        $$ \theta_i \leftarrow \theta_i - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta_i} $$

        **5. Repeat**  
        Each epoch refines the model’s parameters to minimize the loss over the training set.
                """)

    with st.expander("🧠 How Model Configuration Affects Learning", expanded=False):
        st.markdown(f"""
        ### 🔧 Architectural Hyperparameters

        You've selected:
        - **{num_layers} hidden layers**
        - **{units} units per layer**
        - **{activation} activation function**

        **Hidden Layers**  
        Each hidden layer adds a level of abstraction, allowing the model to learn hierarchical representations.  
        More layers increase expressiveness but also risk vanishing gradients and overfitting.

        **Units per Layer**  
        Units define the dimensionality of the feature space at each layer.  
        Too few units may underfit; too many may overfit or slow convergence.

        **Activation Function**  
        - **ReLU**: Sparse activations, fast convergence, but can suffer from dead neurons.  
        - **Tanh**: Smooth and zero-centered, but prone to saturation.  
        - **Sigmoid**: Useful for probabilistic outputs, but slow and saturates easily.

        These choices directly affect gradient flow, convergence speed, and generalization.
                """)
    # End of main()

main()
