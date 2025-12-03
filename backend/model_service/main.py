from fastapi import FastAPI
import torch
import torch.nn as nn
import torch.optim as optim
import threading, time

app = FastAPI()

# Global state
model, optimizer, losses, epoch, config = None, None, [], 0, {}
DEVICE = torch.device("cpu")

# Cached dataset
x_data, y_data = None, None

# Simulation state
sim_thread, sim_running = None, False

def build_model(input_dim=1, output_dim=1, num_layers=2, units=16, activation="ReLU"):
    act = {"ReLU": nn.ReLU(), "Tanh": nn.Tanh(), "Sigmoid": nn.Sigmoid()}[activation]
    layers = [nn.Linear(input_dim, units)]
    for _ in range(num_layers - 1):
        layers += [act, nn.Linear(units, units)]
    layers += [act, nn.Linear(units, output_dim)]
    return nn.Sequential(*layers)

def generate_data(reset=False):
    global x_data, y_data
    if x_data is None or y_data is None or reset:
        x_data = torch.linspace(-2, 2, 200).unsqueeze(1).to(DEVICE)
        y_data = torch.sin(3 * x_data) + 0.3 * torch.randn_like(x_data)
    return x_data, y_data

@app.post("/reset_data")
def reset_data():
    generate_data(reset=True)
    return {"status": "data reset"}

@app.post("/init")
def init_model(num_layers: int = 2, units: int = 16, activation: str = "ReLU", lr: float = 0.01):
    global model, optimizer, losses, epoch, config
    model = build_model(1, 1, num_layers, units, activation).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses, epoch = [], 0
    config = {"num_layers": num_layers, "units": units, "activation": activation, "lr": lr}
    generate_data(reset=True)  # reset dataset on init
    return {"status": "initialized", "config": config}

@app.post("/train_one_epoch")
def train_one_epoch():
    global model, optimizer, losses, epoch
    if model is None:
        return {"error": "Model not initialized"}
    x, y = generate_data()
    criterion = nn.MSELoss()
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    losses.append(float(loss.item()))
    epoch += 1
    return {"epoch": epoch, "loss": losses[-1]}

@app.post("/train_epochs")
def train_epochs(n: int = 5):
    global model, optimizer, losses, epoch
    if model is None:
        return {"error": "Model not initialized"}
    x, y = generate_data()
    criterion = nn.MSELoss()
    for _ in range(n):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        epoch += 1
    return {"epochs": epoch, "final_loss": losses[-1], "losses": losses}

@app.get("/curvefit")
def curvefit():
    global model
    if model is None:
        return {"error": "Model not initialized"}
    x, y = generate_data()
    with torch.no_grad():
        y_pred = model(x)
    return {
        "x": x.squeeze().cpu().tolist(),
        "y_true": y.squeeze().cpu().tolist(),
        "y_pred": y_pred.squeeze().cpu().tolist(),
    }

@app.get("/summary")
def summary():
    return {"epochs": epoch, "final_loss": (losses[-1] if losses else None), "losses": losses}

@app.get("/architecture")
def architecture():
    if model is None:
        return {"error": "Model not initialized"}
    return config

# --- Simulation control ---
def run_simulation():
    global sim_running
    while sim_running:
        train_one_epoch()
        time.sleep(1)

@app.post("/start_sim")
def start_sim():
    global sim_thread, sim_running
    if not sim_running:
        sim_running = True
        sim_thread = threading.Thread(target=run_simulation, daemon=True)
        sim_thread.start()
    return {"status": "simulation started"}

@app.post("/stop_sim")
def stop_sim():
    global sim_running
    sim_running = False
    return {"status": "simulation stopped"}