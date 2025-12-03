from fastapi import FastAPI
import torch

app = FastAPI()

def generate_simple():
    x = torch.linspace(-2, 2, 200).unsqueeze(1)
    y = torch.sin(3 * x) + 0.3 * torch.randn_like(x)
    return x, y

@app.get("/rawdata")
def rawdata():
    x, y = generate_simple()
    return {"x": x.squeeze().tolist(), "y_true": y.squeeze().tolist()}


@app.get("/rawdata/complex")
def rawdata_complex(seed: int = 0):
    torch.manual_seed(seed)
    x = torch.linspace(-2, 2, 400).unsqueeze(1)
    y = torch.cos(5 * x) + 0.3 * torch.sin(2 * x) + 0.2 * torch.randn_like(x)
    return {"x": x.squeeze().tolist(), "y_true": y.squeeze().tolist()}