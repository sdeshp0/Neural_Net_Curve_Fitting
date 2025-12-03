# Neural Network Curve Fitting

## 📖 What this app does
This project provides an interactive dashboard for experimenting with neural network curve fitting.
    
Users can:
- Generate synthetic noisy sine wave data (or extend to complex functions)
- Initialize a neural network with chosen hyperparameters (layers, units, activation, learning rate)
- Train the model one epoch at a time or continuously via simulation
- Visualize both the raw data and the fitted curve as training progresses
- Track training progress with a loss curve plot (loss vs epoch)
- See a schematic of the neural network architecture that reflects the actual configuration
- Reset the dataset on demand to start a fresh experiment

A secondary purpose of this project is to recreate functionality of the original app using 
**Microservices Architecture**. The frontend (Streamlit) handles UI and visualization, while backend services handle data generation and model training.

### 📝 Note
    The original Streamlit app is present in the parent directory as streamlit_app.py 
    
    This app can be run using the following command:
    >>> streamlit run streamlit_app.py

    This should result in the following output:

        You can now view your Streamlit app in your browser.

        Local URL: http://localhost:8501
        Network URL: http://192.168.86.28:8501

---
## 🧩 Architecture Diagram

### Services Map
```
+-------------------+
|   Streamlit UI    |
|  (Frontend)       |
+---------+---------+
          |
          | REST calls
          v
+---------+---------+        +-------------------+
|   Data Service    |        |   Model Service   |
|  (FastAPI, 8001)  |        | (FastAPI, 8000)   |
+---------+---------+        +---------+---------+
          |                           |
          | provides synthetic        | handles init, training,
          | raw data (x, y_true)      | predictions, summaries
          |                           | continuous simulation,
          +-------------+-------------+ dataset reset, loss curve
                        |
                        v
                +-------------------+
                |   Streamlit UI    |
                |  displays charts  |
                |  auto-refreshes   |
                +-------------------+
```

### 🐳 Docker Compose (single host
```commandline
+---------------------------------------------------+
|                  Docker Host                      |
|                                                   |
|  +-------------------+   +-------------------+    |
|  |  Streamlit UI     |   |  Model Service    |    |
|  |  (container:8501) |   |  (container:8000) |    |
|  +---------+---------+   +---------+---------+    |
|            |                       |              |
|            v                       v              |
|  +-------------------+                          |
|  |  Data Service     |                          |
|  |  (container:8001) |                          |
|  +-------------------+                          |
|                                                   |
|   Networking: Docker bridge network,              |
|   services reachable via container name.          |
+---------------------------------------------------+
```
- All services run on **one machine**.
- Networking is handled by Docker’s internal bridge.
- Scaling requires manually starting more containers.
- Good for local dev and demos.


### ☸️ Kubernetes (clustered, scalable

```commandline
+---------------------------------------------------+
|               Kubernetes Cluster                  |
|                                                   |
|  +-------------------+   +-------------------+    |
|  |  Streamlit Pod    |   |  Model Pod(s)     |    |
|  |  (Deployment)     |   |  (Deployment)     |    |
|  +---------+---------+   +---------+---------+    |
|            |                       |              |
|            v                       v              |
|  +-------------------+                          |
|  |  Data Pod(s)      |                          |
|  |  (Deployment)     |                          |
|  +-------------------+                          |
|                                                   |
|  Services:                                        |
|   - streamlit-frontend (ClusterIP / Ingress)      |
|   - model-service (ClusterIP, scalable replicas)  |
|   - data-service (ClusterIP)                      |
|                                                   |
|  Features:                                        |
|   - DNS service discovery                         |
|   - Load balancing across pods                    |
|   - Auto-scaling (HPA)                            |
|   - Self-healing (pods rescheduled on failure)    |
+---------------------------------------------------+
```
- Each component runs as a **Deployment** with its own pods.
- **Services** provide stable DNS names (model-service.svc.cluster.local).
- **Ingress** exposes Streamlit externally.
- Kubernetes handles scaling, resilience, and monitoring.


---
## 🚀 Running with Docker Compose

### Prerequisites
- Docker & Docker Compose installed
- (Optional) GNU Make for convenience

### Build and run

```bash
docker compose -f deploy/docker-compose.yml build
docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d
```

### Services
- Frontend (Streamlit) → http://localhost:8501
- Model Service → http://localhost:8000
- Data Service → http://localhost:8001

### Stopping

```bash
docker compose -f deploy/docker-compose.yml down
```

### Using GNU Make

The following commands can be used as defined in the Makefile
```bash
make -C deploy build  # builds containers
make -C deploy up     # runs containers
make -C deploy down   # shuts down containers
make -C deploy logs   # brings up logs
make -C deploy clean  # shuts down containers, volumes and removes images
```
---
## ☸️ Running with Kubernetes

### Prerequisites
- Kubernetes cluster (local via Minikube, or cloud provider)
- kubectl configured
- Images pushed to a registry accessible by the cluster

### Apply Manifests

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/data-service.yaml
kubectl apply -f deploy/k8s/model-service.yaml
kubectl apply -f deploy/k8s/streamlit-frontend.yaml
kubectl apply -f deploy/k8s/ingress.yaml
```

### Access
- The frontend is exposed via an Ingress at http://nn.local (update host in ingress.yaml).
- For local dev, add to /etc/hosts:
```bash
127.0.0.1 nn.local
```

### Scaling

```bash
kubectl scale deployment model-service -n nn-curve-fit --replicas=3
```