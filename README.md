# mnist-mlops
By ***Mohammed Musthafa Rafi***
Here i will be creating an end-to-end machine learning pipeline using the MNIST dataset.

This project provides a complete end-to-end machine learning pipeline using the MNIST dataset. It demonstrates key MLops principles by incorporating model training with PyTorch, monitoring using Weights & Biases (W&B), deployment with FastAPI, and containerization with Docker. Additionally, Prometheus and Grafana are used to monitor API metrics, making the solution production-ready and easily scalable.

### Key Features:
- **Model Training**: A Logistic Regression model trained on the MNIST dataset using PyTorch.
- **Logging**: Training metrics logged with [Weights & Biases](https://wandb.ai/).
- **Model Deployment**: FastAPI to serve model predictions.
- **Containerization**: Docker to package the FastAPI application.
- **Monitoring**: Prometheus to collect metrics and Grafana for visualization.

## Running Locally

### Prerequisites
- Python 3.9
- Docker
- Weights & Biases account (for logging)

### Step-by-Step Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/itsMustafamr/mnist-mlops.git
   cd mnist-mlops
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**:
   - Make sure you are logged in to Weights & Biases.
   - Run the training script:
   ```bash
   python src/train.py
   ```

5. **Run the FastAPI Server**:
   ```bash
   uvicorn src.deploy:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Access the Prediction Endpoint**:
   - Visit `http://localhost:8000/docs` to access the FastAPI Swagger UI and test the `/predict` endpoint.

## Docker Container Instructions

### Build Docker Image
```bash
docker build -t mnist-api .
```

### Run Docker Container
- Run the container:
  ```bash
  docker run -p 8001:8000 mnist-api
  ```
- The API will be available at `http://localhost:8001`.

### Monitoring Using Prometheus and Grafana
1. **Prometheus**:
   ```bash
   docker run --network fastapi-prometheus-network -p 9091:9090 -v /home/exouser/mnist-prometheus.yml:/etc/prometheus/prometheus.yml prom/prometheus
   ```
   - Access Prometheus at: [http://localhost:9091](http://localhost:9091)

2. **Grafana**:
   ```bash
   docker run -d --network fastapi-prometheus-network -p 3000:3000 grafana/grafana
   ```
   - Access Grafana at: [http://localhost:3000](http://localhost:3000).
   - Add Prometheus as a data source and create dashboards to visualize metrics.

## Weights & Biases Report
The complete report on model training, metrics, and analysis can be found here:
[W&B Report Link](https://wandb.ai/itsmustafamr/mnist-mlops/runs/b17lkbbw)

## References & Further Reading
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/)
- [Grafana Documentation](https://grafana.com/docs/)

# More references used are,
- **Deep Learning with PyTorch** by Eli Stevens, Luca Antiga, and Thomas Viehmann
- **Designing Machine Learning Systems** by Chip Huyen
