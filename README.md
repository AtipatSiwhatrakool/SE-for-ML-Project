# SE-for-ML-Project

### Prerequisites
Before running the application, please review your hardware configuration. If you **do not** have an NVIDIA GPU, you must open the `airflow_pipeline/docker-compose.yml` file and remove or comment out the following block to avoid runtime errors:

```yaml
      deploy:
        resources:
          reservations:
            devices:
              - driver: nvidia
                count: 1
                capabilities: [gpu]
```

### Running the Application

1. Navigate to the pipeline directory:
   ```bash
   cd airflow_pipeline
   ```

2. Start the Docker containers in the background:
   ```bash
   docker-compose up -d
   ```

3. **Logging into Airflow**:
   The Airflow UI will be available at `http://localhost:8080` (default username is usually `admin`). To find your auto-generated password, run the following command in your terminal:
   ```bash
   docker logs airflow_pipeline-airflow-webserver-1 2>&1 | grep -i password
   ```

## Model Inference & Usage

**Note:** You do not need to run the Airflow DAG to train a model! The models have already been trained. You can directly load and choose the best model from the `mlruns` directory located inside `airflow_pipeline/`.