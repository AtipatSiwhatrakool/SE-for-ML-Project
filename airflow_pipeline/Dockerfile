FROM apache/airflow:3.0.0-python3.10

USER root
# Install system dependencies if required (e.g. for building some packages or git)
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
         build-essential \
         git \
  && apt-get autoremove -yqq --purge \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements and install
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /requirements.txt
