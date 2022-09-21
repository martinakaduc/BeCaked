## How to use

### Install required packages

```
pip install -r requirements.txt
```

### save model and register it in MLFflow

```
python prediction.py 

```

### (optional) Run MLflow’s Tracking UI to view  the registered model  

```
mlflow ui --backend-store-uri sqlite:///mlruns.db

```

### Serve the registered model locally

You should run the following commands.
```
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db 

mlflow models serve -m models:/becaked-model/latest -p 2000 --no-conda

```

### TEST the REST API

```
PATH : http://127.0.0.1:2000/invocations

HEADERS: Content-Type: application/json

BODY:[{"start_date":161,"end_date":192}]



```