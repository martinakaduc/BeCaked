from becaked import BeCakedModel
import wrapper
from becaked import BeCakedModel
from utils import *
from data_utils import DataLoader

import mlflow.pyfunc
from mlflow.models.signature import infer_signature
import pandas as pd


class PredictionModel():
       # set Tracking Uri
       mlflow.set_tracking_uri("sqlite:///mlruns.db")

       with mlflow.start_run() as run:

            # save the model in MLflow
            mlflow.sklearn.save_model( wrapper.BecakedWrapper(), "becaked-model" )    
            
            # add model signature (defines the schema of a model’s inputs and outputs)
            becaked_inputs = pd.DataFrame([["2022/09/01","2022/10/01"]], columns=["start_date","end_date"])
           
            results= wrapper.BecakedWrapper().get_predict_result(161 , 192)
            signature = infer_signature(becaked_inputs,results)
            
            # Construct and return a pyfunc-compatible model wrapper
            artifacts = {
            "Original_model": "becaked-model"
            }
            model_data = mlflow.pyfunc.log_model(
                artifact_path="ML-model-1",
                python_model=wrapper.BecakedWrapper(),
                artifacts=artifacts,
                code_path = ["wrapper.py"],
                signature=signature

                )

            # register the model in Mlflow
            registered_model_name="becaked-model"
            mv = mlflow.register_model(model_data.model_uri, registered_model_name)   

       