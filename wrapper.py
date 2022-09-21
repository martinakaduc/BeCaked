import mlflow
from becaked import BeCakedModel
from utils import *
from data_utils import DataLoader
from datetime import datetime

class BecakedWrapper(mlflow.pyfunc.PythonModel):
     
    def load_context(self,context):
        self.model=mlflow.pyfunc.load_model(context.artifacts["Original_model"])

    def predict(self, context, model_input):
         start_date = datetime.fromisoformat(model_input.at[0,"start_date"]).timetuple().tm_yday
         end_date = datetime.fromisoformat(model_input.at[0,"end_date"]).timetuple().tm_yday
         results= self.get_predict_result(start_date, end_date)
         return results

    def get_predict_result(self,start_date,end_date):
        becaked_model = BeCakedModel(day_lag=10)
        data_loader = DataLoader()
        data = data_loader.get_data_world_series()
        results= get_prediction_result(becaked_model, data, start_date , end_date, step=31, day_lag=10)
        return results
    