import joblib
from pydantic import BaseModel
import logging
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChargeInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

class ChargeModel:
    def __init__(self):
        try:
            self.model_fname_ = 'model_pipeline.pkl'
            self.model = joblib.load(self.model_fname_)
            logger.info(f'Model loaded successfully from {self.model_fname_}')
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            raise

    def predict_price(self, age, sex, bmi, children, smoker, region):
        try:
            test_input = np.array([[age, sex, bmi, children, smoker, region]])
            # Convert the test input to a DataFrame with the appropriate column names
            test_input_df = pd.DataFrame(test_input, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
            logger.info(f'Test input DataFrame: {test_input_df}')
            prediction = self.model.predict(test_input_df)
            logger.info(f'Prediction made successfully for input: {test_input_df}')
            return prediction[0]
        except Exception as e:
            logger.error(f'Error making prediction: {e}')
            raise
