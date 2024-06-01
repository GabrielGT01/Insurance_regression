import uvicorn
from fastapi import FastAPI, HTTPException
from model import ChargeInput, ChargeModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
model = ChargeModel()

@app.post('/predict')
def predict_price(charge: ChargeInput):
    data = charge.dict()
    try:
        logger.info(f'Received data for prediction: {data}')
        prediction = model.predict_price(
            data['age'], data['sex'], data['bmi'], data['children'], data['smoker'], data['region']
        )
        logger.info(f'Prediction result: {prediction}')
        return {
            'prediction': prediction
        }
    except Exception as e:
        logger.error(f'Error in /predict endpoint: {e}')
        raise HTTPException(status_code=500, detail="Prediction error")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)




    
