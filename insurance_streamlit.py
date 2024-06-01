import streamlit as st
import numpy as np
import pandas as pd
import logging
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.markdown("### This application helps calculate the insurance cost")

st.markdown("**Information required**")
st.markdown("* age: age of primary beneficiary")
st.markdown("* sex: insurance contractor gender, female, male")
st.markdown("* bmi: Body mass index, providing an understanding of body, ideally 18.5 to 24.9")
st.markdown("* children: Number of children covered by health insurance / Number of dependents")
st.markdown("* smoker: Smoking YES/NO")
st.markdown("* region: the beneficiary's residential area in the US, northeast, southeast, southwest, northwest")

age = st.slider("Choose your age:", min_value=10, max_value=80, value=25, step=1)
sex = st.selectbox("Choose your Gender:", ['male', 'female'])
bmi = st.number_input('Enter your BMI:', min_value=10.5, max_value=50.5)
children = st.slider("Number of kids you have:", min_value=0, max_value=5, value=0, step=1)
smoker = st.selectbox("Do you smoke:", ['yes', 'no'])
region = st.selectbox('What part of the country do you live in:', ['southwest', 'southeast', 'northwest', 'northeast'])

@st.cache_data
def load_model():
    try:
        model_fname = 'model_pipeline.pkl'
        model = joblib.load(model_fname)
        logger.info(f'Model loaded successfully from {model_fname}')
        return model
    except Exception as e:
        logger.error(f'Error loading model: {e}')
        raise

def predict_price(model, age, sex, bmi, children, smoker, region):
    try:
        test_input = np.array([[age, sex, bmi, children, smoker, region]])
        # Convert the test input to a DataFrame with the appropriate column names
        test_input_df = pd.DataFrame(test_input, columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        logger.info(f'Test input DataFrame: {test_input_df}')
        prediction = model.predict(test_input_df)
        logger.info(f'Prediction made successfully for input: {test_input_df}')
        return prediction[0]
    except Exception as e:
        logger.error(f'Error making prediction: {e}')
        raise

# Load the model outside the main block to cache it
model = load_model()

if __name__ == '__main__':
    if st.button("Calculate Insurance Cost"):
        try:
            amount = predict_price(model, age, sex, bmi, children, smoker, region)
            st.write('The predicted insurance cost is:', amount)
        except Exception as e:
            st.write('Error in prediction:', str(e))
