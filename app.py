import joblib
import streamlit as st
from utils import predict_species

# Load the joblib model
model = joblib.load("notebook/iris_model.joblib")

# Streamlit app
st.set_page_config(page_title="Iris Project")

# Write the title
st.title("Iris End to End Project")
st.subheader("by Utkarsh Gaikwad")

# Input from user
sep_len = st.number_input("Sepal Length : ", min_value=0.00, step=0.01)
sep_wid = st.number_input("Sepal Width : ", min_value=0.00, step=0.01)
pet_len = st.number_input("Petal Length : ", min_value=0.00, step=0.01)
pet_wid = st.number_input("Petal Width : ", min_value=0.00, step=0.01)

# Create a predict button
button = st.button("Predict", type="primary")

# if button pressed
if button:
    st.subheader("Results : ")
    pred, prob = predict_species(model, sep_len, sep_wid, pet_len, pet_wid)
    st.subheader(f"Predicted Species : {pred}")
    st.subheader(f"Probabilities")
    st.dataframe(prob)
