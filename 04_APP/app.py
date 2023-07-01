import streamlit as st
import numpy as np
import pickle
from PIL import Image

# page configs
st.set_page_config(
    page_title="Ice Cream Revenue Prediction App",
    page_icon="🍧",
    layout="wide",
    menu_items=None,
    initial_sidebar_state="collapsed"
)

# applying custom css styles
with open("04_APP/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# reading the pickle file
with open("03_MODELS/icecream_linear_model.pkl", 'rb') as f:
    model = pickle.load(f)

# working on main layout
st.title("Ice Cream Revenue Prediction")

# creating 2 columns sections
col1, col2 = st.columns([0.45, 0.65], gap="medium")

# working on col1 section:
image = Image.open("05_RESOURCES/ice-cream.jpg")
with col1:
    st.image(image, use_column_width=True)
    st.write("""The application utilizes machine learning techniques to assist in formulating effective business 
    strategies. Its underlying model has undergone training using historical data that includes information on 
    revenue and temperature.""")

    st.write("""
    By providing a straightforward temperature input in degrees Celsius, this application is capable of predicting 
    the potential revenue that can be generated on a given day.""")

# working on col2 section
with col2:
    with st.form('user_inputs'):
        temp = st.number_input(label="Temperature (Degree Celcius):", min_value=0.0, max_value=60.0, value=22.0)
        st.form_submit_button()

    # input feature preparation
    input_temp = np.array(temp).reshape(-1, 1)

    # predicting revenue
    predicted_revenue = model.predict(input_temp)[0]

    st.write(
        f"At a temperature of ${input_temp[0][0]:0.2f}^\circ C$, the business has the potential to generate "
        f"approximately **:green[${predicted_revenue:0.2f}]** in revenue")
