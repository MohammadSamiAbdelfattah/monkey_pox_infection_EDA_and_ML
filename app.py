import streamlit as st
import pickle
import numpy as np
import json

# Title and Info
st.markdown("<h2 style='text-align: center; color: black;'>Prediction of Monkey Pox Infection</h2>", unsafe_allow_html=True)

st.info("If you want to know more about Monkey Pox Infection, Check this [Mpox (monkeypox) outbreak](https://www.who.int/emergencies/situations/monkeypox-oubreak-2022#:~:text=WHO%20Emergency%20Appeal%3A%20Monkeypox%20%2D%20July,number%20of%20cases%20and...)")

st.write(" ")

# Set labels with values 
CHOICES = {"choose...": "choose...", 1: "True", 0: "False"}

def format_func(option):
    return CHOICES[option]

# Choose options for prediction
systemic_illness = st.selectbox('Is there any **Systemic Illness**?', ["choose...", 'Fever', 'Swollen Lymph Nodes', 'Muscle Aches and Pain', 'None'],)
rectal_pain = st.selectbox('Is there a **Rectal Pain**?', options=list(CHOICES.keys()), format_func=format_func)
sore_throat = st.selectbox('Is there a **Sore Throat**?', options=list(CHOICES.keys()), format_func=format_func)
penile_edema = st.selectbox('Is there a **Penile edema**?', options=list(CHOICES.keys()), format_func=format_func)
oral_lesions = st.selectbox('Are there **Oral Lesions**?', options=list(CHOICES.keys()), format_func=format_func)
solitary_lesion = st.selectbox('Is there a **Solitary Lesion**?', options=list(CHOICES.keys()), format_func=format_func)
swollen_tonsils = st.selectbox('Is there a **Swollen Tonsils**?', options=list(CHOICES.keys()), format_func=format_func)
hiv = st.selectbox('Is there a **HIV Infection**?', options=list(CHOICES.keys()), format_func=format_func)
std = st.selectbox('Is there a **Sexually Transmitted Infection**?', options=list(CHOICES.keys()), format_func=format_func)

st.markdown("""---""")

# load the model
def load_model():
    with open('trained_model.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

rfc = data["model"]
le_systemic_illness = data["le_systemic_illness"]

# Button for prediction
col1, col2, col3 = st.columns(3)

with col1:
    st.write(" ")

with col2:
    predict = st.button('Get the prediction')

with col3:
    st.write(" ")


# Prediction function
if predict:
    if (systemic_illness == "choose..." or rectal_pain == "choose..." or sore_throat == "choose..." or penile_edema == "choose..." or oral_lesions == "choose..." or solitary_lesion == "choose..." or swollen_tonsils == "choose..." or hiv == "choose..." or std == "choose..."):
        st.markdown("<h3 style='text-align: center; color: red;'>Please, Select valid options!!!!</h3>", unsafe_allow_html=True)
    else:    
        x = np.array([[systemic_illness, rectal_pain, sore_throat, penile_edema, oral_lesions, solitary_lesion, swollen_tonsils, hiv, std]])
        x[:, 0] = le_systemic_illness.transform(x[:, 0])

        infection = rfc.predict(x)

        if infection == [0]:
            st.markdown("<h1 style='text-align: center; color: green;'>You don't have the infection</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: red;'>You have the infection</h1>", unsafe_allow_html=True)
