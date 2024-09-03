import streamlit as st
import joblib
import torch
import time
from random_data_generation import generate_random_input  # Import the random data generation function
from random_pipeline import get_character_prediction
from model import RNNClassifier, CNNLSTMClassifier

print(torch.__version__)
# Force CPU usage
torch.cuda.is_available = lambda : False
device = torch.device("cpu")
print(f"Using device: {device}")
# Load the label encoder, scalers, and models
label_encoder = joblib.load(r"C:\Users\user\Downloads\final_project_neurosync-anusha (1)\final_project_neurosync-anusha\models\label_encoder (1).pkl")
standard_scaler = joblib.load(r"C:\Users\user\Downloads\final_project_neurosync-anusha (1)\final_project_neurosync-anusha\models\standard_scaler (1).pkl")
min_max_scaler = joblib.load(r"C:\Users\user\Downloads\final_project_neurosync-anusha (1)\final_project_neurosync-anusha\models\min_max_scaler (1).pkl")
rnn_model = torch.load(r"C:\Users\user\Downloads\final_project_neurosync-anusha (1)\final_project_neurosync-anusha\models\rnn_model_31_ynhfblre.pkl", map_location=device)
cnn_lstm_model = torch.load(r"C:\Users\user\Downloads\final_project_neurosync-anusha (1)\final_project_neurosync-anusha\models\cnn_lstm_model_31_ynhfblre.pkl", map_location=device)

char_to_word = {
    'y' : 'YES',
    'n' : 'NO',
    'f' : 'FORWARD',
    'b' : 'BACK',
    'l' : 'LEFT',
    'r' : 'RIGHT',
    'e' : 'EMERGENCY',
    'h' : 'HUNGRY'
}

# Streamlit app
st.title("Real-Time Character Prediction")
st.write("This app uses a machine learning model to predict characters using EEG Signals.")

predicted_text_placeholder = st.empty()
    
if 'predicted_characters' not in st.session_state:
    st.session_state.predicted_characters = []

# Start and stop buttons
col1, col2, col3, col4, col5 = st.columns(5)
start_button = col1.button("Start Prediction")
stop_button = col2.button("Stop Prediction")
space_button = col3.button("Add Space")
backspace_button = col4.button("⌫ Backspace")
clear_button = col5.button("Clear")

if 'running' not in st.session_state:
    st.session_state.running = False

if backspace_button:
    if len(st.session_state.predicted_characters) > 0:
        st.session_state.predicted_characters.pop()
        st.success("Last character removed!")
    
if clear_button:
    st.session_state.predicted_characters = []
    st.success("Prediction cleared!")

predicted_text = ''.join(st.session_state.predicted_characters)
st.markdown(f"```\n{predicted_text}\n```", unsafe_allow_html=True)

def update_prediction():
    if st.session_state.running:
        data = generate_random_input()
        predicted_char = get_character_prediction(data, standard_scaler, min_max_scaler, rnn_model, cnn_lstm_model, label_encoder)
        st.session_state.predicted_characters.append(char_to_word[predicted_char])
        st.session_state.predicted_characters.append(" " * 2)  # Add two characters of whitespace
        predicted_text = ''.join(st.session_state.predicted_characters)
        predicted_text_placeholder.markdown(f"```\n{predicted_text}\n```", unsafe_allow_html=True)

if start_button:
    if not st.session_state.running:
        st.session_state.running = True
        st.success("Prediction started!")
        # Call update_prediction once to start
        update_prediction()
    
if stop_button:
    st.session_state.running = False
    st.success("Prediction stopped!")

if space_button:
    st.session_state.predicted_characters.append(" " * 2)  # Add two characters of whitespace

# Call update_prediction only when running is True
while st.session_state.running:
    update_prediction()
    time.sleep(3.0)
