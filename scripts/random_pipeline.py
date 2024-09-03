import time
from prediction import preprocess_data, get_prediction
import joblib
import torch
from random_data_generation import generate_random_input  # Import the random data generation function
from model import RNNClassifier, CNNLSTMClassifier

# # Load the label encoder, scalers and models
# label_encoder = joblib.load(r'C:\Users\Admin\Downloads\Anusha\AIML\8th sem\18AIP83 Project Work Phase  2\real time pipeline\models\label_encoder.pkl')

# standard_scaler = joblib.load(r"C:\Users\Admin\Downloads\Anusha\AIML\Final Project\real time pipeline\models\standard_scaler.pkl")
# min_max_scaler = joblib.load(r"C:\Users\Admin\Downloads\Anusha\AIML\Final Project\real time pipeline\models\min_max_scaler.pkl")


# rnn_model = torch.load(r"C:\Users\Admin\Downloads\Anusha\AIML\Final Project\real time pipeline\models\rnn_model_14.pkl", map_location=torch.device('cpu'))
# cnn_lstm_model = torch.load(r"C:\Users\Admin\Downloads\Anusha\AIML\Final Project\real time pipeline\models\cnn_lstm_model_14.pkl", map_location=torch.device('cpu'))

# rnn_model.eval()
# cnn_lstm_model.eval()

def get_character_prediction(data, standard_scaler, min_max_scaler, rnn_model, cnn_lstm_model, label_encoder):
    processed_data = preprocess_data(data, standard_scaler, min_max_scaler)
    input_tensor = torch.tensor(processed_data, dtype=torch.float32)
    predicted = get_prediction(input_tensor, rnn_model, cnn_lstm_model)
    return label_encoder.inverse_transform(predicted)[0]

# def data_collection_thread():
#     while True:
#         data = generate_random_input()  
#         processed_data = preprocess_data(data, standard_scaler, min_max_scaler)
#         input_tensor = torch.tensor(processed_data, dtype=torch.float32)
#         predicted = get_prediction(input_tensor, rnn_model, cnn_lstm_model)
#         print("Predicted Character:", label_encoder.inverse_transform(predicted))

# if __name__ == "__main__":
#     print('starting')
#     while True:
#         try:
#             data_collection_thread()
#             print('continuing')
#             time.sleep(1)
#         except KeyboardInterrupt:
#             print("Interrupted, exiting...")
#             break
