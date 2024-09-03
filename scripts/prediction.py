import torch

def preprocess_data(data, standard_scaler, min_max_scaler):
    normalized_data = standard_scaler.transform(data)
    scaled_data = min_max_scaler.transform(normalized_data)

    return scaled_data

def get_prediction(input_data, rnn_model, cnn_lstm_model):

    with torch.no_grad():
        rnn_output = rnn_model(input_data)
        probs_rnn = torch.nn.functional.softmax(rnn_output, dim=1)
        cnn_output = cnn_lstm_model(input_data)
        probs_cnn_lstm = torch.nn.functional.softmax(cnn_output, dim=1)
        combined_probs_batch = (probs_cnn_lstm + probs_rnn) / 2
        _, predicted = torch.max(combined_probs_batch, 1)

    return predicted