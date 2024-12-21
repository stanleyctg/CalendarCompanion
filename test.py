import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
with open('MalwareDetectionModel.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)  # Ensure you save the scaler during training and load it here

# Example new data
new_data = pd.DataFrame({
    'source_ip': [3232235931],
    'destination_ip': [167772222],
    'source_port': [44814],
    'destination_port': [31706],
    'packet_length': [1165],
    'protocol_TCP': [0.0],
    'protocol_UDP': [1.0]
})

# Transform the new data using the loaded scaler
# new_data_scaled = scaler.transform(new_data)

# Making predictions
predictions = loaded_model.predict(new_data)
print("Predictions:", predictions)
