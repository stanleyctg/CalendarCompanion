import pyshark
import pandas as pd
import socket
import struct
import pickle
import argparse
import traceback

def ip_to_int(ip):
    try:
        return struct.unpack("!I", socket.inet_aton(ip))[0]
    except socket.error:
        return int(socket.inet_pton(socket.AF_INET6, ip).hex(), 16)

def load_scalar(scaler_path='scaler.pkl'):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f) 
    return scaler

def load_model(model_path='MalwareDetectionModel.pkl'):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def continuous_capture(interface, port):
    capture = pyshark.LiveCapture(interface=interface, display_filter=f'tcp.port == {port} || udp.port == {port}')
    model = load_model('MalwareDetectionModel.pkl')
    scaler = load_scalar('scaler.pkl')
    
    columns = ['source_ip', 'destination_ip', 'source_port', 'destination_port', 
               'packet_length', 'protocol_TCP', 'protocol_UDP']
    
    for packet in capture.sniff_continuously():
        if 'IP' in packet or 'IPv6' in packet:
            try:
                protocol_layer = packet.highest_layer
                
                # Check if protocol layer has ports
                if hasattr(packet[protocol_layer], 'srcport') and hasattr(packet[protocol_layer], 'dstport'):
                    source_ip = packet.ip.src if 'IP' in packet else packet.ipv6.src
                    destination_ip = packet.ip.dst if 'IP' in packet else packet.ipv6.dst
                    source_port = int(packet[protocol_layer].srcport)
                    destination_port = int(packet[protocol_layer].dstport)
                    packet_length = int(packet.length)
                    protocol_tcp = 1.0 if 'TCP' in packet.highest_layer else 0.0
                    protocol_udp = 1.0 if 'UDP' in packet.highest_layer else 0.0

                    # Prepare data for scaling and prediction
                    data = pd.DataFrame([{
                        'source_ip': ip_to_int(source_ip),
                        'destination_ip': ip_to_int(destination_ip),
                        'source_port': source_port,
                        'destination_port': destination_port,
                        'packet_length': packet_length,
                        'protocol_TCP': protocol_tcp,
                        'protocol_UDP': protocol_udp
                    }], columns=columns)

                    new_data_scaled = scaler.transform(data.values)
                    prediction = model.predict(new_data_scaled)
                    prediction = 'Benign' if prediction == 1 else 'Malicious'
                    print(f"Prediction for packet: {prediction}")
                else:
                    print(f"Skipping packet without port information: {packet}")

            except Exception as e:
                print(f"Error processing packet: {e}")
                traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Malware Detection via Network Capture")
    parser.add_argument("--interface", type=str, default=r'\Device\NPF_Loopback', help="Network interface to monitor")
    parser.add_argument("--port", type=str, default="5000", help="Port to filter traffic")
    args = parser.parse_args()
    continuous_capture(interface=args.interface, port=args.port)
