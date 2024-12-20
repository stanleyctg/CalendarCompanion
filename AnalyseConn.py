import pyshark
from collections import defaultdict
import socket
# from sklearn.preprocessing import StandardScaler
import numpy as np

pyshark.config.tshark_path = '/usr/bin/tshark' 
flows = {}

class FlowStats:
    def __init__(self, origin_5tuple):
        self.origin = origin_5tuple
        self.orig_pkts = 0
        self.orig_ip_bytes = 0
        self.resp_pkts = 0
        self.resp_ip_bytes = 0
        self.proto_icmp = 0
        self.proto_tcp = 0
        self.proto_udp = 0

    def update(self, packet):
        src_ip = packet.ip.src
        src_port = packet[packet.transport_layer].srcport if packet.transport_layer else None
        ip_len = int(packet.ip.len) if hasattr(packet.ip, 'len') else 0

        if (src_ip, src_port) == self.origin:
            self.orig_pkts += 1
            self.orig_ip_bytes += ip_len
        else:
            self.resp_pkts += 1
            self.resp_ip_bytes += ip_len

    def to_feature_vector(self):
        return np.array([
            self.orig_pkts,
            self.orig_ip_bytes,
            self.resp_pkts,
            self.resp_ip_bytes,
            self.proto_icmp,
            self.proto_tcp,
            self.proto_udp
        ]).reshape(1, -1)

def get_flow_key(packet):
    src_ip = packet.ip.src
    dst_ip = packet.ip.dst
    if packet.transport_layer in ['TCP', 'UDP']:
        src_port = packet[packet.transport_layer].srcport
        dst_port = packet[packet.transport_layer].dstport
        protocol = packet.transport_layer.lower()
    else:
        src_port = '0'
        dst_port = '0'
        protocol = 'icmp'
    return (src_ip, src_port, dst_ip, dst_port, protocol)

def set_protocol_flags(flow_stats, protocol):
    if protocol == 'tcp':
        flow_stats.proto_tcp = 1
    elif protocol == 'udp':
        flow_stats.proto_udp = 1
    elif protocol == 'icmp':
        flow_stats.proto_icmp = 1

capture = pyshark.LiveCapture(interface='lo', bpf_filter='port 5000')

for packet in capture.sniff_continuously():
    if 'IP' not in packet:
        continue

    flow_key = get_flow_key(packet)
    if flow_key not in flows:
        src_ip, src_port, dst_ip, dst_port, protocol = flow_key
        origin_5tuple = (src_ip, src_port)
        flow_stats = FlowStats(origin_5tuple)
        set_protocol_flags(flow_stats, protocol)
        flows[flow_key] = flow_stats

    flows[flow_key].update(packet)

    # Prepare the feature vector
    feature_vector = flows[flow_key].to_feature_vector()
    print(feature_vector)
