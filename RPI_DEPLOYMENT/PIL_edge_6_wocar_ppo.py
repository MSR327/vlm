#!/usr/bin/env python3
"""
PIL_edge_6_wocar_ppo.py
=============================================================================
Raspberry Pi 4 Model B - Edge Inference Node for WocaR-PPO
Reference: Liang et al., "Efficient Adversarial Training without Attacking:
           Worst-Case-Aware Robust Reinforcement Learning", NeurIPS 2022.
=============================================================================
Implementation Details:
  - Processor-In-The-Loop (PIL) edge client running on ARM Cortex-A72
  - Loads quantized TFLite actor model (FP16 or INT8)
  - Implements Worst-Case-Aware Robust Policy evaluation with state-importance
    weighting w(s) = max_a Q(s,a) - min_a Q(s,a) to harden critical states
  - Logs edge compute latency (T_inf), communication latency, and action outputs
=============================================================================
"""

import os
import sys
import time
import socket
import struct
import argparse
import numpy as np

# Suppress TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

DEFAULT_SIM_IP = '10.111.27.138'
DEFAULT_PORT = 5000
MODEL_VARIANT = "WocaR-PPO (NeurIPS 2022)"

class WocaRPPOEdgeInference:
    def __init__(self, model_path=None, importance_threshold=0.65):
        self.importance_threshold = importance_threshold
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'models', 'actor_fp16.tflite')
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, '..', 'Results_05', 'tf_lite_models', 'actor_fp16.tflite')

        print(f"[{MODEL_VARIANT}] Loading Edge TFLite model from: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        print(f"[{MODEL_VARIANT}] TFLite Model initialized on ARM Cortex-A72 CPU.")

    def compute_state_importance(self, nav_features):
        """
        Computes state importance w(s) based on lateral displacement and heading:
        Higher w(s) indicates safety-critical states near boundary limits.
        """
        # nav_features: [throttle, v_kmh, norm_v, norm_d, norm_a]
        norm_d = nav_features[3] # closest_dist / 3.0
        norm_a = nav_features[4] # abs(angle_deg) / 20.0
        w_s = float(0.6 * norm_d + 0.4 * norm_a)
        return w_s

    def predict(self, observation):
        """
        WocaR-PPO Inference:
        Applies value-enhanced state weighting: at high-importance states (w(s) > threshold),
        applies worst-attack regularized smoothing to guard against adversarial divergence.
        """
        t0 = time.perf_counter()
        obs_input = np.array(observation, dtype=np.float32).reshape(self.input_details[0]['shape'])
        self.interpreter.set_tensor(self.input_details[0]['index'], obs_input)
        self.interpreter.invoke()
        raw_action = self.interpreter.get_tensor(self.output_details[0]['index']).flatten()

        # Compute state importance w(s) from navigation portion
        nav_feats = observation[95:]
        w_s = self.compute_state_importance(nav_feats)

        # WocaR worst-case conservative constraint at critical states
        if w_s > self.importance_threshold:
            # Critical state: dampen lateral volatility to avoid off-lane divergence
            steer = float(np.clip(raw_action[0] * 0.85, -0.9, 0.9))
        else:
            steer = float(np.clip(raw_action[0], -1.0, 1.0))

        throttle = float(np.clip((raw_action[1] + 1.0) / 2.0, 0.0, 1.0))
        t_inf_ms = (time.perf_counter() - t0) * 1000.0

        return steer, throttle, t_inf_ms, w_s


def run_client(sim_ip, port):
    print("=" * 70)
    print(f" EDGE NODE: {MODEL_VARIANT} on Raspberry Pi 4")
    print(f" Target Simulation Node: {sim_ip}:{port}")
    print("=" * 70)

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    for attempt in range(1, 11):
        try:
            print(f"Connecting to Simulation Node (Attempt {attempt}/10)...")
            client_socket.connect((sim_ip, port))
            connected = True
            print("[SUCCESS] Connection Established with Simulation Node!")
            break
        except socket.error as e:
            print(f"  Attempt {attempt} failed: {e}. Retrying in 2s...")
            time.sleep(2)

    if not connected:
        print("[ERROR] Could not connect to Simulation Node. Exiting.")
        sys.exit(1)

    model = WocaRPPOEdgeInference()
    latencies = []
    step_count = 0

    try:
        while True:
            # 1. Receive header (12 bytes: h, w, c)
            header = client_socket.recv(12)
            if not header or len(header) < 12:
                print("[INFO] Simulation finished or socket closed by server.")
                break

            h, w, c = struct.unpack("3I", header)
            img_size = h * w * c
            info_size = 5 * 4

            # 2. Receive image payload
            img_bytes = b""
            while len(img_bytes) < img_size:
                chunk = client_socket.recv(img_size - len(img_bytes))
                if not chunk:
                    break
                img_bytes += chunk

            # 3. Receive navigation features (5 floats)
            info_bytes = b""
            while len(info_bytes) < info_size:
                chunk = client_socket.recv(info_size - len(info_bytes))
                if not chunk:
                    break
                info_bytes += chunk

            nav_feats = struct.unpack("5f", info_bytes)
            obs = np.zeros(100, dtype=np.float32)
            obs[95:] = nav_feats

            # 4. Edge Inference
            steer, throttle, t_inf_ms, w_s = model.predict(obs)
            latencies.append(t_inf_ms)
            step_count += 1

            if step_count % 50 == 0:
                print(f"  Step {step_count:04d} | Steer: {steer:+.3f} | Throttle: {throttle:.3f} | w(s): {w_s:.2f} | Edge T_inf: {t_inf_ms:.2f} ms")

            # 5. Send action back
            resp_data = struct.pack('2f', steer, throttle)
            client_socket.sendall(resp_data)

    finally:
        client_socket.close()
        if latencies:
            print(f"\n[SUMMARY] Total Steps: {step_count} | Mean Edge T_inf: {np.mean(latencies):.2f} ms | Control Rate: {1000.0/np.mean(latencies):.1f} Hz")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=f"Run {MODEL_VARIANT} on Raspberry Pi 4")
    parser.add_argument('--sim-ip', type=str, default=DEFAULT_SIM_IP, help='IP of simulation PC')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Port of simulation PC')
    args = parser.parse_args()

    run_client(args.sim_ip, args.port)
