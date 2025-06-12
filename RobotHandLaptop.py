import os

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress absl logs (used by MediaPipe)
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import cv2
import socket
import json
import threading
import math
import tkinter as tk
from tkinter import messagebox
import mediapipe as mp
import math

ALPHA_VALUE = 0.85

class SpreadCalibrator:
    def __init__(self):
        # MCP, TIP, and MCP neighbor for baseline
        self.fingers = {
            "index": (5, 8, 9),    # index_mcp, index_tip, middle_mcp
            "middle": (9, 12, 13), # middle_mcp, middle_tip, ring_mcp
            "ring": (13, 16, 17),  # ring_mcp, ring_tip, pinky_mcp
            "pinky": (17, 20, 13)  # pinky_mcp, pinky_tip, ring_mcp (use ring_mcp as neighbor for pinky)
        }
        self.baseline_spread = {}
        self.min_spread = {}
        self.max_spread = {}

    def vector_angle(self, v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        mag1 = math.sqrt(sum(a * a for a in v1))
        mag2 = math.sqrt(sum(a * a for a in v2))
        if mag1 == 0 or mag2 == 0:
            return 0
        # Clamp for safety
        cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
        return math.acos(cos_angle) * 180 / math.pi

    def get_palm_normal(self, landmarks):
        # Use wrist (0), index_mcp (5), pinky_mcp (17) to define palm plane
        p0 = landmarks[0]
        p1 = landmarks[5]
        p2 = landmarks[17]
        v1 = [p1[i] - p0[i] for i in range(3)]
        v2 = [p2[i] - p0[i] for i in range(3)]
        # Cross product to get normal
        normal = [
            v1[1]*v2[2] - v1[2]*v2[1],
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0]
        ]
        return normal

    def get_finger_spread(self, landmarks):
        spread = {}
        palm_normal = self.get_palm_normal(landmarks)
        for name, (mcp, tip, mcp_neighbor) in self.fingers.items():
            # Baseline: MCP to neighbor MCP
            baseline = [landmarks[mcp_neighbor][i] - landmarks[mcp][i] for i in range(3)]
            # Finger: MCP to TIP
            finger_vec = [landmarks[tip][i] - landmarks[mcp][i] for i in range(3)]
            angle = self.vector_angle(baseline, finger_vec)
            # Use sign: if finger is to the right of baseline, positive; left, negative
            cross = [
                baseline[1]*finger_vec[2] - baseline[2]*finger_vec[1],
                baseline[2]*finger_vec[0] - baseline[0]*finger_vec[2],
                baseline[0]*finger_vec[1] - baseline[1]*finger_vec[0]
            ]
            sign = 1 if sum(c*p for c, p in zip(cross, palm_normal)) > 0 else -1
            spread[name] = angle * sign
        return spread

    def average_frame_spread(self, num_frames=15):
        cap = cv2.VideoCapture(0)
        spreads_list = []
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
            while len(spreads_list) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_landmarks[0].landmark]
                    spread = self.get_finger_spread(landmarks)
                    if spread:
                        spreads_list.append(spread)
                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()
        if not spreads_list:
            return {}
        avg_spread = {}
        for finger in self.fingers:
            values = [s.get(finger, 0) for s in spreads_list if finger in s]
            if values:
                avg_spread[finger] = sum(values) / len(values)
        return avg_spread

    def calibrate(self):
        def prompt(msg):
            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo("Calibration Step", msg)
            root.destroy()
        prompt("Hold your hand in a neutral spread position.")
        self.baseline_spread = self.average_frame_spread()
        prompt("Close your fingers together.")
        self.min_spread = self.average_frame_spread()
        prompt("Open your fingers as wide as possible.")
        self.max_spread = self.average_frame_spread()
        if not (self.baseline_spread and self.min_spread and self.max_spread):
            print("Error: Calibration incomplete or failed. Please try again.")
            return
        data = {
            "baseline": self.baseline_spread,
            "min": self.min_spread,
            "max": self.max_spread
        }
        with open("spread_calibration.json", "w") as f:
            json.dump(data, f)
        print("✅ Calibration complete and saved.")

    def load_calibration(self):
        if not os.path.exists("spread_calibration.json"):
            print("Calibration file not found.")
            return False
        with open("spread_calibration.json", "r") as f:
            data = json.load(f)
        self.baseline_spread = data.get("baseline", {})
        self.min_spread = data.get("min", {})
        self.max_spread = data.get("max", {})
        return True

    def scale_spread(self, raw, baseline, min_val, max_val):
        try:
            if raw < baseline and baseline != min_val:
                scaled = -5 * (baseline - raw) / (baseline - min_val)
            elif raw > baseline and max_val != baseline:
                scaled = 5 * (raw - baseline) / (max_val - baseline)
            else:
                scaled = 0
        except ZeroDivisionError:
            scaled = 0
        return max(-5, min(5, scaled))

class HandTracker:
    def __init__(self, on_scaled_spread):
        self.on_scaled_spread = on_scaled_spread
        self.spread_cal = SpreadCalibrator()
        self.fingers = self.spread_cal.fingers

    def calibrate(self):
        self.spread_cal.calibrate()

    def run_tracking(self):
        if not self.spread_cal.load_calibration():
            print("Please calibrate spread first.")
            return
        cap = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        with mp_hands.Hands(max_num_hands=1) as hands:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)
                if result.multi_hand_landmarks:
                    landmarks = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_landmarks[0].landmark]
                    spread = self.spread_cal.get_finger_spread(landmarks)
                    if spread:
                        scaled = {}
                        for name in self.fingers:
                            if (name in spread and
                                name in self.spread_cal.baseline_spread and
                                name in self.spread_cal.min_spread and
                                name in self.spread_cal.max_spread):
                                scaled_val = self.spread_cal.scale_spread(
                                    spread[name],
                                    self.spread_cal.baseline_spread[name],
                                    self.spread_cal.min_spread[name],
                                    self.spread_cal.max_spread[name]
                                )
                                scaled[name] = round(scaled_val, 2)
                        self.on_scaled_spread(scaled)
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.multi_hand_landmarks[0],
                        mp_hands.HAND_CONNECTIONS
                    )
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

# THIS IS FOR THE CURLING LOGIC
# class HandTracker:
#    def __init__(self, on_scaled_angles):
#        self.on_scaled_angles = on_scaled_angles
#        self.finger_mins = {}
#        self.finger_maxs = {}
#        self.mp_hands = mp.solutions.hands
#        self.mp_drawing = mp.solutions.drawing_utils
#        self.fingers = {
#            "index": [5, 6, 7, 8],
#            "middle": [9, 10, 11, 12],
#            "ring": [13, 14, 15, 16],
#            "pinky": [17, 18, 19, 20]
#        }


#    def calculate_angle(self, a, b, c):
#        ab = [b[i] - a[i] for i in range(3)]
#        cb = [b[i] - c[i] for i in range(3)]
#        dot = sum(ab[i] * cb[i] for i in range(3))
#        mag_ab = math.sqrt(sum(x ** 2 for x in ab))
#        mag_cb = math.sqrt(sum(x ** 2 for x in cb))
#        if mag_ab * mag_cb == 0: return 0
#        angle = math.acos(dot / (mag_ab * mag_cb))
#        return math.degrees(angle)


#    def scale_angle(self, name, raw):
#        min_a = self.finger_mins.get(name, 30)
#        max_a = self.finger_maxs.get(name, 160)
#        if max_a - min_a == 0: return 90
#        # INVERTED: when raw is max, returns 0; when raw is min, returns 180
#        return round(max(0, min(1, (max_a - raw) / (max_a - min_a))) * 180)


#    def calibrate(self):
#        def capture_position(position):
#            root = tk.Tk()
#            root.withdraw()
#            messagebox.showinfo(f"{position} Calibration", f"Please {position.lower()} your hand and click OK")
#            cap = cv2.VideoCapture(0)
#            ret, frame = cap.read()
#            frame = cv2.flip(frame, 1)
#            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


#            with self.mp_hands.Hands(max_num_hands=1) as hands:
#                result = hands.process(rgb)


#            cap.release()
#            root.destroy()


#            if not result.multi_hand_landmarks:
#                print("No hand detected.")
#                return None


#            lm = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_landmarks[0].landmark]
#            snapshot = {}
#            for name, ids in self.fingers.items():
#                a, b, c = lm[ids[0]], lm[ids[1]], lm[ids[2]]
#                snapshot[name] = self.calculate_angle(a, b, c)
#            return snapshot


#        maxs = capture_position("Open")
#        mins = capture_position("Closed")
#        if maxs and mins:
#            self.finger_maxs = maxs
#            self.finger_mins = mins
#            print("Calibration complete.")
#        else:
#            print("Calibration failed.")


#    def run_tracking(self):
#        cap = cv2.VideoCapture(0)
#        with self.mp_hands.Hands(max_num_hands=1) as hands:
#            while cap.isOpened():
#                ret, frame = cap.read()
#                if not ret:
#                    break
#                frame = cv2.flip(frame, 1)
#                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                result = hands.process(rgb)


#                if result.multi_hand_landmarks:
#                    landmarks = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_landmarks[0].landmark]
#                    scaled = {}
#                    for name, ids in self.fingers.items():
#                        a, b, c = landmarks[ids[0]], landmarks[ids[1]], landmarks[ids[2]]
#                        raw_angle = self.calculate_angle(a, b, c)
#                        scaled[name] = self.scale_angle(name, raw_angle)
#                    self.on_scaled_angles(scaled)
#                    self.mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0],
#                                                   self.mp_hands.HAND_CONNECTIONS)


#                cv2.imshow("Tracking", frame)
#                if cv2.waitKey(1) & 0xFF == ord('q'):
#                    break


#        cap.release()
#        cv2.destroyAllWindows()




class RobotHandClient:
   def __init__(self, ip="192.168.0.143", port=9999):
       self.addr = (ip, port)
       self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
       self.sock.connect(self.addr)
       print(f"[INFO] Connected to Raspberry Pi at {ip}:{port}")


   def send_angles(self, angles):
       try:
           data = json.dumps(angles) + "\n"  # Add newline as message delimiter
           self.sock.sendall(data.encode())
       except Exception as e:
           print(f"[ERROR] Sending angles failed: {e}")




class FilteredSender:
   def __init__(self, send_callback, alpha=ALPHA_VALUE):
       self.send_callback = send_callback
       self.alpha = alpha
       self.filtered_angles = {}


   def send_filtered_angles(self, new_angles):
       smoothed = {}
       for finger, angle in new_angles.items():
           if finger not in self.filtered_angles:
               self.filtered_angles[finger] = angle
           else:
               # Apply low-pass filter: new = alpha * old + (1 - alpha) * new
               self.filtered_angles[finger] = (
                       self.alpha * self.filtered_angles[finger] + (1 - self.alpha) * angle
               )
           smoothed[finger] = round(self.filtered_angles[finger])
       print(f"[DEBUG] Sending angles to servos: {smoothed}")  # Debug print
       self.send_callback(smoothed)


class RobotHandApp:
   def __init__(self):
       self.client = RobotHandClient()
       self.filtered_sender = FilteredSender(self.client.send_angles)
       self.tracker = HandTracker(self.filtered_sender.send_filtered_angles)

   def start(self):
       root = tk.Tk()
       root.title("Robot Hand Client")
       tk.Button(root, text="Calibrate", font=("Arial", 12),
                 command=self.tracker.calibrate).pack(pady=5)
       tk.Button(root, text="Start Tracking", font=("Arial", 12),
                 command=lambda: threading.Thread(target=self.tracker.run_tracking, daemon=True).start()).pack(pady=5)
       root.mainloop()


# CODE FOR CURLING FINGER LOGIC
# class RobotHandApp:
#    def __init__(self):
#        self.client = RobotHandClient()
#        self.filtered_sender = FilteredSender(self.client.send_angles)
#        self.tracker = HandTracker(self.filtered_sender.send_filtered_angles)


#    def start(self):
#        root = tk.Tk()
#        root.title("Robot Hand Client")


#        tk.Button(root, text="Calibrate", font=("Arial", 12),
#                  command=self.tracker.calibrate).pack(pady=5)


#        tk.Button(root, text="Start Tracking", font=("Arial", 12),
#                  command=lambda: threading.Thread(target=self.tracker.run_tracking, daemon=True).start()).pack(pady=5)


#        root.mainloop()




if __name__ == "__main__":
   RobotHandApp().start()
