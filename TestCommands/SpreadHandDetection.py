import cv2
import mediapipe as mp
import math
import json
import os
import tkinter as tk
from tkinter import messagebox

class FingerSpreadTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.fingers = {
            "index": (8, 12),
            "middle": (12, 16),
            "ring": (16, 20),
            "pinky": (20, 16)
        }
        self.baseline_spread = {}
        self.min_spread = {}
        self.max_spread = {}

    def calculate_distance(self, p1, p2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

    def get_finger_spread(self, landmarks):
        # Normalize all distances by hand size (wrist to middle fingertip)
        reference_len = self.calculate_distance(landmarks[0], landmarks[12])
        if reference_len == 0:
            return None

        spread = {}
        for name, (tip, ref) in self.fingers.items():
            dist = self.calculate_distance(landmarks[tip], landmarks[ref])
            spread[name] = dist / reference_len  # Normalize
        return spread

    def average_frame_spread(self, num_frames=15):
        cap = cv2.VideoCapture(0)
        spreads_list = []

        with self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
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
                    cv2.putText(frame, f"Capturing calibration... {len(spreads_list)}/{num_frames}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Calibration", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

        if not spreads_list:
            print("Warning: No valid hand data captured during this step.")
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

        # Clamp the result
        return max(-5, min(5, scaled))

    def track(self):
        if not self.load_calibration():
            print("Please run calibration first.")
            return

        cap = cv2.VideoCapture(0)
        with self.mp_hands.Hands(max_num_hands=1) as hands:
            while True:
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
                        scaled = {}
                        for name in self.fingers:
                            if (name in spread and
                                name in self.baseline_spread and
                                name in self.min_spread and
                                name in self.max_spread):
                                scaled_val = self.scale_spread(
                                    spread[name],
                                    self.baseline_spread[name],
                                    self.min_spread[name],
                                    self.max_spread[name]
                                )
                                scaled[name] = round(scaled_val, 2)

                        print("Scaled finger values:", scaled)

                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, result.multi_hand_landmarks[0],
                        self.mp_hands.HAND_CONNECTIONS
                    )

                cv2.imshow("Finger Spread Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FingerSpreadTracker()

    print("Press 'c' to calibrate, or any other key to skip to tracking.")
    if input().lower() == 'c':
        tracker.calibrate()

    tracker.track()
