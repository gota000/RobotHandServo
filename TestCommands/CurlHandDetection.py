# Import required libraries
import cv2  # For webcam access and image processing
import mediapipe as mp  # For hand tracking
import math  # For angle calculations
import tkinter as tk  # For GUI
from tkinter import simpledialog  # For pop-up dialogs
import threading  # For running hand tracking and GUI at the same time

# --- Mediapipe setup for hand tracking ---
mp_hands = mp.solutions.hands  # Initialize hand detection module
mp_drawing = mp.solutions.drawing_utils  # For drawing hand landmarks

# Define finger joints used to compute angles
# Each finger is represented by 3 points for angle calculation: base, middle, tip
fingers = {
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

# These will store the calibration values
# `finger_mins`: Angle when the finger is fully closed
# `finger_maxs`: Angle when the finger is fully open
finger_mins = {}
finger_maxs = {}


# --- Helper function to calculate angle between three points in 3D space ---
def calculate_angle(a, b, c):
    # Create vectors from points
    ab = [b[i] - a[i] for i in range(3)]
    cb = [b[i] - c[i] for i in range(3)]

    # Dot product and magnitudes of vectors
    dot = sum(ab[i] * cb[i] for i in range(3))
    mag_ab = math.sqrt(sum(x ** 2 for x in ab))
    mag_cb = math.sqrt(sum(x ** 2 for x in cb))

    # Avoid division by zero
    if mag_ab * mag_cb == 0:
        return 0

    # Calculate angle in radians, then convert to degrees
    angle = math.acos(dot / (mag_ab * mag_cb))
    return math.degrees(angle)


# --- Scale the raw angle to a 0–180 degree range ---
def scale_angle(name, raw_angle):
    # If calibration is missing, return a neutral 90 degrees
    if name not in finger_mins or name not in finger_maxs:
        return 90

    min_angle = finger_mins[name]
    max_angle = finger_maxs[name]

    # Prevent divide by zero if angles are the same
    if max_angle - min_angle == 0:
        return 90

    # Normalize the angle to [0, 1]
    scaled = (raw_angle - min_angle) / (max_angle - min_angle)
    scaled = max(0, min(1, scaled))  # Clamp to 0–1

    # Convert to [0, 180]
    return round(scaled * 180, 1)


# --- Main hand tracking function (runs in background) ---
def run_hand_tracking():
    global finger_mins, finger_maxs

    cap = cv2.VideoCapture(0)  # Start webcam

    with mp_hands.Hands(
            model_complexity=1,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if frame not read

            # Flip image for natural view and convert to RGB
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process hand landmarks
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                # Use only the first detected hand
                hand_landmarks = result.multi_hand_landmarks[0]

                # Convert landmarks to a list of (x, y, z)
                landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

                angles = {}  # Raw angles
                scaled = {}  # Scaled angles (0–180)

                # Loop through each finger and compute angles
                for name, ids in fingers.items():
                    a, b, c = landmarks[ids[0]], landmarks[ids[1]], landmarks[ids[2]]
                    angle = calculate_angle(a, b, c)
                    angles[name] = angle
                    scaled[name] = scale_angle(name, angle)

                # Print scaled angles for debug/servo control
                print("Scaled Angles (0–180):", scaled)

                # Draw the hand landmarks on the video feed
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Show the video feed
            cv2.imshow("Hand Tracking with Calibration", frame)

            # Exit on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


# --- Calibration function to record open/closed hand angles ---
def calibrate_hand():
    global finger_mins, finger_maxs

    # Helper function to prompt user and capture one hand position
    def capture_position(position_name):
        root = tk.Tk()
        root.withdraw()  # Hide the root window

        # Prompt user to pose their hand and click OK
        simpledialog.messagebox.showinfo(
            f"{position_name} Calibration",
            f"Please {position_name.lower()} your hand fully and click OK..."
        )

        # Capture one frame from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the hand in the frame
        with mp_hands.Hands(
                model_complexity=1,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
        ) as hands:
            result = hands.process(rgb)

        if result.multi_hand_landmarks:
            # Extract 3D coordinates of landmarks
            landmarks = [(lm.x, lm.y, lm.z) for lm in result.multi_hand_landmarks[0].landmark]
            angle_snapshot = {}

            # Compute angle for each finger
            for name, ids in fingers.items():
                a, b, c = landmarks[ids[0]], landmarks[ids[1]], landmarks[ids[2]]
                angle = calculate_angle(a, b, c)
                angle_snapshot[name] = angle

            print(f"{position_name} angles:", angle_snapshot)
            cap.release()
            root.destroy()
            return angle_snapshot
        else:
            print(f"Failed to detect hand during {position_name} calibration.")
            cap.release()
            root.destroy()
            return None

    # Capture open hand angles (max) and closed hand angles (min)
    maxs = capture_position("Open")
    mins = capture_position("Closed")

    # Save calibration data if successful
    if mins and maxs:
        finger_mins = mins
        finger_maxs = maxs
        print("Calibration complete!")
    else:
        print("Calibration failed. Try again.")


# --- GUI to start calibration and tracking ---
def start_gui():
    root = tk.Tk()
    root.title("Calibration GUI")

    # Title label
    tk.Label(root, text="Robot Hand Calibration", font=("Arial", 14)).pack(pady=10)

    # Calibration button
    tk.Button(root, text="Start Calibration", font=("Arial", 12),
              command=calibrate_hand).pack(pady=5)

    # Hand tracking button (runs in new thread to avoid freezing GUI)
    tk.Button(root, text="Start Hand Tracking", font=("Arial", 12),
              command=lambda: threading.Thread(target=run_hand_tracking, daemon=True).start()).pack(pady=5)

    # Start the GUI loop
    root.mainloop()


# --- Run the program ---
if __name__ == "__main__":
    start_gui()
