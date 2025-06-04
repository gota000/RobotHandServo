import time
import threading
import tkinter as tk
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

from SafeServo import SafeServo  # Custom class to restrict servo movement within a safe angle range

# -------------------------------
# Constants for mid-angles and allowed offset
# -------------------------------
POINTER_MID_ANGLE = 80
MIDDLE_MID_ANGLE = 83
RING_MID_ANGLE = 82
PINKY_MID_ANGLE = 85
ANGLE_OFFSET = 10  # Max allowed offset from mid-angle in both directions

# -------------------------------
# Initialize I2C and PWM driver
# -------------------------------
i2c = busio.I2C(SCL, SDA)  # Setup I2C communication
pca = PCA9685(i2c)  # Create PCA9685 PWM controller instance
pca.frequency = 50  # Set PWM frequency for servos (50Hz)

# -------------------------------
# Create and wrap angle servos with SafeServo for safety
# -------------------------------
pointer_angle_servo_unsafe = servo.Servo(pca.channels[15], min_pulse=500, max_pulse=2600)
pointer_angle_servo = SafeServo(pointer_angle_servo_unsafe, POINTER_MID_ANGLE - ANGLE_OFFSET,
                                POINTER_MID_ANGLE + ANGLE_OFFSET)

middle_angle_servo_unsafe = servo.Servo(pca.channels[12], min_pulse=500, max_pulse=2600)
middle_angle_servo = SafeServo(middle_angle_servo_unsafe, MIDDLE_MID_ANGLE - ANGLE_OFFSET,
                               MIDDLE_MID_ANGLE + ANGLE_OFFSET)

ring_angle_servo_unsafe = servo.Servo(pca.channels[11], min_pulse=500, max_pulse=2600)
ring_angle_servo = SafeServo(ring_angle_servo_unsafe, RING_MID_ANGLE - ANGLE_OFFSET,
                             RING_MID_ANGLE + ANGLE_OFFSET)

pinky_angle_servo_unsafe = servo.Servo(pca.channels[9], min_pulse=500, max_pulse=2600)
pinky_angle_servo = SafeServo(pinky_angle_servo_unsafe, PINKY_MID_ANGLE - ANGLE_OFFSET,
                              PINKY_MID_ANGLE + ANGLE_OFFSET)

# -------------------------------
# Curl servos – direct control, no safety limits
# -------------------------------
pointer_curl_servo = servo.Servo(pca.channels[14], min_pulse=500, max_pulse=2600)
middle_curl_servo = servo.Servo(pca.channels[13], min_pulse=500, max_pulse=2600)
ring_curl_servo = servo.Servo(pca.channels[10], min_pulse=500, max_pulse=2600)
pinky_curl_servo = servo.Servo(pca.channels[8], min_pulse=500, max_pulse=2600)


# -------------------------------
# Initialize all angle servos to their neutral (mid) positions
# -------------------------------
def init_angles():
    pointer_angle_servo.set_angle(POINTER_MID_ANGLE)
    middle_angle_servo.set_angle(MIDDLE_MID_ANGLE)
    ring_angle_servo.set_angle(RING_MID_ANGLE)
    pinky_angle_servo.set_angle(PINKY_MID_ANGLE)


# -------------------------------
# Set all curl servos to uncurled (180°)
# -------------------------------
def uncurl_fingers():
    pointer_curl_servo.angle = 180
    middle_curl_servo.angle = 180
    ring_curl_servo.angle = 180
    pinky_curl_servo.angle = 180


# -------------------------------
# Set all curl servos to curled positions
# -------------------------------
def curl_fingers():
    pointer_curl_servo.angle = 0
    middle_curl_servo.angle = 0
    ring_curl_servo.angle = 0
    pinky_curl_servo.angle = 45  # Slightly less curled than others


# -------------------------------
# Run servo command in background thread to keep GUI responsive
# -------------------------------
def run_in_thread(func):
    threading.Thread(target=func, daemon=True).start()


# -------------------------------
# GUI creation and logic
# -------------------------------
def create_gui():
    window = tk.Tk()
    window.title("Hand Servo Control")

    # --- Add a centered angle control slider ---
    def add_centered_angle_slider(label, servo_obj, mid_angle):
        frame = tk.Frame(window)
        frame.pack(pady=5)
        tk.Label(frame, text=label).pack()

        def update_servo(delta):
            # Adjust servo angle relative to its mid-angle
            target_angle = mid_angle + float(delta)
            run_in_thread(lambda: servo_obj.set_angle(target_angle))

        # Slider ranges from -ANGLE_OFFSET to +ANGLE_OFFSET
        scale = tk.Scale(
            frame, from_=-ANGLE_OFFSET, to=ANGLE_OFFSET,
            orient='horizontal', length=300,
            command=update_servo, resolution=1
        )
        scale.set(0)  # Start at center (mid-angle)
        scale.pack()

    # --- Add a curl slider that inverts values: 0 (uncurled) → 180, 100 (curled) → 0 ---
    def add_curl_slider(label, curl_servo):
        frame = tk.Frame(window)
        frame.pack(pady=5)
        tk.Label(frame, text=label).pack()

        def update_curl(val):
            # Invert slider: GUI 0 = uncurled → servo 180, GUI 100 = curled → servo 0
            inverted_angle = 180 - float(val) * 1.8  # Convert 0–100 slider to 180–0 degrees
            run_in_thread(lambda: setattr(curl_servo, 'angle', inverted_angle))

        scale = tk.Scale(
            frame, from_=0, to=100,
            orient='horizontal', length=300,
            command=update_curl, resolution=1
        )
        scale.set(0)  # Default to fully uncurled (180°)
        scale.pack()

    # --- Add all angle sliders for fingers ---
    add_centered_angle_slider("Pointer Angle", pointer_angle_servo, POINTER_MID_ANGLE)
    add_centered_angle_slider("Middle Angle", middle_angle_servo, MIDDLE_MID_ANGLE)
    add_centered_angle_slider("Ring Angle", ring_angle_servo, RING_MID_ANGLE)
    add_centered_angle_slider("Pinky Angle", pinky_angle_servo, PINKY_MID_ANGLE)

    # --- Add all curl sliders for fingers ---
    add_curl_slider("Pointer Curl", pointer_curl_servo)
    add_curl_slider("Middle Curl", middle_curl_servo)
    add_curl_slider("Ring Curl", ring_curl_servo)
    add_curl_slider("Pinky Curl", pinky_curl_servo)

    # --- Buttons to trigger predefined curl/uncurl gestures ---
    tk.Button(window, text="Curl All Fingers", font=("Arial", 14),
              command=lambda: run_in_thread(curl_fingers)).pack(pady=10)

    tk.Button(window, text="Uncurl All Fingers", font=("Arial", 14),
              command=lambda: run_in_thread(uncurl_fingers)).pack(pady=10)

    window.mainloop()  # Start the GUI event loop


# -------------------------------
# Main program logic
# -------------------------------
try:
    init_angles()  # Set initial neutral angles
    uncurl_fingers()  # Uncurl fingers at start
    create_gui()  # Launch control GUI

except KeyboardInterrupt:
    print("Program stopped.")
