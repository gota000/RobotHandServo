import socket
import json
from board import SCL, SDA
import busio
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo
from SafeServo import SafeServo


# Constants
FREQ = 50
ANGLE_OFFSET = 10


FINGER_CHANNELS = {
   "index": {"angle": 15, "curl": 14, "mid_angle": 78},
   "middle": {"angle": 12, "curl": 13, "mid_angle": 77},
   "ring": {"angle": 11, "curl": 10, "mid_angle": 81},
   "pinky": {"angle": 9, "curl": 8, "mid_angle": 55},
   "thumb": {"angle": 6, "curl": 7, "mid_angle": 90}
}


#test

class Finger:
   def __init__(self, pca, name, angle_ch, curl_ch, mid_angle):
       unsafe = servo.Servo(pca.channels[angle_ch], min_pulse=500, max_pulse=2600)
       self.angle_servo = SafeServo(unsafe, mid_angle - ANGLE_OFFSET, mid_angle + ANGLE_OFFSET)
       self.curl_servo = servo.Servo(pca.channels[curl_ch], min_pulse=500, max_pulse=2600)
       self.mid_angle = mid_angle  # Store mid_angle for initialization

   def set_curl_angle(self, angle):
       self.curl_servo.angle = angle

   def set_angle_to_mid(self):
       self.angle_servo.set_angle(self.mid_angle)

# ...existing code...

class HandController:
   def __init__(self):
       i2c = busio.I2C(SCL, SDA)
       self.pca = PCA9685(i2c)
       self.pca.frequency = FREQ
       self.fingers = {
           name: Finger(self.pca, name, data["angle"], data["curl"], data["mid_angle"])
           for name, data in FINGER_CHANNELS.items()
       }

   def apply_angles(self, angles: dict):
       # Get intended servo angles (default to 0 if not present)
       thumb_angle = angles.get("thumb", 0)
       pointer_angle = angles.get("index", 0)
       middle_angle = angles.get("middle", 0)

       # --- Collision logic ---
       # If thumb > 90, pointer and middle cannot go above 90
       if thumb_angle > 140:
           if pointer_angle > 110:
               pointer_angle = 110
           if middle_angle > 100:
               middle_angle = 100
       # If pointer or middle > 90, thumb cannot go above 90
       if pointer_angle > 110 or middle_angle > 100:
       # if pointer_angle > 90:
           if thumb_angle > 140:
               thumb_angle = 140

       # Update the angles dict with possibly limited values
       angles = dict(angles)
       angles["thumb"] = thumb_angle
       angles["index"] = pointer_angle
       angles["middle"] = middle_angle

       for name, val in angles.items():
           if name in self.fingers:
               self.fingers[name].set_curl_angle(180 - val)


   def set_all_angles_to_mid(self):
       for finger in self.fingers.values():
           finger.set_angle_to_mid()



def run_server(host='0.0.0.0', port=9999):
   controller = HandController()
   controller.set_all_angles_to_mid()  # Move all angle servos to mid at startup
   print(f"[INFO] Listening for control on {host}:{port}")
   with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
       s.bind((host, port))
       s.listen(1)
       conn, addr = s.accept()
       print(f"[INFO] Connection from {addr}")
       with conn:
           buffer = ""
           while True:
               chunk = conn.recv(1024)
               if not chunk:
                   break
               buffer += chunk.decode()
               while "\n" in buffer:
                   line, buffer = buffer.split("\n", 1)
                   if not line.strip():
                       continue
                   try:
                       angles = json.loads(line)
                       controller.apply_angles(angles)
                   except Exception as e:
                       print(f"[ERROR] Invalid data: {e}")




if __name__ == "__main__":
   run_server()
