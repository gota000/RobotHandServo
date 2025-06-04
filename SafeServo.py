class SafeServo:
    def __init__(self, servo, min_angle=0, max_angle=180):
        self.servo = servo
        self.min_angle = min_angle
        self.max_angle = max_angle

    def set_angle(self, angle):
        safe_angle = max(self.min_angle, min(self.max_angle, angle))
        self.servo.angle = safe_angle
