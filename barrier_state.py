class BarrierStateMachine:
    def __init__(self, min_frames_presence=5):
        self.state = "NO_VEHICLE"
        self.presence_counter = 0
        self.min_frames_presence = min_frames_presence

    def update(self, vehicle_detected: bool):
        if vehicle_detected:
            self.presence_counter += 1
        else:
            self.presence_counter = 0

        if self.presence_counter >= self.min_frames_presence:
            self.state = "VEHICLE_WAITING"
        elif self.presence_counter > 0:
            self.state = "VEHICLE_APPROACHING"
        else:
            self.state = "NO_VEHICLE"

        return self.state