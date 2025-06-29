"""Автомат состояний для определения ТС."""

class BarrierStateMachine:
    """Класс для автомата состояний."""
    
    def __init__(self, min_frames_presence=10):
        """Инициализация автомата состояний.

        Args:
            min_frames_presence (int): Минимальное количество кадров, в течение которых транспортное средство должно быть обнаружено.
        """
        self.state = "NO_VEHICLE"
        self.presence_counter = 0
        self.min_frames_presence = min_frames_presence

    def update(self, vehicle_detected: bool):
        """Обновление состояния автомата на основе наличия транспортного средства.

        Args:
            vehicle_detected (bool): Флаг, указывающий, обнаружено ли транспортное средство.

        Returns:
            str: Текущее состояние автомата.
        """
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