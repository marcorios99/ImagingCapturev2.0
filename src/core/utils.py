import os
from core.constants import ProcessingSpeed

class SpeedController:
    def __init__(self):
        self.total_cores = os.cpu_count() or 4
        
    def get_workers(self, speed: ProcessingSpeed, manual_cores: int = None) -> int:
        if speed == ProcessingSpeed.FULL:
            return self.total_cores
        elif speed == ProcessingSpeed.MODERATE:
            return max(1, int(self.total_cores * 0.4))
        elif speed == ProcessingSpeed.MANUAL:
            return min(max(1, manual_cores), self.total_cores)
        return 1