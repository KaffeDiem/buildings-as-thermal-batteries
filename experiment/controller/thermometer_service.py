from datetime import datetime
from dataclasses import dataclass
import glob
import logging
import time
import math

logger = logging.getLogger(__name__)

@dataclass
class TemperatureReading:
    timestamp: datetime
    temperature: float  # Celsius

class ThermometerService:
    BASE_DIR = '/sys/bus/w1/devices/' # Assuming a Raspberry Pi with 1-Wire enabled
    
    def __init__(self) -> None:
        self.use_mock = False
        self.device_file = None
        self._mock_last_temp = 55.0  # Starting temperature for mock
        self._mock_start_time = time.time()
        
        # Try to find DS18B20 sensor
        try:
            device_folders = glob.glob(self.BASE_DIR + '28*')
            if device_folders:
                self.device_file = device_folders[0] + '/w1_slave'
                # Test if we can read from the device
                self._read_temp_raw()
                logger.info(f"DS18B20 sensor detected at {device_folders[0]}")
            else:
                self._enable_mock_mode()
        except Exception as e:
            logger.warning(f"Failed to initialize DS18B20 sensor: {e}")
            self._enable_mock_mode()
    
    def _enable_mock_mode(self):
        """Enable mock mode with a distinct warning message."""
        self.use_mock = True
        print("\n" + "="*70)
        print("⚠️  WARNING: DS18B20 sensor not available, using mock temperature values")
        print("="*70 + "\n")
        logger.warning("Operating in MOCK mode - DS18B20 sensor not available")
    
    def get_current_temperature(self) -> float:
        if self.use_mock:
            return self._get_mock_temperature()
        
        try:
            temp_c, _ = self._read_temp()
            return temp_c
        except Exception as e:
            logger.error(f"Failed to read from DS18B20 sensor: {e}")
            # Fall back to mock on read failure
            if not self.use_mock:
                self._enable_mock_mode()
            return self._get_mock_temperature()
    
    def _read_temp_raw(self):
        if self.device_file is None:
            raise Exception("Device file not initialized")
        with open(self.device_file, 'r') as f:
            lines = f.readlines()
        return lines
    
    def _read_temp(self):
        lines = self._read_temp_raw()
        
        # Check if read was successful (first line should end with 'YES')
        while lines[0].strip()[-3:] != 'YES':
            time.sleep(0.2)
            lines = self._read_temp_raw()
        
        # Parse temperature from second line
        equals_pos = lines[1].find('t=')
        if equals_pos != -1:
            temp_string = lines[1][equals_pos+2:]
            temp_c = float(temp_string) / 1000.0  # Convert from millidegrees
            temp_f = temp_c * 9.0 / 5.0 + 32.0
            return temp_c, temp_f
        
        raise Exception("Could not parse temperature from sensor data")
    
    def _get_mock_temperature(self) -> float:
        elapsed = time.time() - self._mock_start_time
        drift = 5.0 * math.sin(elapsed / 1800.0 * 2 * math.pi)
        noise = 1.5 * math.sin(elapsed * 0.1) * math.cos(elapsed * 0.07)
        target_temp = 55.0 + drift + noise
        self._mock_last_temp = self._mock_last_temp * 0.95 + target_temp * 0.05
        
        return round(self._mock_last_temp, 2)

