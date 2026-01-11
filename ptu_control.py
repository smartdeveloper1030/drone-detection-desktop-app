"""
PTU57/42 Gimbal Control Module
Handles serial communication with PTU57/42 gimbal and safety limits.
"""
import serial
import serial.tools.list_ports
import time
import logging
from typing import Optional, Tuple, List
from threading import Lock

logger = logging.getLogger(__name__)


class PTUControl:
    """
    PTU57/42 Gimbal Control Class
    
    Based on PTU57/42 manual:
    - Default baud rate: 9600
    - Commands end with 'E' (e.g., "H12,45,30,20E")
    - Returns "Done" after each command
    - Safety limits must be set on connection
    """
    
    # PTU57 default limits (from manual)
    PTU57_AZIMUTH_MIN = -90.0  # degrees
    PTU57_AZIMUTH_MAX = 180.0  # degrees
    PTU57_PITCH_MIN = -90.0    # degrees
    PTU57_PITCH_MAX = 135.0    # degrees
    
    # PTU42 default limits (from manual)
    PTU42_AZIMUTH_MIN = -80.0
    PTU42_AZIMUTH_MAX = 180.0
    PTU42_PITCH_MIN = -90.0
    PTU42_PITCH_MAX = 120.0
    
    def __init__(self, model: str = "PTU42"):
        """
        Initialize PTU control.
        
        Args:
            model: PTU model ("PTU57" or "PTU42")
        """
        self.model = model.upper()
        self.serial_port: Optional[serial.Serial] = None
        self.is_connected = False
        self.lock = Lock()
        
        # Set limits based on model
        if self.model == "PTU42":
            self.azimuth_min = self.PTU42_AZIMUTH_MIN
            self.azimuth_max = self.PTU42_AZIMUTH_MAX
            self.pitch_min = self.PTU42_PITCH_MIN
            self.pitch_max = self.PTU42_PITCH_MAX
        else:  # PTU57
            self.azimuth_min = self.PTU57_AZIMUTH_MIN
            self.azimuth_max = self.PTU57_AZIMUTH_MAX
            self.pitch_min = self.PTU57_PITCH_MIN
            self.pitch_max = self.PTU57_PITCH_MAX
        
        # Current position
        self.current_azimuth = 0.0
        self.current_pitch = 0.0
        
        # Safety limits (can be adjusted)
        self.safety_azimuth_min = self.azimuth_min
        self.safety_azimuth_max = self.azimuth_max
        self.safety_pitch_min = self.pitch_min
        self.safety_pitch_max = self.pitch_max
    
    def get_available_ports(self) -> List[str]:
        """
        Get list of available serial ports.
        
        Returns:
            List of port names (e.g., ['COM3', 'COM4'])
        """
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]
    
    def connect(self, port: str, baud_rate: int = 9600, timeout: float = 1.0) -> bool:
        """
        Connect to PTU via serial port and set safety limits.
        
        Args:
            port: Serial port name (e.g., "COM3")
            baud_rate: Baud rate (default 9600)
            timeout: Serial timeout in seconds
            
        Returns:
            True if connection successful and limits set
        """
        with self.lock:
            if self.is_connected:
                logger.warning("PTU already connected. Disconnect first.")
                return False
            
            try:
                # Open serial port
                self.serial_port = serial.Serial(
                    port=port,
                    baudrate=baud_rate,
                    timeout=timeout,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                
                # Wait for connection to stabilize
                time.sleep(0.5)
                
                # Clear any existing data
                if self.serial_port.in_waiting > 0:
                    self.serial_port.reset_input_buffer()
                
                # Do NOT set hardware safety limits automatically on connect.
                # Use software safety limits only to avoid unexpected hardware writes.
                logger.info(f"Using software safety limits for {self.model} (no hardware changes).")

                # Mark connected so we can send the initial safe move command.
                self.is_connected = True

                # Move to zero position (safe position). If this fails, log and continue
                logger.info("Moving to zero position...")
                try:
                    self.move_to_position(0.0, 0.0)
                    time.sleep(1.0)  # Wait for movement to complete
                except Exception as e:
                    logger.warning(f"Move to zero failed: {e}")

                logger.info(f"PTU {self.model} connected successfully on {port}")
                return True
                
            except serial.SerialException as e:
                logger.error(f"Serial connection error: {str(e)}")
                if self.serial_port:
                    try:
                        self.serial_port.close()
                    except:
                        pass
                    self.serial_port = None
                return False
            except Exception as e:
                logger.error(f"Unexpected error during connection: {str(e)}")
                if self.serial_port:
                    try:
                        self.serial_port.close()
                    except:
                        pass
                    self.serial_port = None
                return False
    
    def disconnect(self):
        """Disconnect from PTU."""
        with self.lock:
            if self.serial_port and self.serial_port.is_open:
                try:
                    # Move to safe position before disconnecting
                    self.move_to_position(0.0, 0.0)
                    time.sleep(0.5)
                except:
                    pass
                
                try:
                    self.serial_port.close()
                except:
                    pass
                
                self.serial_port = None
                self.is_connected = False
                logger.info("PTU disconnected")
    
    def _send_command(self, command: str) -> bool:
        """
        Send command to PTU.
        
        Args:
            command: Command string (e.g., "H12,45,30,20E")
        
        Returns:
            True if command sent successfully
        """
        if not self.is_connected or not self.serial_port:
            logger.error("PTU not connected")
            return False
        
        try:
            # Ensure command ends with 'E'
            if not command.endswith('E'):
                command += 'E'
            
            # Send command
            self.serial_port.write(command.encode('ascii'))
            self.serial_port.flush()
            logger.debug(f"Sent command: {command}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending command: {str(e)}")
            return False
    
    def _read_response(self, timeout: float = 1.0) -> str:
        """
        Read response from PTU.
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            Response string
        """
        if not self.is_connected or not self.serial_port:
            return ""
        
        try:
            start_time = time.time()
            response = ""
            
            while (time.time() - start_time) < timeout:
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.read(self.serial_port.in_waiting)
                    response += data.decode('ascii', errors='ignore')
                    if "Done" in response:
                        break
                time.sleep(0.01)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error reading response: {str(e)}")
            return ""
    
    def _clamp_angles(self, azimuth: float, pitch: float) -> Tuple[float, float]:
        """
        Clamp angles to safety limits.
        
        Args:
            azimuth: Azimuth angle in degrees
            pitch: Pitch angle in degrees
        
        Returns:
            Clamped (azimuth, pitch) tuple
        """
        azimuth = max(self.safety_azimuth_min, min(self.safety_azimuth_max, azimuth))
        pitch = max(self.safety_pitch_min, min(self.safety_pitch_max, pitch))
        return azimuth, pitch
    
    def move_to_position(self, azimuth: float, pitch: float, speed: int = 20) -> bool:
        """
        Move PTU to absolute position.
        
        Args:
            azimuth: Azimuth angle in degrees (-90 to 180 for PTU57)
            pitch: Pitch angle in degrees (-90 to 135 for PTU57)
            speed: Movement speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            logger.error("PTU not connected")
            return False
        
        # Clamp to safety limits
        azimuth, pitch = self._clamp_angles(azimuth, pitch)
        
        # Clamp speed
        speed = max(0, min(100, speed))
        
        # Format: H12,azimuth,pitch,speedE
        command = f"H12,{azimuth:.2f},{pitch:.2f},{speed}E"
        
        success = self._send_command(command)
        
        if success:
            self.current_azimuth = azimuth
            self.current_pitch = pitch
            logger.debug(f"Moving to: Azimuth={azimuth:.2f}°, Pitch={pitch:.2f}°, Speed={speed}%")
        
        return success
    
    def move_relative(self, delta_azimuth: float, delta_pitch: float, speed: int = 20) -> bool:
        """
        Move PTU relative to current position.
        
        Args:
            delta_azimuth: Change in azimuth angle (degrees)
            delta_pitch: Change in pitch angle (degrees)
            speed: Movement speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        new_azimuth = self.current_azimuth + delta_azimuth
        new_pitch = self.current_pitch + delta_pitch
        return self.move_to_position(new_azimuth, new_pitch, speed)
    
    def set_speed(self, speed: int) -> bool:
        """
        Set movement speed (for subsequent commands).
        
        Args:
            speed: Speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        # Speed is typically set per command, but we can store it for future use
        self.default_speed = max(0, min(100, speed))
        return True
    
    def go_to_zero(self, speed: int = 20) -> bool:
        """
        Move to zero position (0, 0).
        
        Args:
            speed: Movement speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        return self.move_to_position(0.0, 0.0, speed)
    
    def stop(self) -> bool:
        """
        Stop PTU movement (pause command).
        
        Returns:
            True if command sent successfully
        """
        # H61: Pause (from manual, manual control command)
        # For programming, we might need a different approach
        # Try sending a move to current position with 0 speed
        return self.move_to_position(self.current_azimuth, self.current_pitch, 0)
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current PTU position.
        
        Returns:
            (azimuth, pitch) tuple in degrees
        """
        return (self.current_azimuth, self.current_pitch)
    
