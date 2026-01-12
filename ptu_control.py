"""
PTU57/42 Gimbal Control Module
Handles serial communication with PTU57/42 gimbal and safety limits.
"""
import serial
import serial.tools.list_ports
import time
import logging
import threading
from queue import Queue, Empty
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class PTUCommandType(Enum):
    """Command types for PTU thread communication."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    MOVE_TO_POSITION = "move_to_position"
    MOVE_RELATIVE = "move_relative"
    STOP = "stop"
    GO_TO_ZERO = "go_to_zero"
    SET_SPEED = "set_speed"
    GET_POSITION = "get_position"
    SHUTDOWN = "shutdown"


class PTUControlThread(threading.Thread):
    """
    Background thread for PTU serial communication.
    Prevents UI blocking during serial operations.
    """
    
    def __init__(self, model: str = "PTU42"):
        """
        Initialize PTU control thread.
        
        Args:
            model: PTU model ("PTU57" or "PTU42")
        """
        super().__init__(daemon=True)
        self.model = model.upper()
        self.serial_port: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_running = False
        
        # Command queue (from main thread to PTU thread)
        self.command_queue = Queue()
        
        # Response queue (from PTU thread to main thread)
        self.response_queue = Queue()
        
        # Set limits based on model
        if self.model == "PTU42":
            self.azimuth_min = -80.0
            self.azimuth_max = 180.0
            self.pitch_min = -90.0
            self.pitch_max = 120.0
        else:  # PTU57
            self.azimuth_min = -90.0
            self.azimuth_max = 180.0
            self.pitch_min = -90.0
            self.pitch_max = 135.0
        
        # Safety limits
        self.safety_azimuth_min = self.azimuth_min
        self.safety_azimuth_max = self.azimuth_max
        self.safety_pitch_min = self.pitch_min
        self.safety_pitch_max = self.pitch_max
        
        # Current position (thread-safe, updated by thread)
        self.current_azimuth = 0.0
        self.current_pitch = 0.0
        self.position_lock = threading.Lock()
        
        # Default speed
        self.default_speed = 20
    
    def run(self):
        """Main thread loop - continuously process commands."""
        self.is_running = True
        logger.info("PTU control thread started")
        
        while self.is_running:
            try:
                # Get command from queue (blocking with timeout)
                try:
                    command_type, command_id, args = self.command_queue.get(timeout=0.1)
                except Empty:
                    continue
                
                # Process command
                result = self._process_command(command_type, args)
                
                # Send response back
                self.response_queue.put((command_id, result))
                self.command_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in PTU control thread: {str(e)}")
                if not self.is_running:
                    break
        
        # Cleanup on shutdown
        self._cleanup()
        logger.info("PTU control thread stopped")
    
    def _process_command(self, command_type: PTUCommandType, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a command in the thread.
        
        Args:
            command_type: Type of command
            args: Command arguments
            
        Returns:
            Result dictionary with 'success' and optional data
        """
        try:
            if command_type == PTUCommandType.CONNECT:
                return self._handle_connect(args)
            elif command_type == PTUCommandType.DISCONNECT:
                return self._handle_disconnect()
            elif command_type == PTUCommandType.MOVE_TO_POSITION:
                return self._handle_move_to_position(args)
            elif command_type == PTUCommandType.MOVE_RELATIVE:
                return self._handle_move_relative(args)
            elif command_type == PTUCommandType.STOP:
                return self._handle_stop()
            elif command_type == PTUCommandType.GO_TO_ZERO:
                return self._handle_go_to_zero(args)
            elif command_type == PTUCommandType.SET_SPEED:
                return self._handle_set_speed(args)
            elif command_type == PTUCommandType.GET_POSITION:
                return self._handle_get_position()
            elif command_type == PTUCommandType.SHUTDOWN:
                self.is_running = False
                return {'success': True}
            else:
                return {'success': False, 'error': f'Unknown command: {command_type}'}
        except Exception as e:
            logger.error(f"Error processing command {command_type}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _handle_connect(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle connect command."""
        if self.is_connected:
            return {'success': False, 'error': 'Already connected'}
        
        port = args['port']
        baud_rate = args.get('baud_rate', 9600)
        timeout = args.get('timeout', 1.0)
        
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
            
            logger.info(f"Using software safety limits for {self.model} (no hardware changes).")
            
            self.is_connected = True
            
            # Move to zero position
            logger.info("Moving to zero position...")
            try:
                self._send_command(f"H12,0.0,0.0,{self.default_speed}E")
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Move to zero failed: {e}")
            
            logger.info(f"PTU {self.model} connected successfully on {port}")
            return {'success': True}
            
        except serial.SerialException as e:
            logger.error(f"Serial connection error: {str(e)}")
            if self.serial_port:
                try:
                    self.serial_port.close()
                except:
                    pass
                self.serial_port = None
            return {'success': False, 'error': str(e)}
        except Exception as e:
            logger.error(f"Unexpected error during connection: {str(e)}")
            if self.serial_port:
                try:
                    self.serial_port.close()
                except:
                    pass
                self.serial_port = None
            return {'success': False, 'error': str(e)}
    
    def _handle_disconnect(self) -> Dict[str, Any]:
        """Handle disconnect command."""
        if not self.is_connected or not self.serial_port:
            return {'success': True}  # Already disconnected
        
        try:
            # Move to safe position before disconnecting
            self._send_command(f"H12,0.0,0.0,{self.default_speed}E")
            time.sleep(0.5)
        except:
            pass
        
        try:
            if self.serial_port.is_open:
                self.serial_port.close()
        except:
            pass
        
        self.serial_port = None
        self.is_connected = False
        logger.info("PTU disconnected")
        return {'success': True}
    
    def _handle_move_to_position(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move to position command."""
        if not self.is_connected:
            return {'success': False, 'error': 'PTU not connected'}
        
        azimuth = args['azimuth']
        pitch = args['pitch']
        speed = args.get('speed', self.default_speed)
        
        # Clamp to safety limits
        azimuth = max(self.safety_azimuth_min, min(self.safety_azimuth_max, azimuth))
        pitch = max(self.safety_pitch_min, min(self.safety_pitch_max, pitch))
        speed = max(0, min(100, speed))
        
        # Format: H12,azimuth,pitch,speedE
        command = f"H12,{azimuth:.2f},{pitch:.2f},{speed}E"
        success = self._send_command(command)
        
        if success:
            with self.position_lock:
                self.current_azimuth = azimuth
                self.current_pitch = pitch
            logger.debug(f"Moving to: Azimuth={azimuth:.2f}°, Pitch={pitch:.2f}°, Speed={speed}%")
        
        return {'success': success, 'azimuth': azimuth, 'pitch': pitch}
    
    def _handle_move_relative(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle move relative command."""
        if not self.is_connected:
            return {'success': False, 'error': 'PTU not connected'}
        
        with self.position_lock:
            current_azimuth = self.current_azimuth
            current_pitch = self.current_pitch
        
        delta_azimuth = args['delta_azimuth']
        delta_pitch = args['delta_pitch']
        speed = args.get('speed', self.default_speed)
        
        new_azimuth = current_azimuth + delta_azimuth
        new_pitch = current_pitch + delta_pitch
        
        return self._handle_move_to_position({
            'azimuth': new_azimuth,
            'pitch': new_pitch,
            'speed': speed
        })
    
    def _handle_stop(self) -> Dict[str, Any]:
        """Handle stop command."""
        if not self.is_connected:
            return {'success': False, 'error': 'PTU not connected'}
        
        with self.position_lock:
            current_azimuth = self.current_azimuth
            current_pitch = self.current_pitch
        
        # Move to current position with 0 speed
        command = f"H12,{current_azimuth:.2f},{current_pitch:.2f},0E"
        success = self._send_command(command)
        return {'success': success}
    
    def _handle_go_to_zero(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle go to zero command."""
        speed = args.get('speed', self.default_speed)
        return self._handle_move_to_position({
            'azimuth': 0.0,
            'pitch': 0.0,
            'speed': speed
        })
    
    def _handle_set_speed(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set speed command."""
        speed = args.get('speed', 20)
        self.default_speed = max(0, min(100, speed))
        return {'success': True, 'speed': self.default_speed}
    
    def _handle_get_position(self) -> Dict[str, Any]:
        """Handle get position command."""
        with self.position_lock:
            azimuth = self.current_azimuth
            pitch = self.current_pitch
        return {'success': True, 'azimuth': azimuth, 'pitch': pitch}
    
    def _send_command(self, command: str) -> bool:
        """Send command to PTU."""
        if not self.is_connected or not self.serial_port:
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
    
    def _cleanup(self):
        """Cleanup resources on thread shutdown."""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.close()
            except:
                pass
            self.serial_port = None
        self.is_connected = False
    
    def send_command(self, command_type: PTUCommandType, args: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        """
        Send command to thread and wait for response.
        
        Args:
            command_type: Type of command
            args: Command arguments
            timeout: Timeout in seconds
            
        Returns:
            Response dictionary
        """
        if not self.is_running:
            return {'success': False, 'error': 'Thread not running'}
        
        # Generate unique command ID
        command_id = time.time()
        
        # Send command
        self.command_queue.put((command_type, command_id, args))
        
        # Wait for response
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            try:
                response_id, result = self.response_queue.get(timeout=0.1)
                if response_id == command_id:
                    return result
                else:
                    # Wrong response, put it back
                    self.response_queue.put((response_id, result))
            except Empty:
                continue
        
        return {'success': False, 'error': 'Timeout waiting for response'}
    
    def stop(self):
        """Stop the thread."""
        self.is_running = False
        # Send shutdown command
        self.send_command(PTUCommandType.SHUTDOWN, {})
        logger.info("Stopping PTU control thread...")


class PTUControl:
    """
    PTU57/42 Gimbal Control Class
    
    Wrapper class that uses PTUControlThread for non-blocking serial communication.
    All serial operations are performed in a background thread to prevent UI blocking.
    
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
        
        # Create and start the control thread
        self.thread = PTUControlThread(model=self.model)
        self.thread.start()
        
        # Cached position (updated from thread responses)
        self.current_azimuth = 0.0
        self.current_pitch = 0.0
        
        # Set limits based on model (for reference)
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
        
        # Safety limits (can be adjusted)
        self.safety_azimuth_min = self.azimuth_min
        self.safety_azimuth_max = self.azimuth_max
        self.safety_pitch_min = self.pitch_min
        self.safety_pitch_max = self.pitch_max
    
    @property
    def is_connected(self) -> bool:
        """Check if PTU is connected (thread-safe)."""
        return self.thread.is_connected if self.thread else False
    
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
        Connect to PTU via serial port (non-blocking, uses thread).
        
        Args:
            port: Serial port name (e.g., "COM3")
            baud_rate: Baud rate (default 9600)
            timeout: Serial timeout in seconds
            
        Returns:
            True if connection successful
        """
        if not self.thread or not self.thread.is_running:
            logger.error("PTU control thread not running")
            return False
        
        result = self.thread.send_command(
            PTUCommandType.CONNECT,
            {'port': port, 'baud_rate': baud_rate, 'timeout': timeout},
            timeout=timeout + 2.0
        )
        
        if result.get('success'):
            # Update cached position
            pos_result = self.thread.send_command(PTUCommandType.GET_POSITION, {})
            if pos_result.get('success'):
                self.current_azimuth = pos_result.get('azimuth', 0.0)
                self.current_pitch = pos_result.get('pitch', 0.0)
        
        return result.get('success', False)
    
    def disconnect(self):
        """Disconnect from PTU (non-blocking, uses thread)."""
        if not self.thread or not self.thread.is_running:
            return
        
        self.thread.send_command(PTUCommandType.DISCONNECT, {}, timeout=2.0)
    
    def move_to_position(self, azimuth: float, pitch: float, speed: int = 20) -> bool:
        """
        Move PTU to absolute position (non-blocking, uses thread).
        
        Args:
            azimuth: Azimuth angle in degrees
            pitch: Pitch angle in degrees
            speed: Movement speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            logger.error("PTU not connected")
            return False
        
        # Clamp to safety limits
        azimuth, pitch = self._clamp_angles(azimuth, pitch)
        
        result = self.thread.send_command(
            PTUCommandType.MOVE_TO_POSITION,
            {'azimuth': azimuth, 'pitch': pitch, 'speed': speed},
            timeout=1.0
        )
        
        if result.get('success'):
            self.current_azimuth = result.get('azimuth', azimuth)
            self.current_pitch = result.get('pitch', pitch)
        
        return result.get('success', False)
    
    def move_relative(self, delta_azimuth: float, delta_pitch: float, speed: int = 20) -> bool:
        """
        Move PTU relative to current position (non-blocking, uses thread).
        
        Args:
            delta_azimuth: Change in azimuth angle (degrees)
            delta_pitch: Change in pitch angle (degrees)
            speed: Movement speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            logger.error("PTU not connected")
            return False
        
        result = self.thread.send_command(
            PTUCommandType.MOVE_RELATIVE,
            {'delta_azimuth': delta_azimuth, 'delta_pitch': delta_pitch, 'speed': speed},
            timeout=1.0
        )
        
        if result.get('success'):
            self.current_azimuth = result.get('azimuth', self.current_azimuth)
            self.current_pitch = result.get('pitch', self.current_pitch)
        
        return result.get('success', False)
    
    def set_speed(self, speed: int) -> bool:
        """
        Set movement speed (for subsequent commands).
        
        Args:
            speed: Speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        if not self.thread or not self.thread.is_running:
            return False
        
        result = self.thread.send_command(
            PTUCommandType.SET_SPEED,
            {'speed': speed},
            timeout=0.5
        )
        return result.get('success', False)
    
    def go_to_zero(self, speed: int = 20) -> bool:
        """
        Move to zero position (0, 0) (non-blocking, uses thread).
        
        Args:
            speed: Movement speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        return self.move_to_position(0.0, 0.0, speed)
    
    def stop(self) -> bool:
        """
        Stop PTU movement (non-blocking, uses thread).
        
        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            return False
        
        result = self.thread.send_command(PTUCommandType.STOP, {}, timeout=0.5)
        return result.get('success', False)
    
    def get_position(self) -> Tuple[float, float]:
        """
        Get current PTU position (thread-safe).
        
        Returns:
            (azimuth, pitch) tuple in degrees
        """
        if not self.thread or not self.thread.is_running:
            return (self.current_azimuth, self.current_pitch)
        
        # Try to get latest position from thread
        result = self.thread.send_command(PTUCommandType.GET_POSITION, {}, timeout=0.5)
        if result.get('success'):
            self.current_azimuth = result.get('azimuth', self.current_azimuth)
            self.current_pitch = result.get('pitch', self.current_pitch)
        
        return (self.current_azimuth, self.current_pitch)
    
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
    
    def cleanup(self):
        """Cleanup resources and stop thread."""
        if self.thread and self.thread.is_running:
            self.thread.stop()
            self.thread.join(timeout=2.0)
    