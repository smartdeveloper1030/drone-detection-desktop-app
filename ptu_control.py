"""
PTU Gimbal Control Module
Handles serial communication with PTU gimbal and safety limits.
"""
import serial
import serial.tools.list_ports
import time
import logging
import threading
import re
from queue import Queue, Empty
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


# PTU limit constants - single source of truth (DRY principle)
PTU_AZIMUTH_MIN = -60.0  # degrees (PAN_MIN)
PTU_AZIMUTH_MAX = 60.0    # degrees (PAN_MAX)
PTU_PITCH_MIN = -45.0     # degrees (TILT_MIN)
PTU_PITCH_MAX = 45.0      # degrees (TILT_MAX)
PULSE_TO_DEGREE = 0.0009375
MAX_SPEED = 450

class PTUCommandType(Enum):
    """Command types for PTU thread communication."""
    CONNECT = "connect"
    DISCONNECT = "disconnect"
    MOVE_TO_POSITION = "move_to_position"
    MOVE_RELATIVE = "move_relative"
    MOVE_DIRECTIONAL = "move_directional"  # H61/H62/H63/H64 commands
    STOP = "stop"
    GO_TO_ZERO = "go_to_zero"
    SET_SPEED = "set_speed"
    SET_ACCELERATION = "set_acceleration"
    GET_POSITION = "get_position"
    SEND_RAW_COMMAND = "send_raw_command"  # Send raw command without waiting for Done
    SHUTDOWN = "shutdown"


class PTUControlThread(threading.Thread):
    """
    Background thread for PTU serial communication.
    Prevents UI blocking during serial operations.
    """
    
    def __init__(self):
        """
        Initialize PTU control thread.
        """
        super().__init__(daemon=True)
        self.serial_port: Optional[serial.Serial] = None
        self.is_connected = False
        self.is_running = False
        
        # Command queue (from main thread to PTU thread)
        self.command_queue = Queue()
        
        # Response queue (from PTU thread to main thread)
        self.response_queue = Queue()
        
        # Command history for logging
        self.command_history = []
        self.history_lock = threading.Lock()
        
        # Callback for updating UI command history (set from main thread)
        self.history_callback = None
        
        # Set limits (using module-level constants - DRY)
        # Note: These can be overridden by actual PTU configuration (H92 settings)
        self.azimuth_min = PTU_AZIMUTH_MIN
        self.azimuth_max = PTU_AZIMUTH_MAX
        self.pitch_min = PTU_PITCH_MIN
        self.pitch_max = PTU_PITCH_MAX
        
        # Safety limits
        self.safety_azimuth_min = self.azimuth_min
        self.safety_azimuth_max = self.azimuth_max
        self.safety_pitch_min = self.pitch_min
        self.safety_pitch_max = self.pitch_max
        
        # Current position (thread-safe, updated by thread)
        self.current_azimuth = 0.0
        self.current_pitch = 0.0
        self.position_lock = threading.Lock()
        
        # Default speed and acceleration
        self.default_speed = 20
        self.default_acceleration = 5
    
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
            elif command_type == PTUCommandType.MOVE_DIRECTIONAL:
                return self._handle_move_directional(args)
            elif command_type == PTUCommandType.STOP:
                return self._handle_stop()
            elif command_type == PTUCommandType.GO_TO_ZERO:
                return self._handle_go_to_zero(args)
            elif command_type == PTUCommandType.SET_SPEED:
                return self._handle_set_speed(args)
            elif command_type == PTUCommandType.GET_POSITION:
                return self._handle_get_position()
            elif command_type == PTUCommandType.SEND_RAW_COMMAND:
                return self._handle_send_raw_command(args)
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
            
            logger.info("Using software safety limits (no hardware changes).")
            
            self.is_connected = True
            
            # Move to zero position
            logger.info("Moving to zero position...")
            try:
                self._send_command(f"H51,0,0,{MAX_SPEED}E", wait_for_done=False)
                time.sleep(1.0)
            except Exception as e:
                logger.warning(f"Move to zero failed: {e}")
            
            logger.info(f"PTU connected successfully on {port}")
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
            self._send_command(f"H51,0,0,{MAX_SPEED}E", wait_for_done=False)
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
        
        # Speed conversion: Based on manual and user's config
        # User's config shows: H91: dAdH_speed (deg/s) = 8, H93: Max_speed_percent = 500
        # The H12 command speed parameter appears to be in degrees/second based on manual examples
        # Convert UI percentage (0-100) to degrees/second (0-8 based on user's config)
        if speed <= 100:
            # Treat as UI percentage: map 0-100% to 0-8 deg/s (user's configured max speed)
            # This gives slow, controlled movement
            speed_deg_per_sec = (speed / 100.0) * 8.0
            # Ensure minimum speed of 0.5 deg/s if speed > 0 to ensure movement
            if speed > 0 and speed_deg_per_sec < 0.5:
                speed_deg_per_sec = 0.5
        else:
            # Already in deg/s format, cap at reasonable max (8 deg/s per user config)
            speed_deg_per_sec = min(speed, 8.0)  # Use user's configured max
        
        # Format: H12,azimuth,pitch,speedE (speed in degrees/second)
        # Manual shows examples like "H12,45,30,20E" where parameters are integers
        # Convert to integers for PTU command format
        azimuth_int = int(round(azimuth))
        pitch_int = int(round(pitch))
        speed_int = int(round(speed_deg_per_sec))
        command = f"H51,{azimuth_int},{pitch_int},{speed_int}E"
        success = self._send_command(command, wait_for_done=True, timeout=5.0)
        
        if success:
            with self.position_lock:
                self.current_azimuth = azimuth
                self.current_pitch = pitch
            logger.info(f"Moving to: Azimuth={azimuth_int}°, Pitch={pitch_int}°, Speed={speed_int} deg/s")
        
        return {'success': success, 'azimuth': azimuth, 'pitch': pitch, 'command': command}
    
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
    
    def _handle_move_directional(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle directional movement command (H61/H62/H63/H64)."""
        if not self.is_connected:
            return {'success': False, 'error': 'PTU not connected'}
        
        direction = args.get('direction')  # 'left', 'right', 'up', 'down'
        speed = args.get('speed', self.default_speed)
        
        # Map direction to command
        command_map = {
            'left': 'H61',
            'right': 'H62',
            'up': 'H63',
            'down': 'H64'
        }
        
        if direction not in command_map:
            return {'success': False, 'error': f'Invalid direction: {direction}'}
        
        # Format command: H61,<speed>E (or H62, H63, H64)
        command = f"{command_map[direction]},{speed}E"
        success = self._send_command(command, wait_for_done=True, timeout=2.0)
        return {'success': success, 'command': command}
    
    def _handle_stop(self) -> Dict[str, Any]:
        """Handle stop command using H65E."""
        if not self.is_connected:
            return {'success': False, 'error': 'PTU not connected'}
        
        # Send H65E command to stop movement
        command = "H65E"
        success = self._send_command(command, wait_for_done=True, timeout=2.0)
        return {'success': success, 'command': command}
    
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
    
    def _handle_set_acceleration(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set acceleration command."""
        acceleration = args.get('acceleration', 5)
        self.default_acceleration = max(0, min(100, acceleration))
        # Acceleration is typically handled by PTU firmware automatically
        # But we store it for reference and can use it to influence movement smoothness
        return {'success': True, 'acceleration': self.default_acceleration}
    
    def _handle_get_position(self) -> Dict[str, Any]:
        """Handle get position command using H10 (azimuth/A1) and H20 (pitch/A2)."""
        if not self.is_connected:
            return {'success': False, 'error': 'PTU not connected'}
        
        try:
            # Send H10E to get azimuth position (A1)
            azimuth_response = self._send_command_with_response("H10E", timeout=2.0)
            if not azimuth_response:
                logger.warning("Failed to get azimuth position from H10 command")
                azimuth = self.current_azimuth  # Use cached value on failure
            else:
                azimuth = self._parse_position_response(azimuth_response, "azimuth")
            
            # Send H20E to get pitch position (A2)
            pitch_response = self._send_command_with_response("H20E", timeout=2.0)
            if not pitch_response:
                logger.warning("Failed to get pitch position from H20 command")
                pitch = self.current_pitch  # Use cached value on failure
            else:
                pitch = self._parse_position_response(pitch_response, "pitch")
            
            # Update cached position
            with self.position_lock:
                self.current_azimuth = azimuth
                self.current_pitch = pitch
            
            logger.info(f"Position retrieved: Azimuth={azimuth:.2f}°, Pitch={pitch:.2f}°")
            return {'success': True, 'azimuth': azimuth, 'pitch': pitch}
            
        except Exception as e:
            logger.error(f"Error getting position: {str(e)}")
            # Return cached values on error
            with self.position_lock:
                return {'success': False, 'error': str(e), 
                       'azimuth': self.current_azimuth, 'pitch': self.current_pitch}
    
    def _send_command_with_response(self, command: str, timeout: float = 2.0) -> Optional[str]:
        """
        Send command to PTU and return the response string (not just Done/OK).
        
        Args:
            command: Command string to send (e.g., "H10E")
            timeout: Timeout in seconds
            
        Returns:
            Response string from PTU, or None on failure
        """
        if not self.is_connected or not self.serial_port:
            return None
        
        try:
            # Ensure command ends with 'E'
            if not command.endswith('E'):
                command += 'E'
            
            # Clear input buffer before sending
            if self.serial_port.in_waiting > 0:
                self.serial_port.reset_input_buffer()
            
            # Send command
            command_bytes = command.encode('ascii')
            bytes_written = self.serial_port.write(command_bytes)
            self.serial_port.flush()
            logger.debug(f"Sent command ({bytes_written} bytes): {command}")
            
            # Wait for response
            start_time = time.time()
            response = ""
            time.sleep(0.1)  # Give PTU time to respond
            
            while (time.time() - start_time) < timeout:
                if self.serial_port.in_waiting > 0:
                    chunk = self.serial_port.read(self.serial_port.in_waiting).decode('ascii', errors='ignore')
                    response += chunk
                    logger.debug(f"Received chunk: {repr(chunk)}")
                    
                    # H10 and H20 typically return position values, not "Done"
                    # Response might be: "45" or "45\r\n" or "45 Done" etc.
                    # Try to extract numeric value
                    if response.strip():
                        # Give a bit more time for complete response
                        time.sleep(0.05)
                        # Check if more data is coming
                        if self.serial_port.in_waiting == 0:
                            break
                else:
                    time.sleep(0.05)  # Check every 50ms
            
            # Read any remaining data
            if self.serial_port.in_waiting > 0:
                remaining = self.serial_port.read(self.serial_port.in_waiting).decode('ascii', errors='ignore')
                response += remaining
                logger.debug(f"Received remaining: {repr(remaining)}")
            
            if response.strip():
                logger.debug(f"Full response for {command}: {repr(response)}")
                return response.strip()
            else:
                logger.warning(f"No response received for command {command}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending command {command}: {str(e)}")
            return None
    
    def _parse_position_response(self, response: str, position_type: str) -> float:
        """
        Parse position value from PTU response (H10 or H20).
        PTU returns pulse values that need to be converted to degrees.
        
        Args:
            response: Response string from PTU (contains pulse value)
            position_type: "azimuth" or "pitch" (for logging)
            
        Returns:
            Position value in degrees (float) - converted from pulses
        """
        try:
            # Clean the response - remove "Done", "OK", whitespace, newlines
            cleaned = response.strip()
            
            # Remove common response words
            for word in ['Done', 'DONE', 'done', 'OK', 'ok', '\r', '\n']:
                cleaned = cleaned.replace(word, ' ')
            
            # Extract first numeric value (this is the pulse value)
            match = re.search(r'-?\d+\.?\d*', cleaned)
            if match:
                pulse_value = float(match.group())
                
                # Convert pulse to degrees using PULSE_TO_DEGREE constant
                angle_degrees = pulse_value * PULSE_TO_DEGREE
                
                logger.debug(f"Parsed {position_type} from response '{response}': "
                           f"pulse={pulse_value} -> angle={angle_degrees:.4f}°")
                return angle_degrees
            else:
                logger.warning(f"Could not parse {position_type} from response: {repr(response)}")
                # Return cached value on parse failure
                if position_type == "azimuth":
                    return self.current_azimuth
                else:
                    return self.current_pitch
                    
        except Exception as e:
            logger.error(f"Error parsing {position_type} from response '{response}': {str(e)}")
            # Return cached value on error
            if position_type == "azimuth":
                return self.current_azimuth
            else:
                return self.current_pitch
    
    def _handle_send_raw_command(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle raw command sending without waiting for Done response."""
        if not self.is_connected:
            return {'success': False, 'error': 'PTU not connected'}
        
        command = args.get('command', '')
        if not command:
            return {'success': False, 'error': 'No command provided'}
        
        # Send command with waiting for Done response
        success = self._send_command(command, wait_for_done=True, timeout=5.0)
        return {'success': success, 'command': command}
    
    def _send_command(self, command: str, wait_for_done: bool = True, timeout: float = 2.0) -> bool:
        """
        Send command to PTU and optionally wait for 'Done' response.
        
        Args:
            command: Command string to send
            wait_for_done: If True, wait for 'Done' response
            timeout: Timeout in seconds for waiting for response
            
        Returns:
            True if command sent successfully (and 'Done' received if wait_for_done=True)
        """
        if not self.is_connected or not self.serial_port:
            return False
        
        try:
            # Ensure command ends with 'E'
            if not command.endswith('E'):
                command += 'E'
            
            # Clear input buffer before sending
            if self.serial_port.in_waiting > 0:
                self.serial_port.reset_input_buffer()
            
            # Send command - PTU expects commands ending with 'E' only (no \r\n needed)
            # According to manual, commands end with 'E' (e.g., "H12,45,30,20E")
            command_bytes = command.encode('ascii')
            bytes_written = self.serial_port.write(command_bytes)
            self.serial_port.flush()
            logger.info(f"Sent command ({bytes_written} bytes): {command} (hex: {command_bytes.hex()})")
            
            # Add to command history
            with self.history_lock:
                history_entry = {
                    'command': command,
                    'timestamp': time.time(),
                    'status': 'sent',
                    'direction': 'TX'  # Transmit
                }
                self.command_history.append(history_entry)
                # Keep only last 1000 commands
                if len(self.command_history) > 1000:
                    self.command_history.pop(0)
            
            # Notify UI of sent command
            if self.history_callback:
                try:
                    self.history_callback(command, 'sent', '')
                except:
                    pass
            
            # Wait for 'Done' response if requested
            if wait_for_done:
                start_time = time.time()
                response = ""
                # Give PTU a small delay to start responding
                time.sleep(0.1)
                
                while (time.time() - start_time) < timeout:
                    if self.serial_port.in_waiting > 0:
                        chunk = self.serial_port.read(self.serial_port.in_waiting).decode('ascii', errors='ignore')
                        response += chunk
                        logger.debug(f"Received chunk: {repr(chunk)}")
                        # Check for Done (case insensitive, might have whitespace or newlines)
                        response_upper = response.upper().strip()
                        if 'DONE' in response_upper or 'OK' in response_upper:
                            logger.info(f"Received response: {response.strip()}")
                            # Update command history with response
                            with self.history_lock:
                                if self.command_history:
                                    self.command_history[-1]['status'] = 'done'
                                    self.command_history[-1]['response'] = response.strip()
                            
                            # Notify UI of received response
                            if self.history_callback:
                                try:
                                    self.history_callback(response.strip(), 'done', response.strip())
                                except:
                                    pass
                            return True
                    time.sleep(0.05)  # Check every 50ms
                
                # If we got here, we didn't receive 'Done' in time
                if response.strip():
                    logger.warning(f"Timeout waiting for 'Done' response. Received: {repr(response)}")
                    # Log any response received (even if not "Done")
                    if self.history_callback:
                        try:
                            self.history_callback(response.strip(), 'response', response.strip())
                        except:
                            pass
                else:
                    logger.warning(f"Timeout waiting for 'Done' response. No response received.")
                    # Check if there's any data in the buffer that we might have missed
                    if self.serial_port.in_waiting > 0:
                        remaining = self.serial_port.read(self.serial_port.in_waiting).decode('ascii', errors='ignore')
                        logger.info(f"Found remaining data in buffer: {repr(remaining)}")
                        response += remaining
                        # Log any data found in buffer
                        if self.history_callback and remaining.strip():
                            try:
                                self.history_callback(remaining.strip(), 'response', remaining.strip())
                            except:
                                pass
                
                # For H12 movement commands, the PTU might execute without sending Done
                # According to manual, PTU should send Done, but some models/firmware versions might not
                # If we got any response (even if not "Done"), assume command was received
                if 'H12' in command:
                    if response.strip():
                        logger.info(f"Command sent, received response (not 'Done'): {repr(response)}. Assuming success.")
                    else:
                        logger.info("Command sent, no response received. PTU may execute command without response.")
                    # Assume success for movement commands - PTU will execute them
                    with self.history_lock:
                        if self.command_history:
                            self.command_history[-1]['status'] = 'done'
                            self.command_history[-1]['response'] = response.strip() if response.strip() else 'no_response_assumed_success'
                    return True
                
                # Update command history with timeout for non-movement commands
                with self.history_lock:
                    if self.command_history:
                        self.command_history[-1]['status'] = 'timeout'
                        self.command_history[-1]['response'] = response.strip() if response else 'timeout'
                return False
            
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
    PTU Gimbal Control Class
    
    Wrapper class that uses PTUControlThread for non-blocking serial communication.
    All serial operations are performed in a background thread to prevent UI blocking.
    
    Based on PTU manual:
    - Default baud rate: 9600
    - Commands end with 'E' (e.g., "H12,45,30,20E")
    - Returns "Done" after each command
    - Safety limits must be set on connection
    """
    
    # Note: Limit constants are now defined at module level (see top of file)
    # This ensures DRY principle - single source of truth for all limit values
    
    def __init__(self):
        """
        Initialize PTU control.
        """
        # Create and start the control thread
        self.thread = PTUControlThread()
        self.thread.start()
        
        # Cached position (updated from thread responses)
        self.current_azimuth = 0.0
        self.current_pitch = 0.0
        
        # Set limits (using module-level constants - DRY)
        self.azimuth_min = PTU_AZIMUTH_MIN
        self.azimuth_max = PTU_AZIMUTH_MAX
        self.pitch_min = PTU_PITCH_MIN
        self.pitch_max = PTU_PITCH_MAX
        
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
    
    def set_acceleration(self, acceleration: int) -> bool:
        """
        Set movement acceleration (for subsequent commands).
        
        Args:
            acceleration: Acceleration (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        if not self.thread or not self.thread.is_running:
            return False
        
        result = self.thread.send_command(
            PTUCommandType.SET_ACCELERATION,
            {'acceleration': acceleration},
            timeout=0.5
        )
        return result.get('success', False)
    
    def set_history_callback(self, callback):
        """
        Set callback function for command history updates.
        
        Args:
            callback: Function that takes (command, status, response) parameters
        """
        if self.thread:
            self.thread.history_callback = callback
    
    def go_to_zero(self, speed: int = 20) -> bool:
        """
        Move to zero position (0, 0) (non-blocking, uses thread).
        
        Args:
            speed: Movement speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        return self.move_to_position(0.0, 0.0, speed)
    
    def move_directional(self, direction: str, speed: int = 20) -> bool:
        """
        Move PTU in a direction using H61/H62/H63/H64 commands (non-blocking, uses thread).
        
        Args:
            direction: Direction ('left', 'right', 'up', 'down')
            speed: Movement speed (0-100, percentage)
        
        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            logger.error("PTU not connected")
            return False
        
        result = self.thread.send_command(
            PTUCommandType.MOVE_DIRECTIONAL,
            {'direction': direction, 'speed': speed},
            timeout=1.0
        )
        return result.get('success', False)
    
    def send_raw_command(self, command: str) -> bool:
        """
        Send raw command to PTU without waiting for 'Done' response (non-blocking, uses thread).
        
        Args:
            command: Raw command string to send (e.g., "H12,45,30,20E")
        
        Returns:
            True if command sent successfully
        """
        if not self.is_connected:
            logger.error("PTU not connected")
            return False
        
        result = self.thread.send_command(
            PTUCommandType.SEND_RAW_COMMAND,
            {'command': command},
            timeout=0.5
        )
        return result.get('success', False)
    
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
    
    def get_command_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get command history from PTU thread.
        
        Args:
            limit: Maximum number of commands to return
            
        Returns:
            List of command dictionaries with 'command', 'timestamp', 'status', and optional 'response'
        """
        if not self.thread or not self.thread.is_running:
            return []
        
        with self.thread.history_lock:
            # Return last N commands
            return self.thread.command_history[-limit:] if hasattr(self.thread, 'command_history') else []
    
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
    