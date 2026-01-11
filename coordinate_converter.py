"""
Coordinate conversion from pixel positions to PTU gimbal angles.
Converts camera pixel coordinates to azimuth and pitch angles.
"""
import numpy as np
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class CoordinateConverter:
    """
    Converts pixel coordinates to PTU gimbal angles.
    
    Assumes:
    - Camera is mounted on PTU gimbal
    - Camera optical axis is aligned with gimbal center
    - Field of view (FOV) is known or calibrated
    """
    
    def __init__(self, 
                 image_width: int = 1920,
                 image_height: int = 1080,
                 horizontal_fov: float = 60.0,  # degrees
                 vertical_fov: float = 45.0):  # degrees
        """
        Initialize coordinate converter.
        
        Args:
            image_width: Camera image width in pixels
            image_height: Camera image height in pixels
            horizontal_fov: Horizontal field of view in degrees
            vertical_fov: Vertical field of view in degrees
        """
        self.image_width = image_width
        self.image_height = image_height
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        
        # Calculate pixel-to-degree conversion factors
        self.pixels_per_degree_h = image_width / horizontal_fov
        self.pixels_per_degree_v = image_height / vertical_fov
        
        # Center of image (optical axis)
        self.center_x = image_width / 2.0
        self.center_y = image_height / 2.0
    
    def pixel_to_angle(self, pixel_x: float, pixel_y: float,
                      current_azimuth: float = 0.0,
                      current_pitch: float = 0.0) -> Tuple[float, float]:
        """
        Convert pixel coordinates to gimbal angles.
        
        Args:
            pixel_x: X coordinate in pixels (0 = left, image_width = right)
            pixel_y: Y coordinate in pixels (0 = top, image_height = bottom)
            current_azimuth: Current gimbal azimuth angle (for relative calculation)
            current_pitch: Current gimbal pitch angle (for relative calculation)
        
        Returns:
            (azimuth, pitch) tuple in degrees
        """
        # Calculate offset from center
        offset_x = pixel_x - self.center_x
        offset_y = pixel_y - self.center_y
        
        # Convert pixel offset to angle offset
        angle_offset_x = offset_x / self.pixels_per_degree_h
        angle_offset_y = offset_y / self.pixels_per_degree_v
        
        # Calculate new angles (relative to current position)
        # Positive X offset (right) = positive azimuth (right)
        # Positive Y offset (down) = negative pitch (down)
        new_azimuth = current_azimuth + angle_offset_x
        new_pitch = current_pitch - angle_offset_y  # Negative because pitch up is positive
        
        return new_azimuth, new_pitch
    
    def angle_to_pixel(self, azimuth: float, pitch: float,
                      current_azimuth: float = 0.0,
                      current_pitch: float = 0.0) -> Tuple[float, float]:
        """
        Convert gimbal angles to pixel coordinates.
        
        Args:
            azimuth: Azimuth angle in degrees
            pitch: Pitch angle in degrees
            current_azimuth: Current gimbal azimuth angle
            current_pitch: Current gimbal pitch angle
        
        Returns:
            (pixel_x, pixel_y) tuple
        """
        # Calculate angle offset
        angle_offset_x = azimuth - current_azimuth
        angle_offset_y = -(pitch - current_pitch)  # Negative because pitch up is positive
        
        # Convert angle offset to pixel offset
        offset_x = angle_offset_x * self.pixels_per_degree_h
        offset_y = angle_offset_y * self.pixels_per_degree_v
        
        # Calculate pixel coordinates
        pixel_x = self.center_x + offset_x
        pixel_y = self.center_y + offset_y
        
        return pixel_x, pixel_y
    
    def update_image_size(self, width: int, height: int):
        """
        Update image dimensions (e.g., when camera resolution changes).
        
        Args:
            width: New image width
            height: New image height
        """
        self.image_width = width
        self.image_height = height
        self.center_x = width / 2.0
        self.center_y = height / 2.0
        self.pixels_per_degree_h = width / self.horizontal_fov
        self.pixels_per_degree_v = height / self.vertical_fov
    
    def set_fov(self, horizontal_fov: float, vertical_fov: float):
        """
        Update field of view values.
        
        Args:
            horizontal_fov: Horizontal FOV in degrees
            vertical_fov: Vertical FOV in degrees
        """
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.pixels_per_degree_h = self.image_width / horizontal_fov
        self.pixels_per_degree_v = self.image_height / vertical_fov
