"""
Module de gestion des caméras pour la visualisation 3D
"""

import numpy as np


def rotation_matrix_from_angles(rx, ry, rz):
    """
    Create a 3D rotation matrix from Euler angles rx, ry, rz (in radians)
    Uses the ZYX convention (first Z, then Y, then X)
    
    Parameters:
    rx, ry, rz - Rotation angles in radians around X, Y, Z axes
    
    Returns:
    3x3 rotation matrix
    """
    # Rotation around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rx), -np.sin(rx)],
        [0, np.sin(rx), np.cos(rx)]
    ])
    
    # Rotation around Y axis
    Ry = np.array([
        [np.cos(ry), 0, np.sin(ry)],
        [0, 1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])
    
    # Rotation around Z axis
    Rz = np.array([
        [np.cos(rz), -np.sin(rz), 0],
        [np.sin(rz), np.cos(rz), 0],
        [0, 0, 1]
    ])
    
    # Combine rotations - ZYX convention
    return Rz @ Ry @ Rx

def create_frame(rotation_angles, translation, focal_length=None, plane_size=(0.6, 0.6), 
              optical_center=None, color='blue', name='Frame', parent=None, is_camera=False, 
              show_coordinates=True, connect_to_parent=False):
    """
    Create a general coordinate frame or a camera frame with given parameters
    
    Parameters:
    rotation_angles - tuple of (rx, ry, rz) in degrees
    translation - tuple/array of (tx, ty, tz) frame position
    focal_length - float or tuple (fx, fy), only used if is_camera=True
    plane_size - tuple of (half_width, half_height) for the image plane, only used if is_camera=True
    optical_center - tuple (cx, cy) for the optical center, only used if is_camera=True
    color - color for the visualization
    name - name of the frame
    parent - parent frame to define transformation relative to
    is_camera - whether this is a camera frame (True) or just a coordinate frame (False)
    show_coordinates - whether to show coordinates for points relative to this frame
    connect_to_parent - whether to connect this frame to its parent with a line in visualization
    
    Returns:
    Dictionary containing all frame parameters
    """
    # Convert angles to radians
    rx, ry, rz = [np.deg2rad(angle) for angle in rotation_angles]
    
    # Create rotation matrix
    local_R = rotation_matrix_from_angles(rx, ry, rz)
    
    # Handle parent transformations if provided
    if parent is not None:
        # Get parent's rotation and translation
        parent_R = np.vstack([
            np.hstack([parent['x_axis'].reshape(-1, 1), parent['y_axis'].reshape(-1, 1), parent['z_axis'].reshape(-1, 1)])
        ])
        parent_T = parent['origin']
        
        # Combine parent and local transformations
        R = parent_R @ local_R  # Combined rotation
        T = parent_T + parent_R @ np.array(translation)  # Translation in world coordinates
    else:
        # No parent - use local transformation directly
        R = local_R
        T = np.array(translation)
    
    # Camera axes in world coordinates
    x_axis = R @ np.array([1, 0, 0])
    y_axis = R @ np.array([0, 1, 0])
    z_axis = R @ np.array([0, 0, 1])
    
    # Base frame dictionary with common properties
    frame_dict = {
        'origin': T,
        'x_axis': x_axis,
        'y_axis': y_axis,
        'z_axis': z_axis,
        'color': color,
        'name': name,
        'parent': parent['name'] if parent is not None else None,
        'parent_id': parent.get('id', parent['name']) if parent is not None else None,
        'id': name,  # Use name as default ID
        'is_camera': is_camera,  # Store whether this is a camera frame
        'show_coordinates': show_coordinates,  # Store whether to show coordinates for this frame
        'connect_to_parent': connect_to_parent  # Store whether to connect to parent
    }
    
    # Add camera-specific properties if needed
    if is_camera:
        if focal_length is None:
            focal_length = 1.0  # Default focal length
            
        # Handle focal length (single value or separate fx, fy)
        if isinstance(focal_length, (int, float)):
            fx = fy = focal_length
        else:
            fx, fy = focal_length
            
        # Set optical center (cx, cy) - default is (0,0) which is the center of the image plane
        if optical_center is None:
            cx, cy = 0, 0  # Center of the image plane
        else:
            cx, cy = optical_center
            
        # View direction is along frame's Z axis
        view_dir = z_axis
        
        # Set plane center and camera origin
        # The image plane is positioned at camera origin + fx*view_dir
        base_plane_center = T + fx * view_dir
        
        # Apply optical center offset to the plane center
        # Since cx, cy are now absolute values, use them directly
        optical_offset_x = cx * x_axis 
        optical_offset_y = cy * y_axis
        
        # The plane center is offset from the base center
        plane_center = base_plane_center + optical_offset_x + optical_offset_y
        
        # Plane dimensions
        plane_half_width, plane_half_height = plane_size
        
        # Compute the corners of the image plane around the offset plane center
        u_axis = x_axis  # U axis aligns with frame X
        v_axis = y_axis  # V axis aligns with frame Y
        
        x_vec = u_axis * plane_half_width
        y_vec = v_axis * plane_half_height
        
        # Calculate the corners of the image plane
        corners = [
            plane_center - x_vec - y_vec,
            plane_center + x_vec - y_vec,
            plane_center + x_vec + y_vec,
            plane_center - x_vec + y_vec,
        ]
        
        # Add camera-specific properties to the dictionary
        frame_dict.update({
            'view_dir': view_dir,
            'plane_center': plane_center,
            'optical_center': (cx, cy),
            'plane_normal': view_dir,  # Normal to image plane is same as view direction
            'plane_u_axis': u_axis,
            'plane_v_axis': v_axis,
            'plane_x_vec': x_vec,
            'plane_y_vec': y_vec,
            'plane_corners': corners,
            'focal_length': (fx, fy),
            'plane_size': plane_size,
        })
    
    return frame_dict

# Keep create_camera_frame for backward compatibility
def create_camera_frame(rotation_angles, translation, focal_length, plane_size=(0.6, 0.6), 
                   optical_center=None, color='blue', name='Camera', parent=None, 
                   show_coordinates=True, connect_to_parent=False):
    """
    Create a camera frame with given extrinsic and intrinsic parameters
    (This function is kept for backward compatibility)
    
    Parameters:
    rotation_angles - tuple of (rx, ry, rz) in degrees
    translation - tuple/array of (tx, ty, tz) frame position
    focal_length - float or tuple (fx, fy) 
    plane_size - tuple of (half_width, half_height) for the image plane
    optical_center - tuple (cx, cy) for the optical center
    color - color for visualization
    name - name of the camera
    parent - parent frame to define transformation relative to
    show_coordinates - whether to show coordinates for points relative to this frame
    connect_to_parent - whether to visualize a line connecting to the parent frame
    
    Returns:
    Dictionary containing camera parameters
    """
    return create_frame(
        rotation_angles=rotation_angles,
        translation=translation,
        focal_length=focal_length,
        plane_size=plane_size,
        optical_center=optical_center,
        color=color,
        name=name,
        parent=parent,
        is_camera=True,
        show_coordinates=show_coordinates,
        connect_to_parent=connect_to_parent
    )

def get_point_in_frame_coordinates(frame, point_3d):
    """
    Convert a 3D point from world coordinates to frame local coordinates
    
    Parameters:
    frame - Frame dictionary from create_frame
    point_3d - 3D point in world coordinates (numpy array or list)
    
    Returns:
    3D point in frame local coordinates
    """
    # Ensure point is a numpy array
    point_3d = np.array(point_3d)
    
    # Get the frame's origin and axes
    origin = frame['origin']
    
    # Create rotation matrix from frame axes
    R = np.vstack([
        frame['x_axis'].reshape(1, 3),
        frame['y_axis'].reshape(1, 3),
        frame['z_axis'].reshape(1, 3)
    ])
    
    # Calculate point relative to frame origin
    point_rel = point_3d - origin
    
    # Transform point to frame coordinates
    local_coords = np.matmul(R, point_rel)
    
    return local_coords