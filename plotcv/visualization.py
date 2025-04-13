"""
Module de visualisation pour l'affichage de caméras et points 3D
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from plotcv.projection import add_projected_point, project_point_to_camera


def add_camera_to_plot(fig, camera, axis_length=0.3, show_in_legend=True, row=1, col=1):
    """
    Add a camera or coordinate frame to the plot
    
    Parameters:
    fig - Plotly figure to add the camera/frame to
    camera - Dictionary from create_camera_frame or create_frame
    axis_length - Length of the axis arrows
    show_in_legend - Whether to show elements in the legend
    row, col - Subplot location
    
    Returns:
    Updated figure
    """
    color = camera['color']
    name = camera['name']
    is_camera = camera.get('is_camera', True)  # Default to True for backward compatibility
    
    # Add origin
    fig.add_trace(go.Scatter3d(
        x=[camera['origin'][0]], 
        y=[camera['origin'][1]], 
        z=[camera['origin'][2]],
        mode='markers',
        marker=dict(color=color, size=5, symbol='diamond'),
        name=f'{name} Origin',
        showlegend=show_in_legend
    ), row=row, col=col)
    
    # Add axes for both cameras and frames
    origin = camera['origin']
    
    # Z axis - blue
    z_end = origin + axis_length * camera['z_axis']
    fig.add_trace(go.Scatter3d(
        x=[origin[0], z_end[0]],
        y=[origin[1], z_end[1]],
        z=[origin[2], z_end[2]],
        mode='lines',
        line=dict(color='blue', width=4),
        name=f'{name} Z Axis',
        showlegend=False
    ), row=row, col=col)
    
    # X axis - red
    x_end = origin + axis_length * camera['x_axis']
    fig.add_trace(go.Scatter3d(
        x=[origin[0], x_end[0]],
        y=[origin[1], x_end[1]],
        z=[origin[2], x_end[2]],
        mode='lines',
        line=dict(color='red', width=4),
        name=f'{name} X Axis',
        showlegend=False
    ), row=row, col=col)
    
    # Y axis - green
    y_end = origin + axis_length * camera['y_axis']
    fig.add_trace(go.Scatter3d(
        x=[origin[0], y_end[0]],
        y=[origin[1], y_end[1]],
        z=[origin[2], y_end[2]],
        mode='lines',
        line=dict(color='#00CC00', width=4),
        name=f'{name} Y Axis',
        showlegend=False
    ), row=row, col=col)
    
    # REMARQUE: Nous ne gérons plus les connexions parent ici
    # Cette fonctionnalité est maintenant entièrement gérée par add_camera_connections
    
    # Camera-specific visualizations
    if is_camera:
        # Add image plane
        corners = camera['plane_corners']
        fig.add_trace(go.Mesh3d(
            x=[c[0] for c in corners],
            y=[c[1] for c in corners],
            z=[c[2] for c in corners],
            i=[0, 0],
            j=[1, 2],
            k=[2, 3],
            opacity=0.15,
            color=color,
            flatshading=True,
            name=f'{name} Image Plane',
            showlegend=show_in_legend
        ), row=row, col=col)
        
        # Add ray from origin to plane center (optical axis)
        optical_axis = camera['plane_center']
        fig.add_trace(go.Scatter3d(
            x=[origin[0], optical_axis[0]],
            y=[origin[1], optical_axis[1]],
            z=[origin[2], optical_axis[2]],
            mode='lines',
            line=dict(color=color, dash='dash', width=2),
            name=f'{name} Optical Axis',
            showlegend=False
        ), row=row, col=col)
    
    return fig

def create_camera_view_2d(camera, points_info=None, width=400, height=400, margin=40):
    """
    Create a 2D view of what the camera sees (u,v coordinates in the image plane)
    
    Parameters:
    camera - Camera dictionary from create_camera_frame
    points_info - List of (point_3d, color, name) tuples to project
    width, height - Dimensions of the 2D plot
    margin - Margin around the plot area
    
    Returns:
    A plotly figure with the 2D view
    """
    # Get plane dimensions
    plane_half_width, plane_half_height = camera['plane_size']
    
    # Create the 2D plot
    fig_2d = go.Figure()
    
    # Add a rectangle representing the image boundaries (-w to w, -h to h)
    fig_2d.add_shape(
        type="rect",
        x0=-plane_half_width, y0=-plane_half_height,
        x1=plane_half_width, y1=plane_half_height,
        line=dict(color="black", width=2),
        fillcolor="rgba(255, 255, 255, 0.0)"
    )
    
    # Add center crosshairs
    fig_2d.add_shape(
        type="line", x0=-plane_half_width, y0=0, x1=plane_half_width, y1=0,
        line=dict(color="gray", width=1, dash="dash")
    )
    fig_2d.add_shape(
        type="line", x0=0, y0=-plane_half_height, x1=0, y1=plane_half_height,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Project and add 3D points if provided
    projected_points = []
    if points_info is not None:
        for point_info in points_info:
            point_3d = point_info[0]
            color = point_info[1] if len(point_info) > 1 else camera['color']
            name = point_info[2] if len(point_info) > 2 else "Point"
                
            # Project the 3D point onto the camera's image plane
            proj_3d, u, v = project_point_to_camera(camera, point_3d)
            
            if proj_3d is not None:
                # Add the projected point (as a cross)
                fig_2d.add_trace(go.Scatter(
                    x=[u], y=[v],
                    mode='markers+text',
                    marker=dict(color=color, size=10, symbol='x'),
                    text=[f"({u:.2f}, {v:.2f})"],
                    textposition="top center",
                    name=name,
                    showlegend=False
                ))
                
                projected_points.append((proj_3d, u, v, color, name))
    
    # Set axis properties and limits
    fig_2d.update_xaxes(
        range=[-plane_half_width * 1.2, plane_half_width * 1.2],
        zeroline=True, zerolinecolor='black',
        gridcolor='lightgray'
    )
    fig_2d.update_yaxes(
        range=[-plane_half_height * 1.2, plane_half_height * 1.2],
        zeroline=True, zerolinecolor='black',
        gridcolor='lightgray',
        scaleanchor="x",
        scaleratio=1
    )
    
    # Update layout
    fig_2d.update_layout(
        width=width, height=height,
        margin=dict(l=margin, r=margin, t=margin, b=margin),
        xaxis_title="u (horizontal)",
        yaxis_title="v (vertical)",
        plot_bgcolor="white"
    )
    
    return fig_2d, projected_points

def calculate_scene_range(cameras, points, margin=1.0):
    """Calculate appropriate ranges for the 3D scene based on cameras and points"""
    all_points = []
    
    # Toujours inclure l'origine pour s'assurer que le repère monde est visible
    all_points.append(np.array([0.0, 0.0, 0.0]))
    
    # Add camera/frame origins and other points
    for camera in cameras:
        all_points.append(camera['origin'])
        
        # Add camera-specific points if this is a camera
        if camera.get('is_camera', True):  # Default to True for backward compatibility
            all_points.append(camera['plane_center'])
            all_points.extend(camera['plane_corners'])
    
    # Add 3D points
    for point in points:
        all_points.append(point['coords'])
    
    # If we have no points, use default range
    if len(all_points) <= 1:  # Si seulement l'origine est présente
        return {'x': [-2, 2], 'y': [-2, 2], 'z': [-2, 2]}
    
    # Convert to array for easier calculation
    all_points = np.array(all_points)
    
    # Get min and max for each axis
    min_vals = np.min(all_points, axis=0) - margin
    max_vals = np.max(all_points, axis=0) + margin
    
    return {'x': [min_vals[0], max_vals[0]], 
            'y': [min_vals[1], max_vals[1]], 
            'z': [min_vals[2], max_vals[2]]}

def create_visualization_3d_2d(cameras, points_3d=None, figure_title="Visualization with 3D and 2D Views", 
                           show_world_frame=True, height=900, show_point_coordinates=True,
                           show_camera_connections=True, connection_color='black', 
                           connection_width=2, connection_style='solid', connect_to_world=True):
    """
    Create a complete visualization with 3D scene and 2D projections for each camera
    
    Parameters:
    cameras - List of dictionaries from create_camera_frame or create_frame
    points_3d - List of point dictionaries or array-like 3D points
    figure_title - Title for the figure
    show_world_frame - Whether to show the world coordinate frame
    height - Height of the figure in pixels
    show_point_coordinates - Whether to show dotted lines with local coordinates from frames to points
    show_camera_connections - Whether to show lines connecting cameras to their parents
    connection_color - Color of the connection lines between cameras
    connection_width - Width of the connection lines
    connection_style - Style of the connection lines ('solid', 'dash', 'dot', etc.)
    connect_to_world - Whether to connect cameras without a parent to the world origin
    
    Returns:
    A plotly figure with the complete visualization
    """
    # Validate that cameras is a list
    if cameras is None:
        cameras = []
    elif not isinstance(cameras, list):
        cameras = [cameras]  # Convert to list if single element
    
    # Validate inputs and prepare points list
    if points_3d is None:
        points_3d = []
    elif not isinstance(points_3d, list):
        # Single point array -> convert to list
        points_3d = [points_3d]
    
    # Convert simple points to list of dicts with default values
    processed_points = []
    for point in points_3d:
        if isinstance(point, (list, tuple, np.ndarray)):
            # Simple point coordinates -> create dict with defaults
            processed_points.append({
                'coords': np.array(point),
                'color': 'red',
                'name': 'Point',
                'size': 6
            })
        elif isinstance(point, dict):
            # Dict format -> ensure all required fields
            pt_dict = {
                'coords': np.array(point.get('coords', [0, 0, 0])),
                'color': point.get('color', 'red'),
                'name': point.get('name', 'Point'),
                'size': point.get('size', 6)
            }
            processed_points.append(pt_dict)
    
    # S'assurer que la propriété show_coordinates est correctement définie pour tous les frames
    for camera in cameras:
        # Utiliser la valeur globale de show_point_coordinates si aucune valeur n'est spécifiée
        if 'show_coordinates' not in camera:
            camera['show_coordinates'] = show_point_coordinates
            
    # Separate actual cameras from regular frames for proper visualization
    actual_cameras = [cam for cam in cameras if cam.get('is_camera', True)]  # Default to True for backward compatibility
    ncameras = len(actual_cameras)
    
    # Create figure with appropriate subplots
    if ncameras == 0:
        # No cameras, just a 3D plot for the frames and points
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
    else:
        # Create layout for cameras (1 row for 3D, 1 row for 2D views)
        subplot_titles = ["3D Scene"]
        for i, cam in enumerate(actual_cameras):
            subplot_titles.append(f"{cam['name']} View")
        
        # First row: 3D scene spanning all columns
        # Second row: a 2D view for each camera
        specs = [
            [{"type": "scene", "colspan": ncameras}] + [None] * (ncameras - 1),
        ]
        # Add second row with 'xy' cell per camera
        specs.append([{"type": "xy"} for _ in range(ncameras)])
        
        fig = make_subplots(
            rows=2, 
            cols=max(1, ncameras),  # Always at least one column
            specs=specs,
            subplot_titles=subplot_titles,
            vertical_spacing=0.1
        )
    
    # Add world coordinate frame
    if show_world_frame:
        world_axis_length = 2.0  # For better visibility
        world_origin = np.array([0.0, 0.0, 0.0])
        
        # Point at origin to make it more visible
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(color='black', size=8, symbol='circle'),
            name='Origin',
            showlegend=True
        ), row=1, col=1)
        
        # Axes with arrows for better visualization
        fig.add_trace(go.Scatter3d(
            x=[0, world_axis_length], y=[0, 0], z=[0, 0],
            mode='lines', 
            line=dict(color='red', width=8), 
            name='World X Axis'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, world_axis_length], z=[0, 0],
            mode='lines', 
            line=dict(color='green', width=8),
            name='World Y Axis'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, world_axis_length],
            mode='lines', 
            line=dict(color='blue', width=8),
            name='World Z Axis'
        ), row=1, col=1)
    
    # Add points to 3D plot
    for point in processed_points:
        fig.add_trace(go.Scatter3d(
            x=[point['coords'][0]],
            y=[point['coords'][1]],
            z=[point['coords'][2]],
            mode='markers',
            marker=dict(color=point['color'], size=point['size'], symbol='circle'),
            name=point['name'],
            showlegend=True
        ), row=1, col=1)
    
    # Add all frames (cameras and regular frames) to 3D plot
    for frame in cameras:
        add_camera_to_plot(fig, frame, row=1, col=1)
    
    # Add camera connections if requested
    if show_camera_connections:
        add_camera_connections(fig, cameras, row=1, col=1, 
                               line_color=connection_color, 
                               line_width=connection_width, 
                               line_dash=connection_style,
                               connect_to_world=connect_to_world)
    
    # Add dotted lines with coordinates between frames and points
    if len(processed_points) > 0:  # Always call if we have points, function will check per-camera settings
        add_point_frame_connections(fig, cameras, processed_points, row=1, col=1)
    
    # Project points only to actual cameras and add to 3D plot
    all_projections = {}
    if ncameras > 0 and len(processed_points) > 0:
        for i, camera in enumerate(actual_cameras):
            camera_projections = []
            for point in processed_points:
                result, proj_info = add_projected_point(
                    fig, camera, point['coords'], 
                    point_color=point['color'], 
                    name=f"{point['name']} ({camera['name']})",
                    row=1, col=1
                )
                if proj_info[0] is not None:  # If projection exists
                    camera_projections.append((
                        point['coords'], point['color'], 
                        f"{point['name']} ({camera['name']})"
                    ))
            all_projections[camera['name']] = camera_projections
    
    # Create and add 2D views only for actual cameras
    if ncameras > 0:
        for i, camera in enumerate(actual_cameras):
            # Create 2D view even if no points are projected
            cam_points = all_projections.get(camera['name'], [])
            cam_2d_fig, _ = create_camera_view_2d(camera, cam_points)
            
            # Add traces from 2D figure to subplot
            for trace in cam_2d_fig.data:
                fig.add_trace(trace, row=2, col=i+1)
            
            # Copy shapes from 2D figure to subplot
            for shape in cam_2d_fig.layout.shapes:
                fig.add_shape(shape, row=2, col=i+1)
            
            # Configure axes for this subplot
            fig.update_xaxes(
                title_text="u (horizontal)", 
                row=2, col=i+1
            )
            fig.update_yaxes(
                title_text="v (vertical)", 
                scaleanchor="x", 
                scaleratio=1, 
                row=2, col=i+1
            )
    
    # Configure 3D layout with appropriate range
    scene_ranges = calculate_scene_range(cameras, processed_points)
    fig.update_layout(
        title=figure_title,
        scene=dict(
            xaxis=dict(title='X', range=scene_ranges['x'], backgroundcolor="rgb(230, 230, 230)", gridcolor='white'),
            yaxis=dict(title='Y', range=scene_ranges['y'], backgroundcolor="rgb(230, 230, 230)", gridcolor='white'),
            zaxis=dict(title='Z', range=scene_ranges['z'], backgroundcolor="rgb(230, 230, 230)", gridcolor='white'),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        height=height
    )
    
    return fig

def add_camera_connections(fig, cameras, row=1, col=1):
    """
    Add connections between cameras based on parent-child relationships
    
    Parameters:
    fig - Plotly figure to add the connections to
    cameras - List of camera dictionaries
    row, col - Subplot location
    """
    for camera in cameras:
        # Skip if this camera has connect_to_parent disabled
        if not camera.get('connect_to_parent', True):
            continue
            
        camera_id = camera.get('camera_id')
        parent_id = camera.get('parent_id')
        
        # Skip if there's no parent ID
        if parent_id is None or parent_id == '':
            continue
        
        # Find the parent camera
        parent_camera = None
        for cam in cameras:
            if cam.get('camera_id') == parent_id:
                parent_camera = cam
                break
                
        if parent_camera is not None:
            # Get positions
            cam_origin = camera['origin']
            parent_origin = parent_camera['origin']
            
            # Add connection line
            fig.add_trace(go.Scatter3d(
                x=[cam_origin[0], parent_origin[0]],
                y=[cam_origin[1], parent_origin[1]],
                z=[cam_origin[2], parent_origin[2]],
                mode='lines',
                line=dict(color='purple', width=3),
                name=f"Connection {camera_id} -> {parent_id}",
                showlegend=False
            ), row=row, col=col)

def add_point_frame_connections(fig, frames, points, row=1, col=1, line_color='black', line_width=2, line_dash='10px 15px'):
    """
    Add connecting lines between frames and 3D points, with local coordinates displayed
    
    Parameters:
    fig - Plotly figure to add the connections to
    frames - List of frame dictionaries 
    points - List of processed point dictionaries
    row, col - Subplot location
    line_color - Color of the connecting lines
    line_width - Width of the connecting lines
    line_dash - Style of the connecting lines
    """
    # Process each frame
    for frame in frames:
        # Check if this frame should show coordinates
        if not frame.get('show_coordinates', True):
            continue
            
        frame_origin = frame['origin']
        frame_name = frame.get('name', 'Frame')
        
        # For each point, calculate local coordinates relative to the frame
        for point in points:
            point_coords = point['coords']
            point_name = point.get('name', 'Point')
            
            # Create a line connecting the frame origin to the point
            fig.add_trace(go.Scatter3d(
                x=[frame_origin[0], point_coords[0]],
                y=[frame_origin[1], point_coords[1]],
                z=[frame_origin[2], point_coords[2]],
                mode='lines',
                line=dict(color=line_color, width=line_width, dash=line_dash),
                name=f"{frame_name} -> {point_name}",
                showlegend=False
            ), row=row, col=col)
            
            # Calculate the coordinates of the point in the frame's coordinate system
            # Transform from world coordinates to frame coordinates
            x_axis = frame['x_axis']
            y_axis = frame['y_axis']
            z_axis = frame['z_axis']
            
            # Vector from frame origin to point
            vec = point_coords - frame_origin
            
            # Project onto frame axes
            x_local = np.dot(vec, x_axis)
            y_local = np.dot(vec, y_axis)
            z_local = np.dot(vec, z_axis)
            
            # Calculate middle point for text placement
            mid_point = frame_origin + 0.5 * vec
            
            # Add annotation with local coordinates
            fig.add_trace(go.Scatter3d(
                x=[mid_point[0]],
                y=[mid_point[1]],
                z=[mid_point[2]],
                mode='text',
                text=[f"({x_local:.2f}, {y_local:.2f}, {z_local:.2f})"],
                textposition="middle center",
                name=f"{frame_name} -> {point_name} coords",
                showlegend=False
            ), row=row, col=col)
    
    return fig