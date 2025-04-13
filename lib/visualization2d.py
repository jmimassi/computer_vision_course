from math import cos, radians, sin

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from IPython.display import display


def create_frame_2d(fig, origin_x=0, origin_y=0, rotation_deg=0, frame_name="", 
                   frame_index=0, show_labels=True, show_grid=True):
    """
    Create a 2D frame with origin at (origin_x, origin_y) and rotated by rotation_deg degrees.
    
    Args:
        fig: Plotly figure to add the frame to
        origin_x, origin_y: Origin of the frame
        rotation_deg: Rotation of the frame in degrees
        frame_name: Prefix for axis labels
        frame_index: Index of the frame (for reference)
        show_labels: Whether to show axis labels
        show_grid: Whether to show grid lines
    """
    # Convert rotation from degrees to radians
    rotation_rad = radians(rotation_deg)
    
    # Calculate unit vectors for the rotated frame
    x_unit = [cos(rotation_rad), sin(rotation_rad)]
    y_unit = [-sin(rotation_rad), cos(rotation_rad)]
    
    # X axis with arrow (red)
    fig.add_trace(go.Scatter(
        x=[origin_x, origin_x + x_unit[0]], 
        y=[origin_y, origin_y + x_unit[1]],
        mode='lines',
        line=dict(color='red', width=2),
        name=f'{frame_name}X Axis {frame_index}'
    ))

    # Add arrow for X axis
    fig.add_annotation(
        x=origin_x + x_unit[0], 
        y=origin_y + x_unit[1],
        ax=origin_x + 0.8*x_unit[0], 
        ay=origin_y + 0.8*x_unit[1],
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='red'
    )

    # Y axis with arrow (green)
    fig.add_trace(go.Scatter(
        x=[origin_x, origin_x + y_unit[0]], 
        y=[origin_y, origin_y + y_unit[1]],
        mode='lines',
        line=dict(color='green', width=2),
        name=f'{frame_name}Y Axis {frame_index}'
    ))

    # Add arrow for Y axis
    fig.add_annotation(
        x=origin_x + y_unit[0], 
        y=origin_y + y_unit[1],
        ax=origin_x + 0.8*y_unit[0], 
        ay=origin_y + 0.8*y_unit[1],
        xref="x", yref="y",
        axref="x", ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.5,
        arrowwidth=2,
        arrowcolor='green'
    )
    
    # Add the origin
    fig.add_trace(go.Scatter(
        x=[origin_x], y=[origin_y],
        mode='markers',
        marker=dict(color='black', size=8, symbol='circle'),
        name=f'{frame_name}Origin {frame_index}'
    ))
    
    # Add X and Y labels if requested
    if show_labels:
        # X label
        fig.add_annotation(
            x=origin_x + 1.05*x_unit[0], 
            y=origin_y + 1.05*x_unit[1],
            text=f"{frame_name}X",
            showarrow=False,
            font=dict(color='red', size=12),
            xref="x", yref="y"
        )
        
        # Y label
        fig.add_annotation(
            x=origin_x + 1.05*y_unit[0], 
            y=origin_y + 1.05*y_unit[1],
            text=f"{frame_name}Y",
            showarrow=False,
            font=dict(color='green', size=12),
            xref="x", yref="y"
        )
    
    return fig

def point_in_frame(x, y, origin_x, origin_y, rotation_deg):
    """
    Calculate the coordinates of a point in a rotated frame.
    
    Args:
        x, y: Point coordinates in the world frame
        origin_x, origin_y: Origin of the rotated frame
        rotation_deg: Rotation of the frame in degrees
        
    Returns:
        x, y: Coordinates in the rotated frame
    """
    # Convert rotation from degrees to radians
    rotation_rad = radians(rotation_deg)
    
    # Translate point to frame origin
    tx = x - origin_x
    ty = y - origin_y
    
    # Rotate point
    rx = tx * cos(rotation_rad) + ty * sin(rotation_rad)
    ry = -tx * sin(rotation_rad) + ty * cos(rotation_rad)
    
    return rx, ry

def add_point_2d(fig, x, y, label="P", color="blue", show_coords=True, 
                frame_index=0, frames=None):
    """
    Add a point to the 2D figure.
    
    Args:
        fig: Plotly figure to add the point to
        x, y: Coordinates of the point
        label: Point label
        color: Color of the point
        show_coords: Whether to show coordinates
        frame_index: Which frame to show coordinates for (0 = world frame)
        frames: List of frames [x, y, rotation_deg, name]
    """
    # Add the point
    fig.add_trace(go.Scatter(
        x=[x], y=[y],
        mode='markers+text',
        marker=dict(color=color, size=10),
        text=[label],
        textposition="top center",
        name=f"Point {label}"
    ))
    
    # If no frames are specified, assume world frame only
    if frames is None:
        frames = [[0, 0, 0, ""]]
    
    # Get frame information
    frame = frames[frame_index]
    frame_x, frame_y, frame_rotation, frame_name = frame
    
    # Calculate point coordinates in the frame
    if frame_index > 0:  # Only calculate for non-world frames
        px, py = point_in_frame(x, y, frame_x, frame_y, frame_rotation)
    else:
        px, py = x, y
        
    # Draw a dashed line from the frame origin to the point
    fig.add_trace(go.Scatter(
        x=[frame_x, x], y=[frame_y, y],
        mode='lines',
        line=dict(color=color, width=1, dash='dash'),
        name=f'Connection to {frame_name}frame {frame_index}'
    ))
    
    # Add the coordinates as an annotation directly on the line
    if show_coords:
        # Calculate the middle point of the line
        mid_x = (frame_x + x) / 2
        mid_y = (frame_y + y) / 2
        
        fig.add_annotation(
            x=mid_x,
            y=mid_y,
            text=f"({px:.1f}, {py:.1f})",
            showarrow=False,
            font=dict(color=color, size=12),
            bgcolor='rgba(255, 255, 255, 0.7)',
            bordercolor=color,
            borderwidth=1,
            borderpad=3
        )
        
    return fig

def create_2d_visualization(points=None, frames=None, show_coords=True, 
                          show_labels=True, show_grid=True,
                          width=600, height=600, 
                          range_x=[-0.5, 2], range_y=[-0.5, 1.5]):
    """
    Create a complete 2D visualization with multiple frames and points.
    
    Args:
        points: List of points in format [x, y, label, color, frame_index]
        frames: List of frames in format [x, y, rotation_deg, name]
        show_coords: Whether to show coordinates for points
        show_labels: Whether to show axis labels
        show_grid: Whether to show grid lines
        width, height: Figure dimensions
        range_x, range_y: Axis ranges
        
    Returns:
        fig: Plotly figure object
    """
    # Create a new figure
    fig = go.Figure()
    
    # If no frames are specified, create world frame only
    if frames is None:
        frames = [[0, 0, 0, ""]]
    
    # Create all frames
    for i, frame in enumerate(frames):
        x, y, rotation, name = frame
        create_frame_2d(fig, origin_x=x, origin_y=y, rotation_deg=rotation, 
                      frame_name=name, frame_index=i, show_labels=show_labels)
    
    # Add all points if specified
    if points is not None:
        for point in points:
            if len(point) >= 5:
                x, y, label, color, frame_idx = point[:5]
                add_point_2d(fig, x, y, label, color, show_coords, frame_idx, frames)
            else:
                # Default to world frame if frame_idx not specified
                x, y, label, color = point[:4]
                add_point_2d(fig, x, y, label, color, show_coords, 0, frames)
    
    # Configure the figure
    fig.update_layout(
        title="2D Reference Frames and Points",
        xaxis=dict(range=range_x, title=""),
        yaxis=dict(range=range_y, title=""),
        width=width, height=height,
        plot_bgcolor='white',
        yaxis_scaleanchor="x",  # Ensures that X and Y scales are equal
        yaxis_scaleratio=1
    )
    
    # Grid
    if show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray', 
                        zeroline=True, zerolinewidth=2, zerolinecolor='gray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray',
                        zeroline=True, zerolinewidth=2, zerolinecolor='gray')
    
    return fig

def create_interactive_2d_visualization():
    """
    Create an interactive 2D visualization with widgets to control frames and points.
    
    Returns:
        widgets and figure
    """
    # Create widgets for the first frame (always at origin)
    frame0_label = widgets.HTML(value="<b>Frame 0 (Origin):</b>")
    
    # Create widgets for the second frame
    frame1_label = widgets.HTML(value="<b>Frame 1:</b>")
    frame1_x = widgets.FloatSlider(value=0.5, min=-2.0, max=2.0, step=0.1, 
                                 description='X:')
    frame1_y = widgets.FloatSlider(value=0.0, min=-2.0, max=2.0, step=0.1,
                                 description='Y:')
    frame1_rot = widgets.FloatSlider(value=30.0, min=0.0, max=360.0, step=5.0,
                                   description='Rotation:')
    
    # Create widgets for a point
    point_label = widgets.HTML(value="<b>Point P:</b>")
    point_x = widgets.FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.1,
                                description='X:')
    point_y = widgets.FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.1,
                                description='Y:')
    point_frame = widgets.Dropdown(options=[('World Frame', 0), ('Frame 1', 1)],
                                 value=0, description='Coords in:')
    show_coords = widgets.Checkbox(value=True, description='Show coordinates')
    
    # Layout all widgets
    frame0_box = widgets.VBox([frame0_label])
    frame1_box = widgets.VBox([frame1_label, frame1_x, frame1_y, frame1_rot])
    point_box = widgets.VBox([point_label, point_x, point_y, point_frame, show_coords])
    
    controls = widgets.HBox([frame0_box, frame1_box, point_box])
    output = widgets.Output()
    
    # Create and display the interactive visualization
    def update_visualization(change=None):
        with output:
            output.clear_output(wait=True)
            
            # Define frames and points based on widget values
            frames = [
                [0, 0, 0, ""],  # Origin frame
                [frame1_x.value, frame1_y.value, frame1_rot.value, ""]  # Second frame
            ]
            
            points = [
                [point_x.value, point_y.value, "P", "blue", point_frame.value]
            ]
            
            # Create the visualization
            fig = create_2d_visualization(
                points=points,
                frames=frames,
                show_coords=show_coords.value
            )
            
            # Display the figure
            fig.show()
    
    # Connect the update function to widget changes
    frame1_x.observe(update_visualization, names='value')
    frame1_y.observe(update_visualization, names='value')
    frame1_rot.observe(update_visualization, names='value')
    point_x.observe(update_visualization, names='value')
    point_y.observe(update_visualization, names='value')
    point_frame.observe(update_visualization, names='value')
    show_coords.observe(update_visualization, names='value')
    
    # Display the widgets and initial visualization
    display(controls, output)
    update_visualization()
    
    return controls, output

def create_dual_point_interactive_visualization():
    """
    Create an interactive 2D visualization with two points.
    
    Returns:
        widgets and figure
    """
    # Create widgets for the first frame (always at origin)
    frame0_label = widgets.HTML(value="<b>Frame 0 (Origin):</b>")
    
    # Create widgets for the second frame
    frame1_label = widgets.HTML(value="<b>Frame 1:</b>")
    frame1_x = widgets.FloatSlider(value=0.5, min=-2.0, max=2.0, step=0.1, 
                                 description='X:')
    frame1_y = widgets.FloatSlider(value=0.0, min=-2.0, max=2.0, step=0.1,
                                 description='Y:')
    frame1_rot = widgets.FloatSlider(value=30.0, min=0.0, max=360.0, step=5.0,
                                   description='Rotation:')
    
    # Create widgets for point P
    pointp_label = widgets.HTML(value="<b>Point P:</b>")
    pointp_x = widgets.FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.1,
                                description='X:')
    pointp_y = widgets.FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.1,
                                description='Y:')
    pointp_frame = widgets.Dropdown(options=[('World Frame', 0), ('Frame 1', 1)],
                                  value=0, description='Coords in:')
    
    # Create widgets for point Q
    pointq_label = widgets.HTML(value="<b>Point Q:</b>")
    pointq_x = widgets.FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.1,
                                description='X:')
    pointq_y = widgets.FloatSlider(value=1.0, min=-2.0, max=2.0, step=0.1,
                                description='Y:')
    pointq_frame = widgets.Dropdown(options=[('World Frame', 0), ('Frame 1', 1)],
                                  value=1, description='Coords in:')
    
    # General controls
    show_coords = widgets.Checkbox(value=True, description='Show coordinates')
    
    # Layout all widgets
    frame0_box = widgets.VBox([frame0_label])
    frame1_box = widgets.VBox([frame1_label, frame1_x, frame1_y, frame1_rot])
    pointp_box = widgets.VBox([pointp_label, pointp_x, pointp_y, pointp_frame])
    pointq_box = widgets.VBox([pointq_label, pointq_x, pointq_y, pointq_frame, show_coords])
    
    controls = widgets.HBox([frame0_box, frame1_box, pointp_box, pointq_box])
    output = widgets.Output()
    
    # Create and display the interactive visualization
    def update_visualization(change=None):
        with output:
            output.clear_output(wait=True)
            
            # Define frames and points based on widget values
            frames = [
                [0, 0, 0, ""],  # Origin frame
                [frame1_x.value, frame1_y.value, frame1_rot.value, ""]  # Second frame
            ]
            
            points = [
                [pointp_x.value, pointp_y.value, "P", "blue", pointp_frame.value],
                [pointq_x.value, pointq_y.value, "Q", "red", pointq_frame.value]
            ]
            
            # Create the visualization
            fig = create_2d_visualization(
                points=points,
                frames=frames,
                show_coords=show_coords.value
            )
            
            # Display the figure
            fig.show()
    
    # Connect the update function to widget changes
    widgets_to_observe = [
        frame1_x, frame1_y, frame1_rot,
        pointp_x, pointp_y, pointp_frame,
        pointq_x, pointq_y, pointq_frame,
        show_coords
    ]
    
    for w in widgets_to_observe:
        w.observe(update_visualization, names='value')
    
    # Display the widgets and initial visualization
    display(controls, output)
    update_visualization()
    
    return controls, output