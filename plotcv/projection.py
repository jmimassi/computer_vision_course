"""
Module de projection pour la visualisation 3D/2D
"""

import numpy as np
import plotly.graph_objects as go


def project_point_to_camera(camera, point_3d):
    """
    Project a 3D point onto a camera's image plane
    
    Parameters:
    camera - Camera dictionary from create_camera_frame
    point_3d - 3D point coordinates
    
    Returns:
    Tuple of (projection_3d, u, v)
    projection_3d - 3D coordinates of the projection on the image plane
    u, v - 2D coordinates in the image plane coordinate system
    """
    # Calculate ray from camera origin to 3D point
    ray_dir = point_3d - camera['origin']
    
    # Calculate intersection with image plane
    plane_point = camera['plane_center']
    plane_normal = camera['plane_normal']
    
    # Ray-plane intersection formula
    denom = np.dot(ray_dir, plane_normal)
    if abs(denom) < 1e-6:  # Ray is parallel to plane
        return None, None, None
        
    t = np.dot(plane_point - camera['origin'], plane_normal) / denom
    if t <= 0:  # Point is behind the camera
        return None, None, None
        
    # 3D coordinates of projection on image plane
    proj_3d = camera['origin'] + t * ray_dir
    
    # Calculate 2D coordinates (u,v) in the image plane
    vec = proj_3d - camera['plane_center']
    u = np.dot(vec, camera['plane_u_axis'])
    v = np.dot(vec, camera['plane_v_axis'])
    
    return proj_3d, u, v

def add_projected_point(fig, camera, point_3d, point_color=None, marker_size=5, show_ray=True,
                       ray_width=2, name=None, show_in_legend=True, row=1, col=1):
    """
    Add a 3D point and its projection onto a camera's image plane to the plot
    
    Parameters:
    fig - Plotly figure to add to
    camera - Camera dictionary from create_camera_frame
    point_3d - 3D coordinates of the point
    point_color - Color of the 3D point (defaults to camera color)
    marker_size - Size of the point markers
    show_ray - Whether to show the projection ray
    ray_width - Width of the projection ray line
    name - Name prefix for legend entries (defaults to camera name)
    show_in_legend - Whether to show the point in the legend
    row, col - Subplot location
    
    Returns:
    Updated figure and projection info
    """
    if point_color is None:
        point_color = camera['color']
        
    if name is None:
        name = f"({camera['name']})"
        
    # Add the 3D point
    fig.add_trace(go.Scatter3d(
        x=[point_3d[0]],
        y=[point_3d[1]],
        z=[point_3d[2]],
        mode='markers',
        marker=dict(color=point_color, size=marker_size, symbol='circle'),
        name=name,
        showlegend=show_in_legend
    ), row=row, col=col)
    
    # Project the point to the camera
    proj_3d, u, v = project_point_to_camera(camera, point_3d)
    
    if proj_3d is not None:
        # Add the projected point
        fig.add_trace(go.Scatter3d(
            x=[proj_3d[0]],
            y=[proj_3d[1]],
            z=[proj_3d[2]],
            mode='markers',
            marker=dict(color=point_color, size=marker_size, symbol='x'),
            name=f'{name} Projection',
            showlegend=show_in_legend
        ), row=row, col=col)
        
        # Add projection ray
        if show_ray:
            fig.add_trace(go.Scatter3d(
                x=[camera['origin'][0], point_3d[0]],
                y=[camera['origin'][1], point_3d[1]],
                z=[camera['origin'][2], point_3d[2]],
                mode='lines',
                line=dict(color=point_color, dash='dash', width=ray_width),
                name=f'{name} Ray',
                showlegend=False
            ), row=row, col=col)
        
        # Add annotation with projection information
        # Vérifier si les annotations existent avant de les modifier
        if not hasattr(fig.layout, 'scene') or not hasattr(fig.layout.scene, 'annotations'):
            fig.update_layout(
                scene=dict(
                    annotations=[]
                )
            )
            
        fig.update_layout(
            scene=dict(
                annotations=list(fig.layout.scene.annotations) + [
                    dict(
                        showarrow=True,
                        arrowhead=1,
                        arrowcolor=point_color,
                        ax=30,
                        ay=-30,
                        x=proj_3d[0],
                        y=proj_3d[1],
                        z=proj_3d[2],
                        text=f"{name} ({proj_3d[0]:.2f}, {proj_3d[1]:.2f}, {proj_3d[2]:.2f})",
                        align="left",
                        xanchor="center",
                        yanchor="bottom",
                        font=dict(color=point_color, size=10)
                    )
                ]
            )
        )
    
    return fig, (proj_3d, u, v)