"""
plotcv - Une bibliothèque de visualisation pour la vision par ordinateur

Cette bibliothèque permet de visualiser des caméras et des projections 3D/2D
en utilisant des graphiques interactifs.
"""

from plotcv.camera import create_camera_frame
from plotcv.projection import add_projected_point, project_point_to_camera
from plotcv.visualization import (
    add_camera_to_plot,
    create_camera_view_2d,
    create_visualization_3d_2d,
)

__all__ = [
    'create_camera_frame',
    'add_camera_to_plot',
    'create_visualization_3d_2d',
    'create_camera_view_2d',
    'project_point_to_camera',
    'add_projected_point'
]