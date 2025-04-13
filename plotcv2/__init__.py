"""
plotcv2 - Bibliothèque de visualisation 3D pour la vision par ordinateur (version orientée objet)
"""

from .objects import Camera, Frame, Point3D
from .world import World

__all__ = ["World", "Frame", "Camera", "Point3D"]
