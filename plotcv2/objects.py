"""
Définition des classes d'objets 3D pour la visualisation
"""

from abc import ABC, abstractmethod

import numpy as np


class Object3D(ABC):
    """Classe abstraite pour tous les objets 3D dans la scène"""
    
    def __init__(self, name=None, color=None):
        """
        Initialisation d'un objet 3D
        
        Args:
            name (str): Nom de l'objet
            color (str): Couleur pour la visualisation
        """
        self.name = name if name is not None else self.__class__.__name__
        self.color = color if color is not None else "blue"
    
    @abstractmethod
    def get_world_coordinates(self):
        """Renvoie les coordonnées de l'objet dans le repère monde"""
        pass


class Point3D(Object3D):
    """Classe représentant un point 3D"""
    
    def __init__(self, coords, name=None, color=None, size=8, show_connections=True):
        """
        Initialisation d'un point 3D
        
        Args:
            coords (tuple/list/ndarray): Coordonnées 3D (x, y, z)
            name (str): Nom du point
            color (str): Couleur pour la visualisation
            size (int): Taille du marqueur
            show_connections (bool): Activer les connexions aux repères
        """
        super().__init__(name=name, color=color if color is not None else "green")
        self.coords = np.array(coords)
        self.size = size
        self.show_connections = show_connections
    
    def get_world_coordinates(self):
        """Renvoie les coordonnées du point dans le repère monde"""
        return self.coords


class Frame(Object3D):
    """Classe représentant un repère de coordonnées 3D"""
    
    def __init__(self, rotation=(0, 0, 0), translation=(0, 0, 0), 
                 name=None, color=None, parent=None, show_coordinates=True):
        """
        Initialisation d'un repère de coordonnées
        
        Args:
            rotation (tuple): Angles de rotation (rx, ry, rz) en degrés
            translation (tuple): Position du repère (tx, ty, tz)
            name (str): Nom du repère
            color (str): Couleur pour la visualisation
            parent (Frame): Repère parent (None = repère monde)
            show_coordinates (bool): Afficher les coordonnées des points
        """
        super().__init__(name=name, color=color if color is not None else "red")
        
        # Convertir les angles en radians
        self.rotation = np.array([np.deg2rad(angle) for angle in rotation])
        self.translation = np.array(translation)
        self.parent = parent
        self.show_coordinates = show_coordinates
        
        # Calculer les axes et l'origine
        self._update_transform()
    
    def _update_transform(self):
        """Mise à jour des transformations (axes et origine)"""
        # Calculer la matrice de rotation locale
        local_R = self._rotation_matrix_from_angles(*self.rotation)
        
        # Gérer les transformations du parent si nécessaire
        if self.parent is not None:
            # Rotation composée: R_monde = R_parent * R_local
            parent_R = np.vstack([
                self.parent.x_axis.reshape(1, 3),
                self.parent.y_axis.reshape(1, 3),
                self.parent.z_axis.reshape(1, 3)
            ])
            R = parent_R @ local_R
            
            # Translation composée: T = T_parent + R_parent * T_local
            T = self.parent.origin + parent_R @ self.translation
        else:
            # Pas de parent, utiliser les transformations directement
            R = local_R
            T = self.translation
        
        # Axes du repère dans le repère mondial
        self.x_axis = R @ np.array([1, 0, 0])
        self.y_axis = R @ np.array([0, 1, 0])
        self.z_axis = R @ np.array([0, 0, 1])
        self.origin = T
    
    def _rotation_matrix_from_angles(self, rx, ry, rz):
        """
        Créer une matrice de rotation 3D à partir des angles d'Euler (en radians)
        Utilise la convention ZYX (d'abord Z, puis Y, puis X)
        
        Args:
            rx, ry, rz: Angles de rotation en radians autour des axes X, Y, Z
            
        Returns:
            Matrice de rotation 3x3
        """
        # Rotation autour de l'axe X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        
        # Rotation autour de l'axe Y
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        
        # Rotation autour de l'axe Z
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # Combiner les rotations - convention ZYX
        return Rz @ Ry @ Rx
    
    def get_world_coordinates(self):
        """Renvoie l'origine du repère dans le repère monde"""
        return self.origin
    
    def transform_point(self, point):
        """
        Transforme un point du repère monde vers le repère local
        
        Args:
            point (ndarray): Coordonnées du point dans le repère monde
            
        Returns:
            Coordonnées du point dans le repère local
        """
        # Calculer le point par rapport à l'origine du repère
        point_rel = point - self.origin
        
        # Créer une matrice de rotation à partir des axes du repère
        R = np.vstack([
            self.x_axis.reshape(1, 3),
            self.y_axis.reshape(1, 3),
            self.z_axis.reshape(1, 3)
        ])
        
        # Transformer le point
        local_coords = np.matmul(R, point_rel)
        
        return local_coords


class Camera(Frame):
    """Classe représentant une caméra (un repère avec un plan image)"""
    
    def __init__(self, rotation=(0, 0, 0), translation=(0, 0, 0), 
                 focal_length=1.0, plane_size=(0.8, 0.6), optical_center=(0, 0),
                 name=None, color=None, parent=None, show_coordinates=True):
        """
        Initialisation d'une caméra
        
        Args:
            rotation (tuple): Angles de rotation (rx, ry, rz) en degrés
            translation (tuple): Position de la caméra (tx, ty, tz)
            focal_length (float/tuple): Distance focale (ou tuple fx, fy)
            plane_size (tuple): Demi-dimensions du plan image (w, h)
            optical_center (tuple): Centre optique (cx, cy)
            name (str): Nom de la caméra
            color (str): Couleur pour la visualisation
            parent (Frame): Repère parent (None = repère monde)
            show_coordinates (bool): Afficher les coordonnées des points
        """
        # Stocker les paramètres directement pour éviter des problèmes d'attributs
        self._focal_length_value = focal_length
        self._optical_center_value = optical_center if optical_center is not None else (0, 0)
        self._plane_size_value = plane_size
            
        super().__init__(
            rotation=rotation, 
            translation=translation, 
            name=name, 
            color=color if color is not None else "blue", 
            parent=parent,
            show_coordinates=show_coordinates
        )
        
        # Calculer les éléments du plan image
        self._update_image_plane()
    
    def _update_transform(self):
        """Mise à jour des transformations et du plan image"""
        super()._update_transform()
        self._update_image_plane()
    
    def _update_image_plane(self):
        """Calculer les propriétés du plan image"""
        # Direction de vue = axe Z de la caméra
        self.view_dir = self.z_axis
        
        # Extraire la distance focale à partir de la valeur stockée
        if isinstance(self._focal_length_value, (int, float)):
            focal_x = focal_y = float(self._focal_length_value)
        else:
            focal_x = float(self._focal_length_value[0])
            focal_y = float(self._focal_length_value[1])
        
        # Position de base du centre du plan image
        base_plane_center = self.origin + focal_x * self.view_dir
        
        # Appliquer le décalage du centre optique
        cx, cy = self._optical_center_value
        optical_offset_x = cx * self.x_axis
        optical_offset_y = cy * self.y_axis
        
        # Centre du plan image avec décalage optique
        self.plane_center = base_plane_center + optical_offset_x + optical_offset_y
        self.plane_normal = self.view_dir  # Normal au plan = direction de vue
        
        # Axes u,v dans le plan image
        self.u_axis = self.x_axis
        self.v_axis = self.y_axis
        
        # Dimensions du plan image
        w, h = self._plane_size_value
        self.x_vec = self.u_axis * w
        self.y_vec = self.v_axis * h
        
        # Coins du plan image
        self.plane_corners = [
            self.plane_center - self.x_vec - self.y_vec,  # Coin bas-gauche
            self.plane_center + self.x_vec - self.y_vec,  # Coin bas-droite
            self.plane_center + self.x_vec + self.y_vec,  # Coin haut-droite
            self.plane_center - self.x_vec + self.y_vec,  # Coin haut-gauche
        ]
    
    def project_point(self, point):
        """
        Projète un point 3D sur le plan image de la caméra
        
        Args:
            point (ndarray): Coordonnées 3D du point
            
        Returns:
            Tuple (coord_3d, u, v) ou None si le point est derrière la caméra
            coord_3d: Coordonnées 3D du point projeté sur le plan image
            u, v: Coordonnées 2D dans le plan image
        """
        # Calculer le rayon de la caméra vers le point
        if isinstance(point, Point3D):
            point_coords = point.get_world_coordinates()
        else:
            point_coords = np.array(point)
            
        ray_dir = point_coords - self.origin
        
        # Calculer l'intersection avec le plan image
        # Formule d'intersection rayon-plan
        denom = np.dot(ray_dir, self.plane_normal)
        if abs(denom) < 1e-6:  # Le rayon est parallèle au plan
            return None, None, None
            
        t = np.dot(self.plane_center - self.origin, self.plane_normal) / denom
        if t <= 0:  # Le point est derrière la caméra
            return None, None, None
            
        # Coordonnées 3D de la projection sur le plan image
        proj_3d = self.origin + t * ray_dir
        
        # Calculer les coordonnées 2D (u,v) dans le plan image
        vec = proj_3d - self.plane_center
        u = np.dot(vec, self.u_axis)
        v = np.dot(vec, self.v_axis)
        
        return proj_3d, u, v