"""
Classe World pour la gestion d'une scène 3D complète
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .objects import Camera, Frame, Object3D, Point3D


class World:
    """
    Classe principale pour la gestion d'une scène 3D complète.
    Contient tous les objets (points, repères, caméras) et gère la visualisation.
    """
    
    def __init__(self):
        """Initialisation d'un monde vide"""
        self.objects = []
    
    def add_object(self, obj):
        """
        Ajoute un objet au monde
        
        Args:
            obj (Object3D): L'objet à ajouter (Point3D, Frame, Camera)
        """
        if not isinstance(obj, Object3D):
            raise TypeError(f"L'objet doit être une instance de Object3D, pas {type(obj)}")
        
        self.objects.append(obj)
        return obj
    
    def get_objects_by_type(self, obj_type):
        """
        Retourne tous les objets d'un type spécifique
        
        Args:
            obj_type (type): Le type d'objets à récupérer
            
        Returns:
            Liste des objets du type demandé
        """
        return [obj for obj in self.objects if isinstance(obj, obj_type)]
    
    def get_cameras(self):
        """
        Retourne toutes les caméras dans le monde
        
        Returns:
            Liste des caméras
        """
        return self.get_objects_by_type(Camera)
    
    def get_frames(self):
        """
        Retourne tous les repères (hors caméras) dans le monde
        
        Returns:
            Liste des repères non-caméra
        """
        return [obj for obj in self.objects if isinstance(obj, Frame) and not isinstance(obj, Camera)]
    
    def get_points(self):
        """
        Retourne tous les points dans le monde
        
        Returns:
            Liste des points
        """
        return self.get_objects_by_type(Point3D)
    
    def calculate_scene_range(self, margin=1.0):
        """
        Calcule les limites de la scène 3D
        
        Args:
            margin (float): Marge à ajouter aux limites
            
        Returns:
            Dictionnaire avec les limites x, y, z
        """
        all_points = []
        
        # Toujours inclure l'origine pour s'assurer que le repère monde est visible
        all_points.append(np.array([0.0, 0.0, 0.0]))
        
        # Ajouter tous les objets
        for obj in self.objects:
            if isinstance(obj, Camera):
                # Pour une caméra, ajouter l'origine et les coins du plan image
                all_points.append(obj.origin)
                all_points.append(obj.plane_center)
                all_points.extend(obj.plane_corners)
            elif isinstance(obj, Frame):
                # Pour un repère, ajouter l'origine
                all_points.append(obj.origin)
            elif isinstance(obj, Point3D):
                # Pour un point, ajouter ses coordonnées
                all_points.append(obj.coords)
        
        # Si nous n'avons aucun point, utiliser une plage par défaut
        if len(all_points) <= 1:  # Si seulement l'origine est présente
            return {'x': [-2, 2], 'y': [-2, 2], 'z': [-2, 2]}
        
        # Convertir en tableau numpy pour le calcul
        all_points = np.array(all_points)
        
        # Calculer min et max pour chaque axe
        min_vals = np.min(all_points, axis=0) - margin
        max_vals = np.max(all_points, axis=0) + margin
        
        return {'x': [min_vals[0], max_vals[0]], 
                'y': [min_vals[1], max_vals[1]], 
                'z': [min_vals[2], max_vals[2]]}
    
    def visualize(self, figure_title="Visualisation 3D et 2D", show_world_frame=True, height=900):
        """
        Crée une visualisation complète avec vue 3D et projections 2D pour chaque caméra
        
        Args:
            figure_title (str): Titre de la figure
            show_world_frame (bool): Afficher le repère monde
            height (int): Hauteur de la figure en pixels
            
        Returns:
            Figure Plotly avec la visualisation
        """
        # Récupérer les caméras et les points
        cameras = self.get_cameras()
        frames = self.get_frames()
        points = self.get_points()
        
        # Calculer le nombre de caméras pour la mise en page
        ncameras = len(cameras)
        
        # Créer la figure avec les sous-graphiques appropriés
        if ncameras == 0:
            # Aucune caméra, juste une vue 3D
            fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        else:
            # Créer une mise en page pour les caméras (1 ligne pour 3D, 1 ligne pour les vues 2D)
            subplot_titles = ["Scène 3D"]
            for i, cam in enumerate(cameras):
                subplot_titles.append(f"Vue {cam.name}")
            
            # Première ligne : scène 3D sur toutes les colonnes
            # Deuxième ligne : une vue 2D par caméra
            specs = [
                [{"type": "scene", "colspan": ncameras}] + [None] * (ncameras - 1),
            ]
            # Ajouter une ligne avec une cellule 'xy' par caméra
            specs.append([{"type": "xy"} for _ in range(ncameras)])
            
            fig = make_subplots(
                rows=2, 
                cols=max(1, ncameras),  # Au moins une colonne
                specs=specs,
                subplot_titles=subplot_titles,
                vertical_spacing=0.1
            )
        
        # Ajouter le repère monde
        if show_world_frame:
            self._add_world_frame(fig)
        
        # Ajouter les points 3D
        for point in points:
            self._add_point_to_plot(fig, point)
        
        # Ajouter les repères et caméras
        for frame in frames + cameras:
            self._add_frame_to_plot(fig, frame)
        
        # Ajouter les connexions points-repères
        self._add_point_frame_connections(fig)
        
        # Ajouter les projections de points sur les caméras
        if ncameras > 0 and len(points) > 0:
            camera_projections = {}
            for camera in cameras:
                camera_projections[camera.name] = []
                
                for point in points:
                    projection_info = self._add_projected_point(fig, camera, point)
                    if projection_info and projection_info[0] is not None:
                        camera_projections[camera.name].append((point, projection_info))
        
        # Créer les vues 2D pour chaque caméra
        if ncameras > 0:
            for i, camera in enumerate(cameras):
                # Créer la vue 2D
                self._add_camera_view_2d(fig, camera, i+1)
        
        # Configurer la mise en page 3D
        scene_ranges = self.calculate_scene_range()
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
    
    def _add_world_frame(self, fig, axis_length=2.0, row=1, col=1):
        """Ajoute le repère monde à la figure"""
        # Point à l'origine
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(color='black', size=8, symbol='circle'),
            name='Origine',
            showlegend=True
        ), row=row, col=col)
        
        # Axes avec flèches pour une meilleure visualisation
        fig.add_trace(go.Scatter3d(
            x=[0, axis_length], y=[0, 0], z=[0, 0],
            mode='lines', 
            line=dict(color='red', width=8), 
            name='Axe X monde'
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, axis_length], z=[0, 0],
            mode='lines', 
            line=dict(color='green', width=8),
            name='Axe Y monde'
        ), row=row, col=col)
        
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, axis_length],
            mode='lines', 
            line=dict(color='blue', width=8),
            name='Axe Z monde'
        ), row=row, col=col)
        
        return fig
    
    def _add_point_to_plot(self, fig, point, row=1, col=1):
        """Ajoute un point à la figure"""
        fig.add_trace(go.Scatter3d(
            x=[point.coords[0]],
            y=[point.coords[1]],
            z=[point.coords[2]],
            mode='markers',
            marker=dict(color=point.color, size=point.size, symbol='circle'),
            name=point.name,
            showlegend=True
        ), row=row, col=col)
        
        return fig
    
    def _add_frame_to_plot(self, fig, frame, axis_length=0.3, row=1, col=1):
        """Ajoute un repère ou une caméra à la figure"""
        # Ajouter l'origine
        fig.add_trace(go.Scatter3d(
            x=[frame.origin[0]], 
            y=[frame.origin[1]], 
            z=[frame.origin[2]],
            mode='markers',
            marker=dict(color=frame.color, size=5, symbol='diamond'),
            name=f'{frame.name} Origine',
            showlegend=True
        ), row=row, col=col)
        
        # Ajouter les axes
        # Axe Z - bleu
        z_end = frame.origin + axis_length * frame.z_axis
        fig.add_trace(go.Scatter3d(
            x=[frame.origin[0], z_end[0]],
            y=[frame.origin[1], z_end[1]],
            z=[frame.origin[2], z_end[2]],
            mode='lines',
            line=dict(color='blue', width=4),
            name=f'{frame.name} Axe Z',
            showlegend=False
        ), row=row, col=col)
        
        # Axe X - rouge
        x_end = frame.origin + axis_length * frame.x_axis
        fig.add_trace(go.Scatter3d(
            x=[frame.origin[0], x_end[0]],
            y=[frame.origin[1], x_end[1]],
            z=[frame.origin[2], x_end[2]],
            mode='lines',
            line=dict(color='red', width=4),
            name=f'{frame.name} Axe X',
            showlegend=False
        ), row=row, col=col)
        
        # Axe Y - vert
        y_end = frame.origin + axis_length * frame.y_axis
        fig.add_trace(go.Scatter3d(
            x=[frame.origin[0], y_end[0]],
            y=[frame.origin[1], y_end[1]],
            z=[frame.origin[2], y_end[2]],
            mode='lines',
            line=dict(color='#00CC00', width=4),
            name=f'{frame.name} Axe Y',
            showlegend=False
        ), row=row, col=col)
        
        # Spécificités des caméras
        if isinstance(frame, Camera):
            # Ajouter le plan image
            corners = frame.plane_corners
            fig.add_trace(go.Mesh3d(
                x=[c[0] for c in corners],
                y=[c[1] for c in corners],
                z=[c[2] for c in corners],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                opacity=0.15,
                color=frame.color,
                flatshading=True,
                name=f'{frame.name} Plan image',
                showlegend=True
            ), row=row, col=col)
            
            # Ajouter l'axe optique
            fig.add_trace(go.Scatter3d(
                x=[frame.origin[0], frame.plane_center[0]],
                y=[frame.origin[1], frame.plane_center[1]],
                z=[frame.origin[2], frame.plane_center[2]],
                mode='lines',
                line=dict(color=frame.color, dash='dash', width=2),
                name=f'{frame.name} Axe optique',
                showlegend=False
            ), row=row, col=col)
        
        return fig
    
    def _add_point_frame_connections(self, fig, row=1, col=1, line_color='black', line_width=2):
        """Ajoute les connexions entre points et repères"""
        frames = self.get_frames() + self.get_cameras()
        points = self.get_points()
        
        # Traiter chaque repère
        for frame in frames:
            # Vérifier si ce repère doit montrer les coordonnées
            if not frame.show_coordinates:
                continue
                
            # Pour chaque point, calculer les coordonnées locales par rapport au repère
            for point in points:
                # Ignorer si ce point ne veut pas de connexions
                if not point.show_connections:
                    continue
                    
                # Créer une ligne reliant l'origine du repère au point
                fig.add_trace(go.Scatter3d(
                    x=[frame.origin[0], point.coords[0]],
                    y=[frame.origin[1], point.coords[1]],
                    z=[frame.origin[2], point.coords[2]],
                    mode='lines',
                    line=dict(color=line_color, width=line_width, dash='dash'),
                    name=f"{frame.name} -> {point.name}",
                    showlegend=False
                ), row=row, col=col)
                
                # Calculer les coordonnées du point dans le système du repère
                local_coords = frame.transform_point(point.coords)
                
                # Calculer le milieu pour le placement du texte
                mid_point = frame.origin + 0.5 * (point.coords - frame.origin)
                
                # Ajouter l'annotation avec les coordonnées locales
                fig.add_trace(go.Scatter3d(
                    x=[mid_point[0]],
                    y=[mid_point[1]],
                    z=[mid_point[2]],
                    mode='text',
                    text=[f"({local_coords[0]:.2f}, {local_coords[1]:.2f}, {local_coords[2]:.2f})"],
                    textposition="middle center",
                    name=f"{frame.name} -> {point.name} coords",
                    showlegend=False
                ), row=row, col=col)
        
        return fig
    
    def _add_projected_point(self, fig, camera, point, show_ray=True, ray_width=2, row=1, col=1):
        """Ajoute un point projeté sur une caméra à la figure"""
        # Projeter le point sur la caméra
        proj_3d, u, v = camera.project_point(point)
        
        if proj_3d is not None:
            # Ajouter le point projeté
            fig.add_trace(go.Scatter3d(
                x=[proj_3d[0]],
                y=[proj_3d[1]],
                z=[proj_3d[2]],
                mode='markers',
                marker=dict(color=point.color, size=5, symbol='x'),
                name=f'{point.name} ({camera.name})',
                showlegend=True
            ), row=row, col=col)
            
            # Ajouter le rayon de projection
            if show_ray:
                fig.add_trace(go.Scatter3d(
                    x=[camera.origin[0], point.coords[0]],
                    y=[camera.origin[1], point.coords[1]],
                    z=[camera.origin[2], point.coords[2]],
                    mode='lines',
                    line=dict(color=point.color, dash='dash', width=ray_width),
                    name=f'{point.name} ({camera.name}) Rayon',
                    showlegend=False
                ), row=row, col=col)
            
        return (proj_3d, u, v)
    
    def _add_camera_view_2d(self, fig, camera, col_idx, row=2):
        """Ajoute une vue 2D de la caméra à la figure"""
        # Récupérer les dimensions du plan
        w, h = camera._plane_size_value
        
        # Créer un rectangle représentant les limites de l'image (-w à w, -h à h)
        fig.add_shape(
            type="rect",
            x0=-w, y0=-h,
            x1=w, y1=h,
            line=dict(color="black", width=2),
            fillcolor="rgba(255, 255, 255, 0.0)",
            row=row, col=col_idx
        )
        
        # Ajouter un réticule central
        fig.add_shape(
            type="line", x0=-w, y0=0, x1=w, y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            row=row, col=col_idx
        )
        fig.add_shape(
            type="line", x0=0, y0=-h, x1=0, y1=h,
            line=dict(color="gray", width=1, dash="dash"),
            row=row, col=col_idx
        )
        
        # Projeter et ajouter les points 3D
        for point in self.get_points():
            # Projeter le point sur le plan image
            proj_3d, u, v = camera.project_point(point)
            
            if proj_3d is not None:
                # Ajouter le point projeté
                fig.add_trace(go.Scatter(
                    x=[u], y=[v],
                    mode='markers+text',
                    marker=dict(color=point.color, size=10, symbol='x'),
                    text=[f"({u:.2f}, {v:.2f})"],
                    textposition="top center",
                    name=point.name,
                    showlegend=False
                ), row=row, col=col_idx)
        
        # Configurer les axes
        fig.update_xaxes(
            range=[-w * 1.2, w * 1.2],
            zeroline=True, zerolinecolor='black',
            gridcolor='lightgray',
            title_text="u (horizontal)",
            row=row, col=col_idx
        )
        fig.update_yaxes(
            range=[-h * 1.2, h * 1.2],
            zeroline=True, zerolinecolor='black',
            gridcolor='lightgray',
            scaleanchor="x",
            scaleratio=1,
            title_text="v (vertical)",
            row=row, col=col_idx
        )
        
        return fig