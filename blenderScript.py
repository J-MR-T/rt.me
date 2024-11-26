import bpy
import json
import mathutils
import bmesh
from typing import Dict, List, Any, Optional
from math import degrees
from pathlib import Path

class BlenderSceneExporter:
    # whichMode = "phong"
    whichMode = "pathtrace"
    useVertexNormals = True

    def __init__(self):
        self.scene = bpy.context.scene
        self.depsgraph = bpy.context.evaluated_depsgraph_get()
        
    def export_scene(self, filepath: str) -> None:
        """
        Export the current Blender scene to a custom JSON format.
        
        Args:
            filepath: Path where the JSON file will be saved
        """
        try:
            scene_data = {
                "rendermode": self.whichMode,
                "camera": self._get_camera_data(),
                "scene": {
                    "backgroundcolor": [0.25, 0.25, 0.25],
                    "lightsources": self._get_light_sources(),
                    "shapes": self._get_shapes()
                }
            }

            if(self.whichMode == "pathtrace"):
                scene_data["scene"]["backgroundcolor"] = [0, 0, 0]
                scene_data["pathtracingOpts"] = {
                    "samplesPerPixel" : 4,
                    "apertureSamplesPerPixelSample" : 1,
                    "pointLightSamplesPerBounce" : 1,
                    "hemisphereSamplesPerBounce" : 1,
                    "incremental": True
                }
            
            # Ensure directory exists
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(scene_data, f, indent=4)
                
        except Exception as e:
            raise RuntimeError(f"Failed to export scene: {str(e)}")

    def _get_camera_data(self) -> Dict[str, Any]:
        """Extract camera data from the active scene camera."""
        camera = self.scene.camera
        if not camera:
            raise ValueError("No active camera in scene")
            
        # Get camera transformation data
        cam_matrix = camera.matrix_world
        cam_loc = cam_matrix.to_translation()
        cam_rot = cam_matrix.to_quaternion()
        
        return {
            "type": "pinhole",
            "width": self.scene.render.resolution_x,
            "height": self.scene.render.resolution_y,
            "position": list(cam_loc),
            "lookAt": list(cam_loc + cam_rot @ mathutils.Vector((0.0, 0.0, -1.0))),
            "upVector": list(cam_rot @ mathutils.Vector((0.0, 1.0, 0.0))),
            "fov": degrees(camera.data.angle),
            "exposure": 0.1
        }

    def _get_light_sources(self) -> List[Dict[str, Any]]:
        """Extract all point light data from the scene."""
        lights = []
        for light_obj in [obj for obj in self.scene.objects if obj.type == 'LIGHT']:
            if light_obj.data.type == 'POINT':
                light_color = light_obj.data.color * (light_obj.data.energy / 1000)
                lights.append({
                    "type": "pointlight",
                    "position": list(light_obj.location),
                    "intensity": [light_color.r, light_color.g, light_color.b]
                })
        return lights

    def _get_shapes(self) -> List[Dict[str, Any]]:
        """Process all mesh objects in the scene and convert them to the appropriate shape type."""
        shapes = []
        mesh_objects = [obj for obj in self.scene.objects if obj.type == 'MESH']
        
        for obj in mesh_objects:
            shape = self._process_mesh_object(obj)
            if isinstance(shape, list):
                shapes.extend(shape)
            else:
                shapes.append(shape)
                
        return shapes

    def _process_mesh_object(self, obj: bpy.types.Object) -> Any:
        """
        Process a single mesh object and return its shape representation.
        
        Args:
            obj: Blender mesh object to process
            
        Returns:
            Either a dictionary representing a primitive shape or a list of triangle dictionaries
        """
        # Get evaluated version of the mesh
        evaluated_obj = obj.evaluated_get(self.depsgraph)
        temp_mesh = evaluated_obj.to_mesh()
        
        try:
            # Check for primitive shapes first
            if self._is_primitive_sphere(obj, temp_mesh):
                return self._create_sphere_data(obj)
            elif self._is_primitive_cylinder(obj):
                return self._create_cylinder_data(obj)
            else:
                return self._triangulate_mesh(obj, temp_mesh)
        finally:
            # Clean up temporary mesh
            evaluated_obj.to_mesh_clear()

    def _triangulate_mesh(self, obj: bpy.types.Object, mesh: bpy.types.Mesh) -> List[Dict[str, Any]]:
        """Convert a mesh to triangles, ensuring proper triangulation."""
        triangles = []
        
        # Make a copy of the mesh for triangulation
        temp_mesh = mesh.copy()
        temp_mesh.transform(obj.matrix_world)
        
        # Triangulate the mesh
        bm = bmesh.new()
        bm.from_mesh(temp_mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(temp_mesh)
        bm.free()
        
        # Create triangles
        for poly in temp_mesh.polygons:
            if len(poly.vertices) != 3:
                continue  # Skip non-triangular faces (shouldn't happen after triangulation)
                
            triangle = {
                "type": "triangle",
                "v0": list(temp_mesh.vertices[poly.vertices[0]].co),
                "v1": list(temp_mesh.vertices[poly.vertices[1]].co),
                "v2": list(temp_mesh.vertices[poly.vertices[2]].co)
            }
            
            if self.useVertexNormals:
                triangle["v0Normal"] = list(temp_mesh.vertices[poly.vertices[0]].normal)
                triangle["v1Normal"] = list(temp_mesh.vertices[poly.vertices[1]].normal)
                triangle["v2Normal"] = list(temp_mesh.vertices[poly.vertices[2]].normal)
                
            
            
            triangles.append(triangle)
            
        # Clean up
        bpy.data.meshes.remove(temp_mesh)
        
        return triangles

    @staticmethod
    def _is_primitive_sphere(obj: bpy.types.Object, mesh: bpy.types.Mesh) -> bool:
        """Check if the object is likely a primitive sphere."""
        return (obj.name.startswith("Sphere") and 
                len(mesh.vertices) <= 100 and
                abs(obj.scale.x - obj.scale.y) < 0.01 and
                abs(obj.scale.y - obj.scale.z) < 0.01)

    @staticmethod
    def _is_primitive_cylinder(obj: bpy.types.Object) -> bool:
        """Check if the object is likely a primitive cylinder."""
        return (obj.name.startswith("Cylinder") and
                abs(obj.scale.x - obj.scale.y) < 0.01)

    def _create_sphere_data(self, obj: bpy.types.Object) -> Dict[str, Any]:
        """Create sphere primitive data."""
        return {
            "type": "sphere",
            "center": list(obj.location),
            "radius": max(obj.dimensions) / 2
        }

    def _create_cylinder_data(self, obj: bpy.types.Object) -> Dict[str, Any]:
        """Create cylinder primitive data."""
        # Get the cylinder's local up axis transformed to world space
        up_vector = obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))
        up_vector.normalize()
        
        return {
            "type": "cylinder",
            "center": list(obj.location),
            "axis": list(up_vector),
            "radius": obj.dimensions[0] / 2,
            "height": obj.dimensions[2]
        }

def export_scene(filepath: str) -> None:
    """Convenience function to export the current scene."""
    exporter = BlenderSceneExporter()
    exporter.export_scene(filepath)

if __name__ == "__main__":
    export_scene('/tmp/scene.json')

