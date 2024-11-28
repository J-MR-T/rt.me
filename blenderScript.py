import bpy
import json
import mathutils
import bmesh
import os
import subprocess
from typing import Dict, List, Any, Optional, Tuple
import math
from math import degrees
from pathlib import Path

class BlenderSceneExporter:
    whichMode = "pathtrace"
    useVertexNormals = True

    # mirror along the x-axis, then rotate 90 degrees on the z axis
    fixTransform = mathutils.Matrix.Scale(-1, 4, (1, 0, 0)) @ mathutils.Matrix.Rotation(math.radians(-90), 4, 'Z')

    invFixTransform = fixTransform.inverted()
    
    def __init__(self):
        self.scene = bpy.context.scene
        self.depsgraph = bpy.context.evaluated_depsgraph_get()
        self.texture_cache = {}  # Cache for processed textures
        
    def _ensure_texture_dir(self) -> None:
        """Ensure the textures directory exists."""
        os.makedirs('/tmp/textures', exist_ok=True)
        
    def _convert_texture_to_ppm(self, image_path: str) -> str:
        """
        Convert texture to PPM format using ImageMagick.
        Returns the path to the converted texture.
        """
        if image_path in self.texture_cache:
            return self.texture_cache[image_path]
            
        self._ensure_texture_dir()
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        ppm_path = f'/tmp/textures/{base_name}.ppm'
        
        try:
            subprocess.run(['convert', image_path, ppm_path], check=True)
            self.texture_cache[image_path] = f'/tmp/textures/{base_name}.ppm'
            return self.texture_cache[image_path]
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to convert texture {image_path}: {e}")
            return None
            
    def _get_material_data(self, material: bpy.types.Material) -> Dict[str, Any] | None:
        """Extract material data from a Blender material."""
        if not material or not material.use_nodes:
            return None
            
        # Try to find Principled BSDF node
        principled = None
        image_texture = None
        
        for node in material.node_tree.nodes:
            if node.type == 'BSDF_PRINCIPLED':
                principled = node
            elif node.type == 'TEX_IMAGE':
                image_texture = node
                
        if not principled:
            return None
            
        material_data = {
            "ks": 0.5,  # Default specular intensity
            "kd": 1.0,  # Default diffuse intensity
            "diffusecolor": list(principled.inputs['Base Color'].default_value[:3]),
            "speculartint": mathutils.Vector(principled.inputs['Specular Tint'].default_value[:]).length,
            "subsurface": principled.inputs['Subsurface Weight'].default_value,
            "metallic": principled.inputs['Metallic'].default_value,
            "roughness": principled.inputs['Roughness'].default_value,
            "anisotropic": principled.inputs['Anisotropic'].default_value,
            "sheen": principled.inputs['Sheen Weight'].default_value,
            "sheentint": mathutils.Vector(principled.inputs['Sheen Tint'].default_value[:]).length,
            "clearcoat": principled.inputs['Coat Weight'].default_value,
            "clearcoatgloss": 1 - principled.inputs['Coat Roughness'].default_value,
            "emissive" : principled.inputs['Emission Strength'].default_value
        }
        
        # Handle texture if present
        if image_texture and image_texture.image:
            texture_path = bpy.path.abspath(image_texture.image.filepath)
            if texture_path:
                converted_path = self._convert_texture_to_ppm(texture_path)
                if converted_path:
                    material_data["texture"] = converted_path
                else:
                    raise RuntimeError(f"Failed to convert texture: {str(texture_path)}/{str(converted_path)}")
            else:
                raise RuntimeError(f"Failed to find texture: {str(texture_path)}")
                    
        return material_data
        
    def _get_default_material(self) -> Dict[str, Any]:
        """Return default material properties."""
        return {
            "ks": 0.0,
            "kd": 1.0,
            "diffusecolor": [0.8, 0.8, 0.8],
            "speculartint": 0.0,
            "subsurface": 0.0,
            "metallic": 0.0,
            "roughness": 0.5,
            "anisotropic": 0.0,
            "sheen": 0.0,
            "sheentint": 0.0,
            "clearcoat": 0.0,
            "clearcoatgloss": 0.0
        }
        
    def _get_uv_coords(self, mesh: bpy.types.Mesh, poly_idx: int, loop_indices: List[int]) -> List[List[float]]:
        """Get UV coordinates for a face."""
        if not mesh.uv_layers.active:
            return [[0, 0], [1, 1], [0, 1]]  # Default UV mapping
            
        uv_layer = mesh.uv_layers.active.data

        uvs = [list(uv_layer[loop_idx].uv) for loop_idx in loop_indices]
        for uv in uvs:
            # right handed coordinates
            uv[1] = 1 - uv[1]

        return uvs
        
    def _triangulate_mesh(self, obj: bpy.types.Object, mesh: bpy.types.Mesh) -> List[Dict[str, Any]]:
        """Convert a mesh to triangles, with material and UV support."""
        triangles = []
        
        # Make a copy of the mesh for triangulation
        temp_mesh = mesh.copy()
        temp_mesh.transform(obj.matrix_world)
        temp_mesh.transform(self.fixTransform)
        
        # Triangulate the mesh
        bm = bmesh.new()
        bm.from_mesh(temp_mesh)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.to_mesh(temp_mesh)
        bm.free()
        
        # Create triangles
        for poly in temp_mesh.polygons:
            if len(poly.vertices) != 3:
                continue
                
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
            
            # Get material
            material = obj.material_slots[poly.material_index].material if obj.material_slots else None
            material_data = self._get_material_data(material)
            if material_data:
                # Get UV coordinates
                uv_coords = self._get_uv_coords(temp_mesh, poly.index, poly.loop_indices)
                if uv_coords:
                    material_data["txv0"] = uv_coords[0]
                    material_data["txv1"] = uv_coords[1]
                    material_data["txv2"] = uv_coords[2]
                    
                triangle["material"] = material_data

            triangles.append(triangle)
            
        # Clean up
        bpy.data.meshes.remove(temp_mesh)
        
        return triangles
        
    def _create_sphere_data(self, obj: bpy.types.Object) -> Dict[str, Any]:
        """Create sphere primitive data with material support."""

        loc = self.fixTransform @ obj.location

        sphere_data = {
            "type": "sphere",
            "center": loc[:],
            "radius": max(obj.dimensions) / 2
        }
        
        # Add material
        material = obj.material_slots[0].material if obj.material_slots else None
        material_data = self._get_material_data(material)
        if material_data:
            sphere_data["material"] = material_data
        
        
        return sphere_data
        
    def _create_cylinder_data(self, obj: bpy.types.Object) -> Dict[str, Any]:
        """Create cylinder primitive data with material support."""
        up_vector = obj.matrix_world.to_3x3() @ mathutils.Vector((0, 0, 1))
        up_vector.normalize()

        loc = self.fixTransform @ obj.location

        
        cylinder_data = {
            "type": "cylinder",
            "center": loc[:],
            "axis": list(up_vector),
            "radius": obj.dimensions[0] / 2,
            "height": obj.dimensions[2]
        }
        
        # Add material
        material = obj.material_slots[0].material if obj.material_slots else None
        
        material_data = self._get_material_data(material)
        if material_data:
            cylinder_data["material"] = material_data
        
        return cylinder_data
        
    def export_scene(self, filepath: str) -> None:
        """
        Export the current Blender scene to a custom JSON format.
        
        Args:
            filepath: Path where the JSON file will be saved
        """
        try:
            scene_data = {
                "rendermode": self.whichMode,
                "pathtracingOpts": {},
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
        cam_loc = cam_matrix.to_translation() @ self.invFixTransform
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
                    "position": list(light_obj.location @ self.invFixTransform),
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

    @staticmethod
    def _is_primitive_sphere(obj: bpy.types.Object, mesh: bpy.types.Mesh) -> bool:
        """Check if the object is likely a primitive sphere."""
        return (obj.name.startswith("Sphere") and 
                abs(obj.scale.x - obj.scale.y) < 0.01 and
                abs(obj.scale.y - obj.scale.z) < 0.01)

    @staticmethod
    def _is_primitive_cylinder(obj: bpy.types.Object) -> bool:
        """Check if the object is likely a primitive cylinder."""
        return (obj.name.startswith("Cylinder") and
                abs(obj.scale.x - obj.scale.y) < 0.01)

def export_scene(filepath: str) -> None:
    """Convenience function to export the current scene."""
    exporter = BlenderSceneExporter()
    exporter.export_scene(filepath)

if __name__ == "__main__":
    export_scene('/tmp/scene.json')



