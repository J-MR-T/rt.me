import bpy
import json
import mathutils

# Function to triangulate all meshes
def triangulate_object(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()
    bpy.ops.object.mode_set(mode='OBJECT')

# Function to export the scene
def export_scene(filepath):
    scene_data = {
        "rendermode": "phong",
        "camera": {
            "type": "pinhole",
            "width": bpy.context.scene.render.resolution_x,
            "height": bpy.context.scene.render.resolution_y,
            "position": list(bpy.context.scene.camera.location),
            "lookAt": list(bpy.context.scene.camera.matrix_world.to_translation() + bpy.context.scene.camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 0.0, -1.0))),
            "upVector": list(bpy.context.scene.camera.matrix_world.to_quaternion() @ mathutils.Vector((0.0, 1.0, 0.0))),
            "fov": bpy.context.scene.camera.data.angle * (180 / 3.14159),  # convert radians to degrees
            "exposure": 0.1  # Example exposure, you might adjust based on your scene
        },
        "scene": {
            "backgroundcolor": [0.25, 0.25, 0.25],
            "lightsources": [],
            "shapes": []
        }
    }

    # Process lights
    for light in [obj for obj in bpy.context.scene.objects if obj.type == 'LIGHT']:
        if light.data.type == 'POINT':
            intensityC = light.data.color*light.data.energy/1000
            intensity = [intensityC.r, intensityC.g, intensityC.b]
            light_data = {
                "type": "pointlight",
                "position": list(light.location),
                "intensity": intensity
            }
            scene_data["scene"]["lightsources"].append(light_data)

    # Process objects
    for obj in [o for o in bpy.context.scene.objects if o.type == 'MESH']:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode='OBJECT')
        depsgraph = bpy.context.evaluated_depsgraph_get()
        evaluated_obj = obj.evaluated_get(depsgraph)
        temp_mesh = evaluated_obj.to_mesh()

        # Create a new mesh in the main database to avoid ReferenceError
        new_mesh = temp_mesh.copy()

        try:
            # Check for primitive shapes
            if obj.name.startswith("Sphere") and len(new_mesh.vertices) <= 100:  # Approximate small UV spheres
                scene_data["scene"]["shapes"].append({
                    "type": "sphere",
                    "center": list(obj.location),
                    "radius": max(obj.dimensions) / 2
                })
            elif obj.name.startswith("Cylinder"):
                # Get axis direction and height
                scene_data["scene"]["shapes"].append({
                    "type": "cylinder",
                    "center": list(obj.location),
                    "axis": [0, 0, 1],  # Simplified, assuming cylinders are along z-axis
                    "radius": obj.dimensions[0] / 2,
                    "height": obj.dimensions[2]
                })
            else:
                # Convert mesh to triangles if it's not already
                triangulate_object(obj)
                for poly in new_mesh.polygons:
                    triangle = {
                        "type": "triangle",
                        "v0": list(obj.matrix_world @ new_mesh.vertices[poly.vertices[0]].co),
                        "v1": list(obj.matrix_world @ new_mesh.vertices[poly.vertices[1]].co),
                        "v2": list(obj.matrix_world @ new_mesh.vertices[poly.vertices[2]].co)
                    }
                    scene_data["scene"]["shapes"].append(triangle)
        finally:
            # Clean up both temporary meshes
            evaluated_obj.to_mesh_clear()
            bpy.data.meshes.remove(new_mesh)

    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(scene_data, f, indent=4)

# Example usage:
export_scene('/tmp/scene.json')

