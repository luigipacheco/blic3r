import bpy
import bmesh
from mathutils import Vector, Matrix
from bpy.props import FloatProperty, IntProperty, EnumProperty
from bpy.types import Operator, Panel
import math

bl_info = {
    "name": "Simple Retopo",
    "author": "Luis Pacheco", 
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > Retopo",
    "description": "Simple retopology tool with contour cuts",
    "category": "Mesh",
}

class MESH_OT_simple_retopo(Operator):
    """Simple retopology using contour cuts that wrap around geometry"""
    bl_idname = "mesh.simple_retopo"
    bl_label = "Simple Retopo"
    bl_options = {'REGISTER', 'UNDO'}
    
    axis: EnumProperty(
        name="Axis",
        description="Axis along which to create contour cuts",
        items=[
            ('Z', "Z (Up)", "Cut along Z axis"),
            ('Y', "Y (Front)", "Cut along Y axis"), 
            ('X', "X (Right)", "Cut along X axis")
        ],
        default='Z'
    )
    
    cut_distance: FloatProperty(
        name="Cut Distance",
        description="Distance between contour cuts",
        default=0.2,
        min=0.001,
        max=10.0,
        step=0.01,
        precision=3
    )
    
    subdivisions: IntProperty(
        name="Subdivisions",
        description="Number of subdivisions around each contour", 
        default=16,
        min=4,
        max=1024
    )
    
    offset: FloatProperty(
        name="Surface Offset",
        description="Offset from original surface",
        default=0.001,
        min=0.0,
        max=1.0,
        step=0.001,
        precision=4
    )
    
    def invoke(self, context, event):
        # Get values from scene properties
        self.axis = context.scene.retopo_axis
        self.cut_distance = context.scene.retopo_cut_distance
        self.subdivisions = context.scene.retopo_subdivisions
        self.offset = context.scene.retopo_offset
        return self.execute(context)
    
    @classmethod
    def poll(cls, context):
        return (context.active_object is not None and 
                context.active_object.type == 'MESH' and
                context.mode == 'OBJECT')
    
    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'MESH':
            self.report({'ERROR'}, "No mesh object selected")
            return {'CANCELLED'}
        
        try:
            # Create bmesh from mesh and apply world transform
            bm = bmesh.new()
            bm.from_mesh(obj.data)
            bm.transform(obj.matrix_world)
            
            # Ensure faces and normals are up to date
            bm.faces.ensure_lookup_table()
            bm.normal_update()
            bmesh.ops.triangulate(bm, faces=bm.faces)
            bm.faces.ensure_lookup_table()
            
            # Get axis info
            axis_map = {'X': 0, 'Y': 1, 'Z': 2}
            axis_idx = axis_map[self.axis]
            
            # Calculate bounds along chosen axis
            coords = [v.co[axis_idx] for v in bm.verts]
            min_bound = min(coords)
            max_bound = max(coords)
            
            print(f"Axis {self.axis}: bounds {min_bound:.3f} to {max_bound:.3f}")
            
            # Generate cutting planes at regular intervals
            cut_positions = []
            current = min_bound + self.cut_distance
            while current < max_bound - self.cut_distance * 0.1:
                cut_positions.append(current)
                current += self.cut_distance
            
            print(f"Generated {len(cut_positions)} cut positions")
            
            if len(cut_positions) < 2:
                self.report({'ERROR'}, "Need smaller cut distance or larger object")
                bm.free()
                return {'CANCELLED'}
            
            # Create new mesh
            new_mesh = bpy.data.meshes.new(f"{obj.data.name}_retopo")
            new_bm = bmesh.new()
            
            # Generate contour curves for each cutting plane
            all_loops = []
            for i, cut_pos in enumerate(cut_positions):
                print(f"Processing cut {i+1}/{len(cut_positions)} at {cut_pos:.3f}")
                
                # Find all edge intersections with this plane
                intersections = self.find_plane_intersections(bm, axis_idx, cut_pos)
                
                if len(intersections) < 3:
                    print(f"  Insufficient intersections: {len(intersections)}")
                    continue
                
                # Create ordered contour loops from intersections
                contour_loops = self.create_contour_loops(intersections, axis_idx, cut_pos)
                
                if not contour_loops:
                    print(f"  No valid contour loops created")
                    continue
                
                # Convert each loop to evenly subdivided vertices
                for loop in contour_loops:
                    if len(loop) >= 3:
                        subdivided_loop = self.subdivide_loop(loop, self.subdivisions)
                        if subdivided_loop:
                            # Add vertices to bmesh with offset
                            bm_verts = []
                            for vert_co in subdivided_loop:
                                # Apply surface offset
                                offset_pos = vert_co + Vector((0, 0, self.offset)) if axis_idx == 2 else vert_co
                                bm_verts.append(new_bm.verts.new(offset_pos))
                            all_loops.append(bm_verts)
                            print(f"  Created loop with {len(subdivided_loop)} vertices")
            
            print(f"Total valid loops: {len(all_loops)}")
            
            if len(all_loops) < 2:
                self.report({'ERROR'}, "Not enough valid contour loops")
                bm.free()
                new_bm.free()
                return {'CANCELLED'}
            
            # Update vertex indices
            new_bm.verts.ensure_lookup_table()
            
            # Create quad faces between consecutive loops
            faces_created = 0
            for i in range(len(all_loops) - 1):
                loop1 = all_loops[i]
                loop2 = all_loops[i + 1]
                
                # Only connect if both loops have same subdivision count
                if len(loop1) == len(loop2) == self.subdivisions:
                    for j in range(len(loop1)):
                        v1 = loop1[j]
                        v2 = loop1[(j + 1) % len(loop1)]
                        v3 = loop2[(j + 1) % len(loop2)]
                        v4 = loop2[j]
                        
                        try:
                            new_bm.faces.new([v1, v2, v3, v4])
                            faces_created += 1
                        except ValueError:
                            continue
            
            print(f"Created {faces_created} quad faces")
            
            # Finalize mesh
            new_bm.normal_update()
            new_bm.to_mesh(new_mesh)
            
            # Create new object
            new_obj = bpy.data.objects.new(f"{obj.name}_retopo", new_mesh)
            context.collection.objects.link(new_obj)
            new_obj.matrix_world = obj.matrix_world.copy()
            
            # Clean up
            bm.free()
            new_bm.free()
            
            # Select new object
            context.view_layer.objects.active = new_obj
            new_obj.select_set(True)
            obj.select_set(False)
            
            self.report({'INFO'}, f"Created {len(all_loops)} loops, {faces_created} faces")
            return {'FINISHED'}
            
        except Exception as e:
            self.report({'ERROR'}, f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}
    
    def find_plane_intersections(self, bm, axis_idx, cut_position):
        """Find all points where edges intersect the cutting plane"""
        intersections = []
        tolerance = 1e-6
        
        for edge in bm.edges:
            v1, v2 = edge.verts
            coord1 = v1.co[axis_idx]
            coord2 = v2.co[axis_idx]
            
            # Check if edge crosses the cutting plane
            if ((coord1 <= cut_position <= coord2) or (coord2 <= cut_position <= coord1)):
                if abs(coord1 - coord2) > tolerance:
                    # Calculate intersection point using linear interpolation
                    t = (cut_position - coord1) / (coord2 - coord1)
                    intersection = v1.co.lerp(v2.co, t)
                    intersections.append(intersection)
                elif abs(coord1 - cut_position) < tolerance:
                    # Edge vertex is exactly on plane
                    intersections.append(v1.co.copy())
                elif abs(coord2 - cut_position) < tolerance:
                    intersections.append(v2.co.copy())
        
        # Remove duplicate points
        unique_intersections = []
        for point in intersections:
            is_duplicate = False
            for existing in unique_intersections:
                if (point - existing).length < tolerance * 10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_intersections.append(point)
        
        return unique_intersections
    
    def create_contour_loops(self, intersections, axis_idx, cut_position):
        """Organize intersection points into closed contour loops"""
        if len(intersections) < 3:
            return []
        
        # Get the two perpendicular axes
        axes = [0, 1, 2]
        axes.remove(axis_idx)
        u_axis, v_axis = axes
        
        # Calculate center point in the perpendicular plane
        center_u = sum(p[u_axis] for p in intersections) / len(intersections)
        center_v = sum(p[v_axis] for p in intersections) / len(intersections)
        
        # Sort points by angle around the center
        def angle_from_center(point):
            du = point[u_axis] - center_u
            dv = point[v_axis] - center_v
            return math.atan2(dv, du)
        
        sorted_points = sorted(intersections, key=angle_from_center)
        
        # For now, return as single loop (could be enhanced to detect multiple loops)
        return [sorted_points] if len(sorted_points) >= 3 else []
    
    def subdivide_loop(self, loop_points, target_subdivisions):
        """Convert a contour loop to evenly spaced subdivisions"""
        if len(loop_points) < 3:
            return None
        
        # Calculate total perimeter
        total_length = 0
        for i in range(len(loop_points)):
            p1 = loop_points[i]
            p2 = loop_points[(i + 1) % len(loop_points)]
            total_length += (p2 - p1).length
        
        if total_length < 1e-6:
            return None
        
        # Create evenly spaced points around the loop
        subdivided_points = []
        target_length = total_length / target_subdivisions
        
        current_pos = 0.0
        current_edge = 0
        
        for i in range(target_subdivisions):
            target_pos = i * target_length
            
            # Find which edge this position falls on
            edge_start_pos = 0.0
            for edge_idx in range(len(loop_points)):
                p1 = loop_points[edge_idx]
                p2 = loop_points[(edge_idx + 1) % len(loop_points)]
                edge_length = (p2 - p1).length
                edge_end_pos = edge_start_pos + edge_length
                
                if target_pos <= edge_end_pos or edge_idx == len(loop_points) - 1:
                    # Interpolate along this edge
                    if edge_length > 1e-6:
                        t = (target_pos - edge_start_pos) / edge_length
                        t = max(0.0, min(1.0, t))  # Clamp to [0,1]
                        point = p1.lerp(p2, t)
                    else:
                        point = p1.copy()
                    
                    subdivided_points.append(point)
                    break
                
                edge_start_pos = edge_end_pos
        
        return subdivided_points if len(subdivided_points) == target_subdivisions else None

class VIEW3D_PT_simple_retopo_main(Panel):
    """Main Simple Retopo Panel"""
    bl_label = "Simple Retopo"
    bl_idname = "VIEW3D_PT_simple_retopo_main" 
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Retopo"
    
    def draw(self, context):
        layout = self.layout
        
        obj = context.active_object
        if obj and obj.type == 'MESH' and context.mode == 'OBJECT':
            
            # Settings
            box = layout.box()
            col = box.column()
            col.label(text="Contour Settings:", icon='SETTINGS')
            
            # Axis selection  
            col.label(text="Cut Axis:")
            row = col.row(align=True)
            row.prop(context.scene, "retopo_axis", expand=True)
            
            col.separator()
            col.prop(context.scene, "retopo_cut_distance")
            col.prop(context.scene, "retopo_subdivisions") 
            col.prop(context.scene, "retopo_offset")
            
            layout.separator()
            
            # Main button
            col = layout.column()
            col.scale_y = 2.0
            col.operator("mesh.simple_retopo", text="Generate Contour Mesh", icon='MOD_REMESH')
            
            # Preview info
            box = layout.box()
            col = box.column()
            col.label(text=f"Object: {obj.name}")
            
            # Calculate estimated cuts
            if obj.data.vertices:
                axis_name = context.scene.retopo_axis
                axis_idx = {'X': 0, 'Y': 1, 'Z': 2}[axis_name]
                
                # Transform vertices to world space for accurate bounds
                coords = []
                for v in obj.data.vertices:
                    world_co = obj.matrix_world @ v.co
                    coords.append(world_co[axis_idx])
                
                if coords:
                    min_coord = min(coords)
                    max_coord = max(coords)
                    span = max_coord - min_coord
                    distance = context.scene.retopo_cut_distance
                    estimated_cuts = max(1, int(span / distance))
                    
                    col.label(text=f"Span: {span:.2f}")
                    col.label(text=f"Est. cuts: {estimated_cuts}")
                    col.label(text=f"Subdivs: {context.scene.retopo_subdivisions}")
            
        else:
            box = layout.box()
            col = box.column()
            col.label(text="Select a mesh object", icon='INFO')
            col.label(text="Switch to Object mode")

# Scene properties
def init_scene_props():
    bpy.types.Scene.retopo_axis = EnumProperty(
        name="Axis",
        items=[('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", "")],
        default='Z'
    )
    
    bpy.types.Scene.retopo_cut_distance = FloatProperty(
        name="Cut Distance",
        default=0.2,
        min=0.001,
        max=10.0,
        step=0.01,
        precision=3
    )
    
    bpy.types.Scene.retopo_subdivisions = IntProperty(
        name="Subdivisions",
        default=16, 
        min=4,
        max=1024
    )
    
    bpy.types.Scene.retopo_offset = FloatProperty(
        name="Surface Offset",
        default=0.001,
        min=0.0,
        max=0.1,
        step=0.001,
        precision=4
    )

classes = [
    MESH_OT_simple_retopo,
    VIEW3D_PT_simple_retopo_main,
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    init_scene_props()

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    
    props_to_remove = [
        'retopo_axis', 'retopo_cut_distance',
        'retopo_subdivisions', 'retopo_offset'
    ]
    
    for prop in props_to_remove:
        if hasattr(bpy.types.Scene, prop):
            delattr(bpy.types.Scene, prop)

if __name__ == "__main__":
    register()