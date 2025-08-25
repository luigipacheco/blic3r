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

# --- Helper functions for robust cross-section generation ---

def _plane_basis_from_normal(n: Vector):
    """Create orthonormal basis for plane with given normal"""
    n = n.normalized()
    # pick a non-parallel vector
    tmp = Vector((1,0,0)) if abs(n.x) < 0.9 else Vector((0,1,0))
    u = (tmp.cross(n)).normalized()
    v = n.cross(u).normalized()
    return u, v, n

def _project_to_plane2d(points, plane_co, plane_no):
    """Project 3D points to 2D in the plane"""
    u, v, _ = _plane_basis_from_normal(plane_no)
    out = []
    for p in points:
        d = p - plane_co
        out.append(Vector((d.dot(u), d.dot(v), 0.0)))
    return out

def _signed_area_2d(points2d):
    """Calculate signed area of 2D polygon"""
    a = 0.0
    n = len(points2d)
    for i in range(n):
        x1, y1 = points2d[i].x, points2d[i].y
        x2, y2 = points2d[(i+1)%n].x, points2d[(i+1)%n].y
        a += x1*y2 - x2*y1
    return 0.5 * a

def _arc_length_resample(loop3d, target_n):
    """Resample loop to exactly target_n points by arc length"""
    # compute cumulative lengths
    n = len(loop3d)
    L = [0.0]
    total = 0.0
    for i in range(n):
        p1 = loop3d[i]
        p2 = loop3d[(i + 1) % n]
        seg = (p2 - p1).length
        total += seg
        L.append(total)
    if total < 1e-9:
        return None
    step = total / target_n
    resampled = []
    # walk along edges to place samples
    edge_i = 0
    p1 = loop3d[0]
    p2 = loop3d[1]
    seg_len = (p2 - p1).length
    seg_accum = 0.0
    target = 0.0
    for k in range(target_n):
        target = k * step
        # advance to segment containing target
        while L[edge_i+1] < target and edge_i < n-1:
            edge_i += 1
        # interpolate on edge edge_i
        local_start = L[edge_i]
        local_end = L[edge_i+1]
        t = 0.0 if (local_end - local_start) < 1e-9 else (target - local_start) / (local_end - local_start)
        a = loop3d[edge_i]
        b = loop3d[(edge_i + 1) % n]
        resampled.append(a.lerp(b, max(0.0, min(1.0, t))))
    return resampled

def _best_cyclic_alignment(A, B):
    """Return (offset, reversed) so that B shifted (and maybe reversed) matches A best"""
    n = len(A)
    if n != len(B):
        return 0, False
    # precompute for speed
    def err_for(offset, rev):
        e = 0.0
        if rev:
            for i in range(n):
                d = A[i] - B[(n - (i+offset)) % n]
                e += d.length_squared
        else:
            for i in range(n):
                d = A[i] - B[(i+offset) % n]
                e += d.length_squared
        return e
    best = (0, False)
    best_err = float('inf')
    for off in range(n):
        e0 = err_for(off, False)
        if e0 < best_err:
            best_err, best = e0, (off, False)
        e1 = err_for(off, True)
        if e1 < best_err:
            best_err, best = e1, (off, True)
    return best

def _slice_loops_from_bisect(bm_src, plane_co: Vector, plane_no: Vector, eps=1e-5):
    """Return list of loops; each loop is ordered list of Vector coords on plane."""
    bm = bm_src.copy()
    geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(
        bm, geom=geom, plane_co=plane_co, plane_no=plane_no,
        clear_outer=False, clear_inner=False, use_snap_center=False
    )
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    # Collect edges whose both verts lie on the plane
    def on_plane(v):
        return abs((v.co - plane_co).dot(plane_no)) < eps

    plane_edges = [e for e in bm.edges if on_plane(e.verts[0]) and on_plane(e.verts[1])]
    if not plane_edges:
        bm.free()
        return []

    # Build vertex adjacency on-plane
    adj = {}
    for e in plane_edges:
        a, b = e.verts[0], e.verts[1]
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    # Walk loops
    visited = set()
    loops = []
    for v_start in list(adj.keys()):
        if v_start in visited:
            continue
        # ensure degree 2 for loop; if not, still try to walk
        loop = []
        v = v_start
        prev = None
        while True:
            loop.append(v)
            visited.add(v)
            nbrs = adj.get(v, [])
            nxt = None
            # pick neighbor that isn't prev; prefer consistent traversal
            if len(nbrs) == 1:
                nxt = nbrs[0]
            else:
                for cand in nbrs:
                    if cand != prev:
                        nxt = cand
                        break
            if nxt is None:
                break
            prev, v = v, nxt
            if v == v_start:
                break
        # only keep closed loops with length >= 3
        if len(loop) >= 3 and loop[-1] == loop[0]:
            loop = loop[:-1]
        if len(loop) >= 3:
            loops.append([vv.co.copy() for vv in loop])

    bm.free()
    return loops

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
            
            # Get axis info and create plane normal
            axis_map = {'X': 0, 'Y': 1, 'Z': 2}
            axis_idx = axis_map[self.axis]
            plane_no = Vector((0,0,0))
            plane_no[axis_idx] = 1.0
            
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
            
            # Generate contour curves for each cutting plane using robust bisect method
            all_loops = []
            for i, cut_pos in enumerate(cut_positions):
                print(f"Processing cut {i+1}/{len(cut_positions)} at {cut_pos:.3f}")
                
                # Create plane for this cut
                plane_co = Vector((0,0,0))
                plane_co[axis_idx] = cut_pos
                
                # Get robust slice loops using bmesh.ops.bisect_plane
                loops = _slice_loops_from_bisect(bm, plane_co, plane_no, eps=1e-5)
                if not loops:
                    print(f"  No valid loops at this cut")
                    continue
                
                print(f"  Found {len(loops)} loops")
                
                # Process each loop
                for loop_idx, loop in enumerate(loops):
                    # Enforce consistent winding (CCW looking along plane_no)
                    pts2d = _project_to_plane2d(loop, plane_co, plane_no)
                    if _signed_area_2d(pts2d) < 0.0:
                        loop.reverse()
                        print(f"    Flipped loop {loop_idx} for consistent winding")
                    
                    # Resample to fixed subdivisions
                    loop_res = _arc_length_resample(loop, self.subdivisions)
                    if not loop_res:
                        print(f"    Failed to resample loop {loop_idx}")
                        continue
                    
                    # Apply surface offset along plane normal
                    loop_res = [p + plane_no * self.offset for p in loop_res]
                    
                    # Add vertices to new bmesh
                    bm_verts = [new_bm.verts.new(p) for p in loop_res]
                    all_loops.append(bm_verts)
                    print(f"    Created loop with {len(loop_res)} vertices")
            
            print(f"Total valid loops: {len(all_loops)}")
            
            if len(all_loops) < 2:
                self.report({'ERROR'}, "Not enough valid contour loops")
                bm.free()
                new_bm.free()
                return {'CANCELLED'}
            
            # Update vertex indices
            new_bm.verts.ensure_lookup_table()
            
            # Create quad faces between consecutive loops with cyclic alignment
            faces_created = 0
            for i in range(len(all_loops) - 1):
                loop1 = all_loops[i]
                loop2 = all_loops[i + 1]
                
                # Only connect if both loops have same subdivision count
                if len(loop1) != self.subdivisions or len(loop2) != self.subdivisions:
                    continue
                
                # Compute best cyclic alignment on coordinates
                A = [v.co for v in loop1]
                B = [v.co for v in loop2]
                off, rev = _best_cyclic_alignment(A, B)
                
                # Build reindexed view of loop2
                if rev:
                    B_vs = [loop2[(len(loop2) - (j+off)) % len(loop2)] for j in range(len(loop2))]
                else:
                    B_vs = [loop2[(j+off) % len(loop2)] for j in range(len(loop2))]
                
                # Create quad faces
                for j in range(self.subdivisions):
                    v1 = loop1[j]
                    v2 = loop1[(j + 1) % self.subdivisions]
                    v3 = B_vs[(j + 1) % self.subdivisions]
                    v4 = B_vs[j]
                    
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