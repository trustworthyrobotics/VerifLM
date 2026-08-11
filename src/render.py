import numpy as np
import open3d as o3d
import trimesh
import cv2
from open3d.geometry import TriangleMesh
from typing import Literal

def load_mesh(path):
    loaded = trimesh.load(path, force="scene", process=True)
    loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
    colors = np.asarray(loaded.visual.to_color().vertex_colors)

    mesh = TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(loaded.vertices))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(loaded.faces))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3] / 255.0)
    mesh.compute_vertex_normals()

    return mesh

def render_mesh(mesh: TriangleMesh, show_axes=True, axis_size=0.15):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(mesh)

    if show_axes:
        axes = TriangleMesh.create_coordinate_frame(size=0.15)
        vis.add_geometry(axes)

    vis.get_render_option().background_color = np.array([0,0,0])
    vis.run()
    vis.destroy_window()

def render_for_vlm(active_path, passive_path, output_path, T_passive_active=None, T_passive_world=None, 
                   rendered_frame: Literal["active", "world"]="active", lookat=[0,0,0], front=[0,1,0], up=[0,0,1], zoom=1, axis_size=0.15):
    active_mesh = load_mesh(active_path)
    passive_mesh = load_mesh(passive_path)

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720, visible=False)
    vis.add_geometry(passive_mesh)

    if T_passive_active is not None:
        active_mesh.transform(T_passive_active)
    vis.add_geometry(passive_mesh)
    vis.add_geometry(active_mesh)

    if rendered_frame == "active":
        active_axes = TriangleMesh.create_coordinate_frame(size=axis_size)
        active_axes.transform(T_passive_active)
        vis.add_geometry(active_axes)
    elif rendered_frame == "world":
        T_passive_world[0:3, 3] = np.array([-0.05, 0, 0.05])

        world_axes = TriangleMesh.create_coordinate_frame(size=axis_size)
        world_axes.transform(T_passive_world)
        vis.add_geometry(world_axes)

    opt = vis.get_render_option()
    opt.background_color = np.zeros(3)
    opt.mesh_show_back_face = True

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    # render image
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(str(output_path), do_render=True)
    vis.destroy_window()

    # add text labeling which color axis is which
    img = cv2.imread(str(output_path))
    cv2.putText(img, "Axes: X=Red  Y=Green  Z=Blue", (20, 35),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
    cv2.imwrite(str(output_path), img)

    
    