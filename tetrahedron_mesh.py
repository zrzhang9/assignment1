import numpy as np
import pytorch3d
import torch

from starter.render_360_video import generate_cameras_at_different_angles, render_meshes_with_cameras, save_images_as_gif
from starter.utils import get_device

def generate_tetrahedron_mesh() -> pytorch3d.structures.Meshes:
    """
    Generates a manually defined tetrahedron mesh.

    Returns:
        A tetrahedron mesh.
    """
    # Define the vertices.
    vertices = torch.tensor([
        [0.0, 0.0, 0.0], 
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]).unsqueeze(0)

    # Define the faces.
    faces = torch.tensor([
        [0, 1, 3],
        [0, 3, 2],
        [0, 1, 2],
        [1, 2, 3],
    ]).unsqueeze(0)

    # Define the texture.
    textures = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
    ]]).float()

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    return mesh

def render_tetrahedron_360_deg_mesh(mesh: pytorch3d.structures.Meshes) -> list[np.ndarray]:
    """
    Renders a tetrahedron mesh from 360 degrees and returns a list of rendered images.

    Args:
        mesh: The tetrahedron mesh to render.
    Returns:
        A list of rendered images.
    """

    # Generate the cameras.
    NUMBER_CAMERAS = 100
    DISTANCE_M = 3.0
    cameras = generate_cameras_at_different_angles(NUMBER_CAMERAS, DISTANCE_M)

    # Render the mesh with the cameras.
    rendered_images = render_meshes_with_cameras(mesh, cameras)
    return rendered_images

if __name__ == "__main__":
    # Render the tetrahedron mesh.
    mesh = generate_tetrahedron_mesh()
    mesh = mesh.to(get_device())
    rendered_images = render_tetrahedron_360_deg_mesh(mesh)

    # Save the images as a gif.
    OUTPUT_GIF_FILEPATH = "images/tetrahedron_360.gif"
    save_images_as_gif(rendered_images, OUTPUT_GIF_FILEPATH)