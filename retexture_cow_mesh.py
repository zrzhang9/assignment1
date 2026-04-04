import numpy as np
import pytorch3d
import torch

from starter.render_360_video import generate_cameras_at_different_angles, render_meshes_with_cameras, save_images_as_gif
from starter.utils import get_device, load_cow_mesh

def get_linear_texture(vertices: torch.Tensor, color1: torch.Tensor, color2: torch.Tensor) -> torch.Tensor:
    """
    Returns a texture tensor (linearly interpolated between color1 and color2) for the vertices of a mesh.

    Args:
        vertices: A tensor of shape (N, 3) where N is the number of vertices in the mesh.
        color1: The color to assign to the front of the cow.
        color2: The color to assign to the back of the cow.

    Returns:
        A tensor of shape (N, 3) where N is the number of vertices in the mesh.
    """

    # Get the range of z-coordinates of the vertices.
    print(f"Vertices: {vertices.shape}")
    z_min = vertices[:, :, 2].min()
    z_max = vertices[:, :, 2].max()
    print(f"z_min: {z_min}, z_max: {z_max}")

    # Get the linear interpolation of the colors based on the z-coordinates.
    alpha = (vertices[:, :, 2] - z_min) / (z_max - z_min)
    print(f"Alpha: {alpha.shape}")
    textures = alpha.unsqueeze(-1) * color2 + (1 - alpha.unsqueeze(-1)) * color1
    print(f"Textures: {textures.shape}")

    return textures

def retexture_cow_mesh(cow_mesh_vertices: torch.Tensor, cow_mesh_faces: torch.Tensor, color1: torch.Tensor, color2: torch.Tensor) -> pytorch3d.structures.Meshes:
    """
    Re-textures a cow mesh with a linear interpolation between color1 and color2.

    Args:
        cow_mesh_vertices: The vertices of the cow mesh.
        cow_mesh_faces: The faces of the cow mesh.
        color1: The color to assign to the front of the cow.
        color2: The color to assign to the back of the cow.

    Returns:
        A re-textured cow mesh.
    """
    print(f"Cow mesh vertices: {cow_mesh_vertices.shape}")
    print(f"Cow mesh faces: {cow_mesh_faces.shape}")
    print(f"Color1: {color1}")
    print(f"Color2: {color2}")
    print(f"Linear texture: {get_linear_texture(cow_mesh_vertices, color1, color2).shape}")
    retextured_mesh = pytorch3d.structures.Meshes(
        verts=cow_mesh_vertices,
        faces=cow_mesh_faces,
        textures=pytorch3d.renderer.TexturesVertex(get_linear_texture(cow_mesh_vertices, color1, color2)),
    )
    return retextured_mesh

def render_retextured_cow_mesh(retextured_mesh: pytorch3d.structures.Meshes) -> list[np.ndarray]:
    """
    Renders a re-textured cow mesh.

    Args:
        retextured_mesh: The re-textured cow mesh.

    Returns:
        A list of images.
    """
    # Generate the cameras. 
    NUMBER_CAMERAS = 100
    DISTANCE_M = 3.0
    cameras = generate_cameras_at_different_angles(NUMBER_CAMERAS, DISTANCE_M)
    
    return render_meshes_with_cameras(retextured_mesh, cameras)

if __name__ == "__main__":
    # Load the cow mesh.
    COW_MESH_PATH = "data/cow.obj"
    cow_mesh_vertices, cow_mesh_faces = load_cow_mesh(COW_MESH_PATH)

    # Add the batch dimension to the vertices and faces.
    cow_mesh_vertices = cow_mesh_vertices.unsqueeze(0)
    cow_mesh_faces = cow_mesh_faces.unsqueeze(0)

    # Define the colors. 
    color1 = torch.tensor([1.0, 0.0, 1.0])
    color2 = torch.tensor([1.0, 1.0, 0.0])

    # Re-texture the cow mesh.
    retextured_mesh = retexture_cow_mesh(cow_mesh_vertices, cow_mesh_faces, color1, color2)
    retextured_mesh = retextured_mesh.to(get_device())

    # Render the re-textured cow mesh.
    rendered_images = render_retextured_cow_mesh(retextured_mesh)

    # Save the images as a gif.
    OUTPUT_GIF_FILEPATH = "images/retextured_cow.gif"
    save_images_as_gif(rendered_images, OUTPUT_GIF_FILEPATH)