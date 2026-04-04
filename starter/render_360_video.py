import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer

def render_meshes_with_cameras(mesh: pytorch3d.structures.Meshes, cameras: pytorch3d.renderer.FoVPerspectiveCameras) -> list[np.ndarray]:
    """
    Renders a mesh with a list of cameras at different poses and return a list of images.
    
    Args:
        mesh: The mesh to render.
        cameras: The cameras to render the mesh with.
    Returns:
        A list of images.
    """
    # Create a renderer for the meshes.
    renderer = get_mesh_renderer(image_size=512)

    # Extend the single mesh to a batch of meshes to be able to render with all cameras.
    meshes = mesh.extend(len(cameras))
    rendered_meshes = renderer(meshes, cameras=cameras)

    # Iterate over the rendered meshes and extract the images.
    rendered_images = []
    for rendered_mesh in rendered_meshes:
        image = rendered_mesh[..., :3].cpu().numpy()
        rendered_images.append((image * 255).clip(0, 255).astype(np.uint8))
    
    return rendered_images

def generate_cameras_at_different_angles(number_cameras: int, distance: float) -> list[pytorch3d.renderer.FoVPerspectiveCameras]:
    """
    Generates a list of cameras at different angles.
    Args:
        number_cameras: The number of cameras to generate.
        distance: The distance from the camera to the mesh.
        azimuth_resolution_deg: The resolution of the azimuth angle in degrees.
    Returns:
        A list of cameras.
    """
    # Generate the azimuth angles.
    START_AZIMUTH_DEG = 0
    END_AZIMUTH_DEG = 360
    azimuth_angles_deg = torch.linspace(START_AZIMUTH_DEG, END_AZIMUTH_DEG, number_cameras)

    # Generate the cameras at different azimuth angles.
    ELEVATION_DEG = 2.0
    batch_camera_SO3_world, batch_camera_t_world = pytorch3d.renderer.look_at_view_transform(dist=distance, elev=ELEVATION_DEG, azim=azimuth_angles_deg)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=batch_camera_SO3_world, T=batch_camera_t_world, device=get_device())

    return cameras

def generate_360_images(mesh: pytorch3d.structures.Meshes, number_cameras: int, distance: float) -> list[np.ndarray]:
    """
    Generates a list of images of a mesh at different angles.
    Args:
        mesh: The mesh to generate images of.
        number_cameras: The number of cameras to generate.
        distance: The distance from the camera to the mesh.
    Returns:
        A list of images.
    """
    # Generate the cameras.
    cameras = generate_cameras_at_different_angles(number_cameras, distance)

    # Render the mesh with the cameras.
    rendered_images = render_meshes_with_cameras(mesh, cameras)
    return rendered_images

def save_images_as_gif(images: list[np.ndarray], output_filepath: str) -> None:
    """
    Saves a list of images as a gif.
    Args:
        images: The list of images to save.
        duration: The duration of the gif in milliseconds.
        output_filepath: The path to save the gif to.
    Returns:
        None.
    """
    MILLISECONDS_PER_SECOND = 1000
    FRAMES_PER_SECOND = 15
    duration = MILLISECONDS_PER_SECOND // FRAMES_PER_SECOND 
    imageio.mimsave(output_filepath, images, duration=duration)

if __name__ == "__main__":
    # Load the mesh.
    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow.obj"])
    mesh = mesh.to(get_device())

    # Generate the images.
    NUMBER_CAMERAS = 100
    DISTANCE_M = 3.0
    rendered_images = generate_360_images(mesh, NUMBER_CAMERAS, DISTANCE_M)

    # Save the images as a gif.
    OUTPUT_GIF_FILEPATH = "images/cow_360.gif"
    save_images_as_gif(rendered_images, OUTPUT_GIF_FILEPATH)