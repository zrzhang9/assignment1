import mcubes
import numpy as np
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, get_points_renderer
from starter.render_360_video import generate_cameras_at_different_angles, save_images_as_gif

def generate_parametric_torus_point_cloud(device: torch.device = None) -> pytorch3d.structures.Pointclouds:
    """
    Generates a torus point cloud.

    Returns:
        A torus point cloud.
    """
    if device is None:
        device = get_device()
    
    # Generate the samples of the parameters.
    R = 5.0
    r = 2.0
    phi_samples = torch.linspace(0, 2 * np.pi, 100)
    theta_samples = torch.linspace(0, 2 * np.pi, 100)

    # Generate the samples with the meshgrid.
    phi, theta = torch.meshgrid(phi_samples, theta_samples)

    # Generate the x, y and z coordinates for the torus.
    x = (R + r * torch.sin(theta)) * torch.cos(phi)
    y = (R + r * torch.sin(theta)) * torch.sin(phi)
    z = r * torch.cos(theta)
    
    # Stack the coordinates into a single points tensor.
    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)

    # Generate the colors for the torus.
    colors = (points - points.min()) / (points.max() - points.min())

    # Create the point cloud.
    point_clouds = pytorch3d.structures.Pointclouds(
        points=[points], features=[colors],
    ).to(device)

    return point_clouds

def generate_implicit_torus_mesh(device: torch.device = None) -> pytorch3d.structures.Meshes:
    """
    Generates an implicit torus mesh.

    Returns:
        An implicit torus mesh.
    """
    if device is None:
        device = get_device()
    
    # Define the range of the x, y and z coordinates.
    MIN_VALUE = -10.0
    MAX_VALUE = 10.0
    VOXEL_SIZE = 64
    R = 3.0
    r = 1.5
    
    # Generate the x, y and z samples with the meshgrid.
    x, y, z = torch.meshgrid([torch.linspace(MIN_VALUE, MAX_VALUE, VOXEL_SIZE)] * 3)
    voxels = (torch.sqrt(x ** 2 + y ** 2) - R) ** 2 + z ** 2 - r ** 2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    vertices = (vertices / VOXEL_SIZE) * (MAX_VALUE - MIN_VALUE) + MIN_VALUE

    faces = torch.tensor(faces.astype(int))
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    print(vertices.shape)
    print(faces.shape)
    mesh = pytorch3d.structures.Meshes(
        verts=[vertices],
        faces=[faces],
        textures=textures,
    ).to(device)

    return mesh


def render_torus_point_cloud(point_clouds: pytorch3d.structures.Pointclouds, device: torch.device = None) -> list[np.ndarray]:
    """
    Renders a torus point cloud.

    Args:
        point_clouds: The point cloud to render.
        device: The device to render the point cloud on.
    Returns:
        A list of rendered point clouds.
    """
    if device is None:
        device = get_device()
    
    # Create the renderer.
    renderer = get_points_renderer(image_size=256, device=device)

    # Create the cameras.
    NUMBER_CAMERAS = 100
    DISTANCE_M = 20
    cameras = generate_cameras_at_different_angles(NUMBER_CAMERAS, DISTANCE_M)

    # Extend the point clouds with a batch dimension.
    point_clouds = point_clouds.extend(len(cameras))

    # Render the point clouds.
    batch_rendered_point_clouds = renderer(point_clouds, cameras=cameras)
    rendered_point_clouds = batch_rendered_point_clouds.cpu().numpy()[..., :3]
    rendered_point_clouds = (rendered_point_clouds * 255).clip(0, 255).astype(np.uint8)

    return rendered_point_clouds\
    
def render_mesh(mesh: pytorch3d.structures.Meshes, device: torch.device = None) -> list[np.ndarray]:
    """
    Renders a mesh.

    Args:
        mesh: The mesh to render.
        device: The device to render the mesh on.
    Returns:
        A list of rendered meshes.
    """
    if device is None:
        device = get_device()

    # Construct lights.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
    
    # Create the renderer.
    renderer = get_mesh_renderer(image_size=256, device=device)

    # Create the cameras.
    NUMBER_CAMERAS = 100
    DISTANCE_M = 20
    cameras = generate_cameras_at_different_angles(NUMBER_CAMERAS, DISTANCE_M)

    # Render the mesh.
    mesh = mesh.extend(len(cameras))
    batch_rendered_meshes = renderer(mesh, cameras=cameras, lights=lights)
    rendered_meshes = batch_rendered_meshes.cpu().numpy()[..., :3]
    rendered_meshes = (rendered_meshes * 255).clip(0, 255).astype(np.uint8)

    return rendered_meshes

if __name__ == "__main__":
    # Generate the torus point cloud.
    torus_point_clouds = generate_parametric_torus_point_cloud()

    # Render the point clouds in different viewing angles.
    rendered_point_clouds = render_torus_point_cloud(torus_point_clouds)

    # Save the rendered point clouds as a GIF.
    POINT_CLOUDS_GIF_FILEPATH = "images/rendered_torus_point_clouds.gif"
    save_images_as_gif(rendered_point_clouds, POINT_CLOUDS_GIF_FILEPATH)

    # Generate the implicit torus mesh.
    implicit_torus_mesh = generate_implicit_torus_mesh()

    # Render the mesh in different viewing angles.
    rendered_implicit_torus_mesh = render_mesh(implicit_torus_mesh)

    # Save the rendered implicit torus mesh as a GIF.
    MESH_GIF_FILEPATH = "images/rendered_implicit_torus_mesh.gif"
    save_images_as_gif(rendered_implicit_torus_mesh, MESH_GIF_FILEPATH)