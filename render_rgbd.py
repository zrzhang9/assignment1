import numpy as np
import pytorch3d
import torch

from starter.render_360_video import generate_cameras_at_different_angles, render_meshes_with_cameras, save_images_as_gif
from starter.utils import get_device, get_points_renderer, unproject_depth_image
from starter.render_generic import load_rgbd_data

def render_point_cloud(point_cloud: pytorch3d.structures.Pointclouds, cameras: pytorch3d.renderer.FoVPerspectiveCameras, image_size: int = 256, background_color: tuple[float, float, float] = (1, 1, 1)) -> torch.Tensor:
    """
    Renders a point cloud with cameras at different poses.

    Args:
        point_cloud: The point cloud to render.
        cameras: The cameras to render the point cloud with.
        image_size: The size of the image to render.
        background_color: The background color of the image.
    Returns:
        A list of images.
    """
    point_cloud_renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    point_cloud = point_cloud.extend(len(cameras))
    rendered_point_clouds = point_cloud_renderer(point_cloud, cameras=cameras)
    rendered_point_clouds = rendered_point_clouds.cpu().numpy()[..., :3]
    rendered_point_clouds = (rendered_point_clouds * 255).clip(0, 255).astype(np.uint8)
    return rendered_point_clouds

def rgbd_to_point_cloud(rgbd_data: dict[str, np.ndarray]) -> pytorch3d.structures.Pointclouds:
    """
    Converts RGB-D data to a point cloud.

    Args:
        rgbd_data: The RGB-D data to convert.
    Returns:
        A point cloud.
    """
    # Unproject the depth image into a point cloud.
    points_in_world, rgba = unproject_depth_image(torch.tensor(rgbd_data["rgb"]), torch.tensor(rgbd_data["mask"]), torch.tensor(rgbd_data["depth"]), rgbd_data["cameras"])

    # Construct the point cloud object.
    point_clouds = pytorch3d.structures.Pointclouds(points=[points_in_world.to(get_device())], features=[rgba.to(get_device())])

    return point_clouds

def render_rgbd(rgbd_data: dict[str, np.ndarray], cameras: pytorch3d.renderer.FoVPerspectiveCameras) -> list[np.ndarray]:
    """
    Renders a point cloud from RGB-D data.

    Args:
        rgbd_data: The RGB-D data to render.
        cameras: The cameras to render the point cloud with.
    Returns:
        A list of images.
    """
    # Convert the RGB-D data to a point cloud.
    point_clouds = rgbd_to_point_cloud(rgbd_data)

    # Render the point cloud with the cameras.
    rendered_point_clouds = render_point_cloud(point_clouds, cameras)

    return rendered_point_clouds

if __name__ == "__main__":
    # Load the RGB-D image data.
    rgbd_image_data = load_rgbd_data("data/rgbd_data.pkl")

    # Construct dictionaries for the image data.
    rgbd_image_data1 = {
        "rgb": rgbd_image_data["rgb1"],
        "mask": rgbd_image_data["mask1"],
        "depth": rgbd_image_data["depth1"],
        "cameras": rgbd_image_data["cameras1"]
    }
    rgbd_image_data2 = {
        "rgb": rgbd_image_data["rgb2"],
        "mask": rgbd_image_data["mask2"],
        "depth": rgbd_image_data["depth2"],
        "cameras": rgbd_image_data["cameras2"]
    }
    # rgbd_image_data_all = {
    #     "rgb": torch.cat([])

    # Generate the cameras at different viewing angles.
    NUMBER_CAMERAS = 100
    DISTANCE_M = 6.0
    cameras = generate_cameras_at_different_angles(NUMBER_CAMERAS, DISTANCE_M)

    # Render the RGB-D image data into point clouds.
    rendered_point_clouds1 = render_rgbd(rgbd_image_data1, cameras)
    rendered_point_clouds2 = render_rgbd(rgbd_image_data2, cameras)

    # Combine the two point clouds to get the union of the two.
    point_clouds1 = rgbd_to_point_cloud(rgbd_image_data1)
    point_clouds2 = rgbd_to_point_cloud(rgbd_image_data2)
    all_points = torch.cat([point_clouds1.points_packed(), point_clouds2.points_packed()], dim=0)
    all_features = torch.cat([point_clouds1.features_packed(), point_clouds2.features_packed()], dim=0)
    all_point_clouds = pytorch3d.structures.Pointclouds(points=[all_points], features=[all_features])
    rendered_all_point_clouds = render_point_cloud(all_point_clouds, cameras)

    # Save the rendered point clouds as a GIF.
    OUTPUT_GIF_FILEPATH1 = "images/rendered_point_clouds1.gif"
    OUTPUT_GIF_FILEPATH2 = "images/rendered_point_clouds2.gif"
    OUTPUT_GIF_FILEPATH_ALL_POINT_CLOUDS = "images/all_rendered_point_clouds.gif"
    save_images_as_gif(rendered_point_clouds1, OUTPUT_GIF_FILEPATH1)
    save_images_as_gif(rendered_point_clouds2, OUTPUT_GIF_FILEPATH2)
    save_images_as_gif(rendered_all_point_clouds, OUTPUT_GIF_FILEPATH_ALL_POINT_CLOUDS)