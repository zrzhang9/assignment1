"""
Usage:
    python -m starter.camera_transforms --image_size 512
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
from scipy.spatial.transform import Rotation

from starter.utils import get_device, get_mesh_renderer


def render_textured_cow(
    cow_path="data/cow.obj",
    image_size=256,
    R_relative=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    T_relative=[0, 0, 0],
    device=None,
):
    if device is None:
        device = get_device()
    meshes = pytorch3d.io.load_objs_as_meshes([cow_path]).to(device)
    R_relative = torch.tensor(R_relative).float()
    T_relative = torch.tensor(T_relative).float()
    R = R_relative @ torch.tensor([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    T = R_relative @ torch.tensor([0.0, 0, 3]) + T_relative
    renderer = get_mesh_renderer(image_size=256)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=R.unsqueeze(0), T=T.unsqueeze(0), device=device,
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -3.0]], device=device,)
    rend = renderer(meshes, cameras=cameras, lights=lights)
    return rend[0, ..., :3].cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--output_path", type=str, default="images/textured_cow_rotation_1.jpg")
    args = parser.parse_args()
    world_SO3_camera1 = Rotation.from_euler('z', -90, degrees=True)
    camera_t_camera2 = [0, 0, 2.0]
    camera_SO3_camera3 = Rotation.from_euler('y', 20, degrees=True)
    camera_t_camera3 = [-0.4, -0.2, -0.05]
    camera_SO3_camera4 = Rotation.from_euler('y', 90, degrees=True)
    COW1_FILEPATH = "images/textured_cow_transformation_1.jpg"
    COW2_FILEPATH = "images/textured_cow_transformation_2.jpg"
    COW3_FILEPATH = "images/textured_cow_transformation_3.jpg"
    COW4_FILEPATH = "images/textured_cow_transformation_4.jpg"
    plt.imsave(COW1_FILEPATH, render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, R_relative=camera_SO3_camera1.as_matrix(), T_relative=[0, 0, 0]))
    plt.imsave(COW2_FILEPATH, render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, T_relative=camera_t_camera2))
    plt.imsave(COW3_FILEPATH, render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, R_relative=camera_SO3_camera3.as_matrix(), T_relative=camera_t_camera3))
    plt.imsave(COW4_FILEPATH, render_textured_cow(cow_path=args.cow_path, image_size=args.image_size, R_relative=camera_SO3_camera4.as_matrix(), T_relative=[-3.0, 0, 2.5]))