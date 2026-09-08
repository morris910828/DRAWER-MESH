# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import sys
sys.path.append("./")
import json
import os
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
# import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    output_dir: Path
    """Path to the output directory."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        ray_bundle = RayBundle(
            origins=origins, directions=directions, pixel_area=pixel_area, camera_indices=camera_indices
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        if self.save_world_frame:
            # apply the inverse dataparser transform to the point cloud
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # TODO: Make this work with Density Field
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert self.resolution % 512 == 0, f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    filename: str = "splat.ply"

    @staticmethod
    def write_ply(
        filename: str,
        count: int,
        map_to_tensors: typing.OrderedDict[str, np.ndarray],
        mesh_verts: Optional[np.ndarray] = None,
        mesh_faces: Optional[np.ndarray] = None,
    ):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Optionally appends mesh vertices (element mesh_vertex) and mesh faces (element mesh_face).
        Note: All float values will be converted to float32 for writing.

        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        mesh_verts (np.ndarray, optional): shape (V, 3) float32 mesh vertex positions.
        mesh_faces (np.ndarray, optional): shape (F, 3) int32 mesh face vertex indices.
        """

        # Ensure count matches the length of all tensors
        if not all(len(tensor) == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float, uint8, or int32 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8 or tensor.dtype == np.int32)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float, uint8, or int32 type and not empty")

        with open(filename, "wb") as ply_file:
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")

            ply_file.write(f"element vertex {count}\n".encode())

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                if tensor.dtype.kind == "f":
                    data_type = "float"
                elif tensor.dtype == np.int32:
                    data_type = "int"
                else:
                    data_type = "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            if mesh_verts is not None:
                ply_file.write(f"element mesh_vertex {mesh_verts.shape[0]}\n".encode())
                ply_file.write(b"property float x\n")
                ply_file.write(b"property float y\n")
                ply_file.write(b"property float z\n")

            if mesh_faces is not None:
                ply_file.write(f"element mesh_face {mesh_faces.shape[0]}\n".encode())
                ply_file.write(b"property list uchar int vertex_indices\n")

            ply_file.write(b"end_header\n")

            # Write Gaussian vertex data
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.int32:
                        ply_file.write(np.int32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

            # Write mesh vertices
            if mesh_verts is not None:
                ply_file.write(mesh_verts.astype(np.float32).tobytes())

            # Write mesh faces (each face: 1 byte count=3, then 3 int32 indices)
            if mesh_faces is not None:
                faces_int32 = mesh_faces.astype(np.int32)
                for face in faces_int32:
                    ply_file.write(np.uint8(3).tobytes())
                    ply_file.write(face.tobytes())

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # assert isinstance(pipeline.model, SplatfactoModel)

        model = pipeline.model

        filename = self.output_dir / self.filename

        count = 0
        map_to_tensors = OrderedDict()

        with torch.no_grad():
            # Capture scales/opacities BEFORE finalize_face_assignment() below: their
            # clamp bounds key off gaussians_to_mesh_indices, so reading them after
            # relabeling would reclamp an already-trained Gaussian to a new face's
            # (possibly smaller) bound for no reason other than the relabel itself.
            # See splatfacto_on_mesh_uc.py's export_splatfacto_on_mesh() for the same
            # discipline.
            scales = model.scales.data.cpu().numpy()
            opacities = model.opacities.data.cpu().numpy()

            if hasattr(model, "finalize_face_assignment"):
                model.finalize_face_assignment()

            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            n = count
            CONSOLE.print(f"Exporting {count} Gaussian splats")
            map_to_tensors["x"] = positions[:, 0]
            map_to_tensors["y"] = positions[:, 1]
            map_to_tensors["z"] = positions[:, 2]
            if hasattr(model, "normals") and hasattr(model, "gaussians_to_mesh_indices"):
                normals_np = model.normals[model.gaussians_to_mesh_indices].cpu().numpy().astype(np.float32)
                map_to_tensors["nx"] = normals_np[:, 0]
                map_to_tensors["ny"] = normals_np[:, 1]
                map_to_tensors["nz"] = normals_np[:, 2]
            else:
                map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
                map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
                map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

            if model.config.sh_degree > 0:
                shs_0 = model.shs_0.contiguous().cpu().numpy()
                for i in range(shs_0.shape[1]):
                    map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

                # transpose(1, 2) was needed to match the sh order in Inria version
                shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
                shs_rest = shs_rest.reshape((n, -1))
                for i in range(shs_rest.shape[-1]):
                    map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]
            else:
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                map_to_tensors["colors"] = (colors * 255).astype(np.uint8)

            map_to_tensors["opacity"] = opacities

            # Clamp log-scale: degenerate triangles (nearly collinear verts) have
            # circumradius → ∞, which lets scale_limit = upper_scale * xyz_radius
            # reach 1e8+, crashing the viewer's CUDA rasterizer.
            # Cap at 3× the 99th-percentile face radius to preserve all normal faces.
            _r99 = float(np.percentile(model.xyz_radius[:, 0].cpu().numpy(), 99))
            _max_log_scale = np.log(model.config.upper_scale * _r99 * 3.0 + 1e-20)
            scales = np.minimum(scales, _max_log_scale)
            for i in range(3):
                map_to_tensors[f"scale_{i}"] = scales[:, i, None]

            quats = model.quats.data.cpu().numpy()
            for i in range(4):
                map_to_tensors[f"rot_{i}"] = quats[:, i, None]

            if hasattr(model, "gaussians_to_mesh_indices"):
                map_to_tensors["mesh_face_idx"] = model.gaussians_to_mesh_indices.cpu().numpy().astype(np.int32)

            # Vertex-anchored coverage-filler Gaussians (2026-07-26, see
            # splatfacto_on_mesh_uc.py's populate_modules/get_outputs): a separate
            # population, one per mesh vertex, not tied to any single face. Appended
            # here so they actually show up in the exported PLY -- without this, the
            # whole mechanism trains but is silently dropped at export time, since
            # everything above only ever reads the face-based population.
            # `is not None`, not just hasattr: with config.use_vertex_layer off the model
            # sets vertex_gauss_params to None rather than omitting the attribute, so
            # hasattr alone still passes and this block would then index into None.
            if (
                hasattr(model, "vertex_positions")
                and getattr(model, "vertex_gauss_params", None) is not None
            ):
                v_positions = model.vertex_positions.cpu().numpy()
                v_count = v_positions.shape[0]
                map_to_tensors["x"] = np.concatenate([map_to_tensors["x"], v_positions[:, 0]])
                map_to_tensors["y"] = np.concatenate([map_to_tensors["y"], v_positions[:, 1]])
                map_to_tensors["z"] = np.concatenate([map_to_tensors["z"], v_positions[:, 2]])
                v_normals = model.vertex_normals.cpu().numpy().astype(np.float32)
                map_to_tensors["nx"] = np.concatenate([map_to_tensors["nx"], v_normals[:, 0]])
                map_to_tensors["ny"] = np.concatenate([map_to_tensors["ny"], v_normals[:, 1]])
                map_to_tensors["nz"] = np.concatenate([map_to_tensors["nz"], v_normals[:, 2]])

                if model.config.sh_degree > 0:
                    v_shs_0 = model.vertex_gauss_params["features_dc"].data.contiguous().cpu().numpy()
                    for i in range(v_shs_0.shape[1]):
                        map_to_tensors[f"f_dc_{i}"] = np.concatenate(
                            [map_to_tensors[f"f_dc_{i}"], v_shs_0[:, i, None]]
                        )
                    v_shs_rest = (
                        model.vertex_gauss_params["features_rest"].data.transpose(1, 2).contiguous().cpu().numpy()
                    )
                    v_shs_rest = v_shs_rest.reshape((v_count, -1))
                    for i in range(v_shs_rest.shape[-1]):
                        map_to_tensors[f"f_rest_{i}"] = np.concatenate(
                            [map_to_tensors[f"f_rest_{i}"], v_shs_rest[:, i, None]]
                        )
                else:
                    v_colors = torch.clamp(
                        torch.sigmoid(model.vertex_gauss_params["features_dc"].data), 0.0, 1.0
                    ).cpu().numpy()
                    map_to_tensors["colors"] = np.concatenate(
                        [map_to_tensors["colors"], (v_colors * 255).astype(np.uint8)]
                    )

                v_opacities = model.vertex_gauss_params["opacities"].data.cpu().numpy()
                map_to_tensors["opacity"] = np.concatenate([map_to_tensors["opacity"], v_opacities])

                # _vertex_scales()/_vertex_quats() (2026-07-31): the vertex layer's scale
                # and rotation are now trainable-within-bounds, not the fixed
                # vertex_log_scales/vertex_quats reference tensors -- reading those directly
                # would silently export the pre-training values and throw away everything
                # the optimizer learned, same class of bug as the original scales/opacities
                # export mistake this file already documents fixing elsewhere.
                v_scales = model._vertex_scales().detach().cpu().numpy()
                # The vertex layer gets its OWN cap, not the face-based _max_log_scale
                # above. That one is 3x the 99th percentile of upper_scale * xyz_radius --
                # 3x what a FACE-based row may legitimately reach. A vertex row is bounded
                # by exp(vertex_log_scales) instead, the vertex's 1-ring reach, which spans
                # several faces and is legitimately larger than any one face's radius, so
                # the face-derived cap does not describe a degenerate value here: it
                # describes an ordinary one.
                #
                # Measured on table_gs11 before this fix: the face-derived cap sat at
                # 0.00622 while the vertex layer's own ceiling had a median of 0.01025, so
                # the shared cap clipped 89.5% of the layer. Worse, it clipped BOTH
                # in-plane axes on 56.9% of it, which writes sx == sy and exports those
                # rows as exact circles no matter what ellipse they render as -- the ply
                # reported aspect median 1.000 with 57.3% circles for a layer whose true
                # rendered aspect (read back from the checkpoint) is 1.2055 with 0.2%
                # circles. Every viewer and every downstream analysis inherited that, and
                # it read as "the vertex Gaussians are all circles" in exactly the way an
                # unfixed anisotropy bug would.
                #
                # Same 3x-of-p99 rule and same purpose (keep a degenerate 1-ring from
                # reaching 1e8 and crashing the viewer's rasterizer), just applied to the
                # bound that actually governs this population.
                _v99 = float(np.percentile(torch.exp(model.vertex_log_scales[:, :2]).cpu().numpy(), 99))
                _max_log_scale_v = np.log(_v99 * 3.0 + 1e-20)
                v_scales = np.minimum(v_scales, _max_log_scale_v)
                for i in range(3):
                    map_to_tensors[f"scale_{i}"] = np.concatenate(
                        [map_to_tensors[f"scale_{i}"], v_scales[:, i, None]]
                    )

                v_quats = model._vertex_quats().detach().cpu().numpy()
                for i in range(4):
                    map_to_tensors[f"rot_{i}"] = np.concatenate([map_to_tensors[f"rot_{i}"], v_quats[:, i, None]])

                # No single owning face -- honestly marked -1 (not an arbitrary
                # incident face) rather than implying a face ownership that doesn't
                # exist. Downstream tooling that assumes mesh_face_idx is always a
                # valid face index needs updating to handle -1, not the other way
                # around.
                map_to_tensors["mesh_face_idx"] = np.concatenate(
                    [map_to_tensors["mesh_face_idx"], np.full(v_count, -1, dtype=np.int32)]
                )

                n = n + v_count
                count = n
                CONSOLE.print(f"Exporting {v_count} additional vertex-anchored Gaussians ({n} total)")

            # Face-centroid coverage-filler Gaussians (2026-07-27, see
            # splatfacto_on_mesh_uc.py's populate_modules/get_outputs): same reasoning
            # as the vertex-anchored block above -- without this the mechanism trains
            # but is silently dropped at export time.
            if hasattr(model, "centroid_positions") and hasattr(model, "centroid_gauss_params"):
                c_positions = model.centroid_positions.cpu().numpy()
                c_count = c_positions.shape[0]
                map_to_tensors["x"] = np.concatenate([map_to_tensors["x"], c_positions[:, 0]])
                map_to_tensors["y"] = np.concatenate([map_to_tensors["y"], c_positions[:, 1]])
                map_to_tensors["z"] = np.concatenate([map_to_tensors["z"], c_positions[:, 2]])
                c_normals = model.centroid_normals.cpu().numpy().astype(np.float32)
                map_to_tensors["nx"] = np.concatenate([map_to_tensors["nx"], c_normals[:, 0]])
                map_to_tensors["ny"] = np.concatenate([map_to_tensors["ny"], c_normals[:, 1]])
                map_to_tensors["nz"] = np.concatenate([map_to_tensors["nz"], c_normals[:, 2]])

                if model.config.sh_degree > 0:
                    c_shs_0 = model.centroid_gauss_params["features_dc"].data.contiguous().cpu().numpy()
                    for i in range(c_shs_0.shape[1]):
                        map_to_tensors[f"f_dc_{i}"] = np.concatenate(
                            [map_to_tensors[f"f_dc_{i}"], c_shs_0[:, i, None]]
                        )
                    c_shs_rest = (
                        model.centroid_gauss_params["features_rest"].data.transpose(1, 2).contiguous().cpu().numpy()
                    )
                    c_shs_rest = c_shs_rest.reshape((c_count, -1))
                    for i in range(c_shs_rest.shape[-1]):
                        map_to_tensors[f"f_rest_{i}"] = np.concatenate(
                            [map_to_tensors[f"f_rest_{i}"], c_shs_rest[:, i, None]]
                        )
                else:
                    c_colors = torch.clamp(
                        torch.sigmoid(model.centroid_gauss_params["features_dc"].data), 0.0, 1.0
                    ).cpu().numpy()
                    map_to_tensors["colors"] = np.concatenate(
                        [map_to_tensors["colors"], (c_colors * 255).astype(np.uint8)]
                    )

                c_opacities = model.centroid_gauss_params["opacities"].data.cpu().numpy()
                map_to_tensors["opacity"] = np.concatenate([map_to_tensors["opacity"], c_opacities])

                c_scales = model.centroid_log_scales.cpu().numpy()
                # Own cap, for the same reason the vertex block above needs one: this
                # population's size comes from centroid_log_scales, not from
                # upper_scale * xyz_radius, so the face-based cap is not the right
                # reference for it either. (This layer is currently removed from the
                # model -- see populate_modules' removal note -- so this path is dormant;
                # fixed alongside the vertex block so it does not come back carrying the
                # same defect.)
                _c99 = float(np.percentile(np.exp(c_scales[:, :2]), 99))
                _max_log_scale_c = np.log(_c99 * 3.0 + 1e-20)
                c_scales = np.minimum(c_scales, _max_log_scale_c)
                for i in range(3):
                    map_to_tensors[f"scale_{i}"] = np.concatenate(
                        [map_to_tensors[f"scale_{i}"], c_scales[:, i, None]]
                    )

                c_quats = model.centroid_quats.cpu().numpy()
                for i in range(4):
                    map_to_tensors[f"rot_{i}"] = np.concatenate([map_to_tensors[f"rot_{i}"], c_quats[:, i, None]])

                # Unlike the vertex layer, a centroid Gaussian DOES belong to exactly
                # one face -- mark it honestly with that face's real index rather than
                # -1, since downstream tooling that groups by mesh_face_idx (e.g.
                # report_coverage-style per-face analysis) can meaningfully attribute
                # it. this does mean per-face Gaussian counts computed from
                # mesh_face_idx will be one higher than gaussians_to_mesh_indices alone
                # would suggest -- anything that assumes that equality needs updating.
                map_to_tensors["mesh_face_idx"] = np.concatenate(
                    [map_to_tensors["mesh_face_idx"], np.arange(c_count, dtype=np.int32)]
                )

                n = n + c_count
                count = n
                CONSOLE.print(f"Exporting {c_count} additional centroid-anchored Gaussians ({n} total)")

            if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
                crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                assert crop_obb is not None
                # positions must match map_to_tensors' current row count -- if the
                # vertex-anchored and/or centroid-anchored filler populations were
                # appended above, `positions` (captured before that) is stale/too short
                # and would size-mismatch against map_to_tensors[k] below.
                all_positions = np.stack([map_to_tensors["x"], map_to_tensors["y"], map_to_tensors["z"]], axis=-1)
                mask = crop_obb.within(torch.from_numpy(all_positions)).numpy()
                for k, t in map_to_tensors.items():
                    map_to_tensors[k] = map_to_tensors[k][mask]

                n = map_to_tensors["x"].shape[0]
                count = n

        # post optimization, it is possible have NaN/Inf values in some attributes
        # to ensure the exported ply file has finite values, we enforce finite filters.
        select = np.ones(n, dtype=bool)
        for k, t in map_to_tensors.items():
            if t.dtype == np.int32:
                continue  # int fields cannot have NaN/Inf
            n_before = np.sum(select)
            if k in ["x", "y", "z"]:
                select = np.logical_and(select, np.isfinite(t))
            else:
                select = np.logical_and(select, np.isfinite(t).all(axis=-1))
            n_after = np.sum(select)
            if n_after < n_before:
                CONSOLE.print(f"{n_before - n_after} NaN/Inf elements in {k}")

        if np.sum(select) < n:
            CONSOLE.print(f"values have NaN/Inf in map_to_tensors, only export {np.sum(select)}/{n}")
            for k, t in map_to_tensors.items():
                map_to_tensors[k] = map_to_tensors[k][select]
            count = np.sum(select)

        mesh_verts_np = None
        mesh_faces_np = None
        if hasattr(model, "mesh_verts") and hasattr(model, "mesh_faces"):
            mesh_verts_np = model.mesh_verts.cpu().numpy().astype(np.float32)
            mesh_faces_np = model.mesh_faces.cpu().numpy().astype(np.int32)
            CONSOLE.print(f"Exporting mesh: {mesh_verts_np.shape[0]} vertices, {mesh_faces_np.shape[0]} faces")

        ExportGaussianSplat.write_ply(str(filename), count, map_to_tensors, mesh_verts_np, mesh_faces_np)


Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa
