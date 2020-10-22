#!/usr/bin/env python3
"""
Author : Emmanuel Gonzalez
Date   : 2020-10-21
Purpose: Rotate and merge east and west point clouds.
"""

import argparse
import os
import sys
import open3d as o3d
import numpy as np 
import json
from terrautils.spatial import scanalyzer_to_latlon, scanalyzer_to_utm
from math import pi, cos, sin
import copy
import glob


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dir',
                        nargs='+',
                        metavar='dir',
                        help='Directory containing two point clouds (east and west)')

    parser.add_argument('-o',
                        '--outdir',
                        help='Output directory',
                        metavar='outdir',
                        type=str,
                        default='plymerge_out')

    parser.add_argument('-m',
                        '--metadata',
                        help='Metadata json file',
                        metavar='meta',
                        type=str,
                        default=None,
                        required=True)

    # parser.add_argument('-f',
    #                     '--file',
    #                     help='A readable file',
    #                     metavar='FILE',
    #                     type=argparse.FileType('r'),
    #                     default=None)

    return parser.parse_args()


# --------------------------------------------------
def load_point_clouds(pcd_path, voxel_size=0.0):
    pcds = []
    file_name_list = []
    for i in glob.glob(f'{pcd_path}/*_0.ply'):
        
        file_name = i.split('/')[-1]
        file_name_list.append(file_name)
        pcd = o3d.io.read_point_cloud(i)
        pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)
        pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcds.append(pcd_down)
    
    return pcds, file_name_list


# --------------------------------------------------
def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    icp_coarse = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp


# --------------------------------------------------
def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    pose_graph = o3d.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id])
            print("Build o3d.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry case
                odometry = np.dot(transformation_icp, odometry)
                pose_graph.nodes.append(
                    o3d.registration.PoseGraphNode(np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=False))
            else:  # loop closure case
                pose_graph.edges.append(
                    o3d.registration.PoseGraphEdge(source_id,
                                                   target_id,
                                                   transformation_icp,
                                                   information_icp,
                                                   uncertain=True))
    return pose_graph


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()


    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    for path in args.dir:
        
        voxel_size = 0.50
        pcds_down, file_name_list = load_point_clouds(path, voxel_size)
        file_name = glob.glob(f'{path}/*_metadata.json')[0].split('/')[-1].replace('_metadata.json', '_merged.ply')
        out_path = os.path.join(args.outdir, file_name)

        pcd_combined = o3d.geometry.PointCloud()
        for point_id in range(len(pcds_down)):
           pcd_combined += pcds_down[point_id]

        with open(args.metadata) as f:
            meta_dict = json.load(f)

        gantry_x = float(meta_dict['lemnatec_measurement_metadata']['gantry_system_variable_metadata']['position x [m]'])
        gantry_y = float(meta_dict['lemnatec_measurement_metadata']['gantry_system_variable_metadata']['position y [m]'])
        gantry_z = float(meta_dict['lemnatec_measurement_metadata']['gantry_system_variable_metadata']['position z [m]'])
        scan_direction = int(meta_dict['lemnatec_measurement_metadata']['sensor_variable_metadata']['current setting Scan direction (automatically set at runtime)'])
        scan_distance = float(meta_dict['lemnatec_measurement_metadata']['gantry_system_variable_metadata']['scanDistance [m]'])

        if scan_direction == 0:
            gantry_y = gantry_y - scan_distance/2
            theta = np.radians(90)

            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                       [np.sin(theta), np.cos(theta), 0],
                                       [0, 0, 1]])

            # rotation_matrix = np.array([[0, -1, 0],
            #                             [1, 0, 0],
            #                             [0, 0, 1]])

        elif scan_direction==1:
            gantry_y = gantry_y + scan_distance/2
            theta = np.radians(90)

            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                        [np.sin(theta), np.cos(theta), 0],
                                        [0, 0, 1]])
            # rotation_matrix = np.array([[0, -1, 0],
            #                             [1, 0, 0],
            #                             [0, 0, 1]])
        
        (utm_x, utm_y) = scanalyzer_to_utm(float(gantry_x), float(gantry_y))

        center_z = float(pcd_combined.get_center()[2])
        corrected_pcd = pcd_combined.translate([utm_x, utm_y, gantry_z], relative=False)

        scaled_pcd = corrected_pcd.scale(0.000982699112208, corrected_pcd.get_center())

        rotated_pcd = scaled_pcd.rotate(rotation_matrix, center=scaled_pcd.get_center())

        o3d.io.write_point_cloud(out_path, rotated_pcd)


# --------------------------------------------------
if __name__ == '__main__':
    main()
