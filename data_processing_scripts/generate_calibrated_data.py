# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 18:40:58 2022

@author: lowellm
"""

import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import torchvision
import PIL


import dataset_utils


data_folder = "C:\\Users\\lowellm\\Desktop\\forestry_project\\data\\winter_forestry_building\\"


if __name__ == '__main__':
    
    Utilities = dataset_utils.StereoRigUtilities()
    Utilities.LoadRigParameters()
    Utilities.CreateRectificationMaps()
    
    
    os.mkdir(data_folder + "\\true_disparity\\")
    os.mkdir(data_folder + "\\rectified_camera\\")
    os.mkdir(data_folder + "\\rectified_camera\\left\\")
    os.mkdir(data_folder + "\\rectified_camera\\right\\")
    
    # write rectified stereo parameters to file for future use. 
    
    file = open(data_folder + "\\rectified_camera\\" + "rectified_stereo_parameters.txt","w")
    
    file.writelines("CIM1\n")
    
    IM1 = Utilities.rectified_stereo_parameters["RCIM1"]
    IM1 = IM1[:,0:3]
    
    IM1str = str(IM1)
    IM1str = IM1str.replace("[","")
    IM1str = IM1str.replace("]","")
    
    file.writelines(IM1str + "\n\n")
    
    file.writelines("CIM2\n")
    
    IM2 = Utilities.rectified_stereo_parameters["RCIM2"]
    IM2 = IM2[:,0:3]
    
    IM2str = str(IM2)
    IM2str = IM2str.replace("[","")
    IM2str = IM2str.replace("]","")
    
    file.writelines(IM2str + "\n\n")
    
    Q = Utilities.rectified_stereo_parameters["Q"]
    
    T = -1/Q[3,2]
    
    file.writelines("T\n")
    file.writelines(str(T) + "\n\n")
    
    file.writelines("Q\n")

    Q = Utilities.rectified_stereo_parameters["Q"]
    Qstr = str(Q)
    
    Qstr = Qstr.replace("[", "")
    Qstr = Qstr.replace("]","")
    
    file.writelines(Qstr)
    
    file.flush()
    file.close()
    
    
    # scan through left images to generate file names
    
    file_list = os.listdir(data_folder + "\\camera\\left\\")
    
    file_count = len(file_list)

    for file in file_list:
        
        _ = file.split(".")
        
        
        file_name = _[0]
        file_extension = _[1]
        
        # only process jpg images
        
        if file_extension == "jpg":
    
            # load pointcloud
            
            pcd = o3d.io.read_point_cloud(data_folder + "\\lidar\\" + file_name + ".pcd") 
            
            # loage images
            
            left_img = torchvision.io.read_image(data_folder + "\\camera\\left\\" + file_name + ".jpg")
            right_img = torchvision.io.read_image(data_folder + "\\camera\\right\\" + file_name + ".jpg")
            
            # undistort images
            
            left_undistorted_image = Utilities.UndistortLeftImage(left_img)
            right_undistorted_image = Utilities.UndistortRightImage(right_img)
            
            # generate rectified images
            
            left_rect_img, right_rect_img = Utilities.RectifyImage(left_undistorted_image, right_undistorted_image) 
            
            # save rectified images
            
            torchvision.io.write_jpeg(left_rect_img, data_folder + "\\rectified_camera\\left\\" + file_name + ".jpg")
            torchvision.io.write_jpeg(right_rect_img, data_folder + "\\rectified_camera\\right\\" + file_name + ".jpg")
            
            # load LIDAR data
            points = np.asarray(pcd.points)
            points = np.transpose(points) # create 3xpoint_count array
            
            # generate depth map from lidar data
        
            projected_points_left_rectified_image, projected_points_left_rectified_image_points = Utilities.ProjectLidarToRectifiedLeftImage(points)
            
            # generate disparity values from depth image 
            
            disparity_ground_truth = Utilities.DepthtoDisparityImage(projected_points_left_rectified_image)
            
            # make disparities over 255 invalid by setting to 0
            
            disparity_ground_truth[disparity_ground_truth > 255] = 0
            
            # scale disparity by 256 as will be saved as 16 bit PNG 
            
            disparity_ground_truth = disparity_ground_truth*256
            
            # convert to numpy array
            
            disparity_ground_truth = np.asarray(disparity_ground_truth)
            
            # round disparity to nearest integer
            
            disparity_ground_truth = np.rint(disparity_ground_truth)
            
            # convert to uint16
        
            disparity_ground_truth = disparity_ground_truth.astype(np.uint16)
        
            # convert to PIL image disparity image as png file
            
            disparity_image = PIL.Image.fromarray(disparity_ground_truth)
            
            # save as PNG file
            disparity_image.save(data_folder + "\\true_disparity\\" + file_name + ".png") 
            
    
        
        
        
        
        
        
        
        
        
        
        