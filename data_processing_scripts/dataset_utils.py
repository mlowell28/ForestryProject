# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:21:27 2022

@author: lowellm
"""

import cv2 as cv
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib
import math
import json
from PIL import Image
import torch
import torchvision




class StereoRigUtilities:
    
    def __init__(self, file=None):     
        None
   
    def LoadRigParameters(self, file=None):
        
        if file == None:
            directory = os.path.dirname(__file__)
            file_path = directory + "\\parameters.txt"
            
        file = open(file_path)
            
            
        rig_parameters = {}
            
        # load camera 1 intrinisc matrix

        while(file.readline() != "CIM1\n"):
            None
            
        row_1 = file.readline()
        row_2 = file.readline()
        row_3 = file.readline()

        CIM1 = np.array([np.fromstring(row_1, sep = " "), np.fromstring(row_2, sep = " "), np.fromstring(row_3, sep = " ")])      
        rig_parameters["CIM1"] = CIM1
         
        # load camera 1 distortion coefficents

        file.seek(0)
        while(file.readline() != "DC1\n"):
            None
            
        DC1_string = file.readline()
        DC1 = np.fromstring(DC1_string, sep = " ")
        rig_parameters["DC1"] = DC1

        # load camera 2 intrinsic matrix

        file.seek(0)
        while(file.readline() != "CIM2\n"):
            None
            
        row_1 = file.readline()
        row_2 = file.readline()
        row_3 = file.readline()

        CIM2 = np.array([np.fromstring(row_1, sep = " "), np.fromstring(row_2, sep = " "), np.fromstring(row_3, sep = " ")]) 
        rig_parameters["CIM2"] = CIM2

        # load camera 2 distortion coefficents                 

        file.seek(0)
        while(file.readline() != "DC2\n"):
            None
            
        DC2_string = file.readline()
        DC2 = np.fromstring(DC2_string, sep = " ")
        rig_parameters["DC2"] = DC2

        # load camera translation from camera 2 to camera 1

        file.seek(0)
        while(file.readline() != "C2T\n"):
            None
            
        C2T_string = file.readline()
        C2T = np.fromstring(C2T_string, sep = " ")
        rig_parameters["C2T"] = C2T

        # load rotation from camera 2 to camera 1

        file.seek(0)
        while(file.readline() != "C2R\n"):
            None
            
        row_1 = file.readline()
        row_2 = file.readline()
        row_3 = file.readline()

        C2R = np.array([np.fromstring(row_1, sep = " "), np.fromstring(row_2, sep = " "), np.fromstring(row_3, sep = " ")])
        rig_parameters["C2R"] = C2R

        # load image size

        file.seek(0)
        while(file.readline() != "image_size\n"):
            None
            
        image_size_string = file.readline()
        image_size = np.fromstring(image_size_string, sep = " ", dtype =int)       
        rig_parameters["image_size"] = tuple(image_size)

        # load lidar translation to camera 1

        file.seek(0)
        while(file.readline() != "LT\n"):
            None
            
        LT_string = file.readline()
        LT = np.fromstring(LT_string, sep = " ")
        LT = LT.reshape((3,1))
        rig_parameters["LT"] = LT

        # load lidar rotation to camera 1

        file.seek(0)
        while(file.readline() != "LR\n"):
            None
            
        row_1 = file.readline()
        row_2 = file.readline()
        row_3 = file.readline()

        LR = np.array([np.fromstring(row_1, sep = " "), np.fromstring(row_2, sep = " "), np.fromstring(row_3, sep = " ")]) 
        rig_parameters["LR"] = LR

        file.close()
        
        self.rig_parameters = rig_parameters
        
        return rig_parameters
            
    
    def CreateRectificationMaps(self, rig_parameters=None):
        
        if rig_parameters == None:
            rig_parameters = self.rig_parameters
        
        # generate rectification maps from parameters
        RCRM1, RCRM2, RCIM1, RCIM2, Q, ROI1, ROI2 = cv.stereoRectify(rig_parameters["CIM1"], rig_parameters["DC1"], rig_parameters["CIM2"], rig_parameters["DC2"], rig_parameters["image_size"], rig_parameters["C2R"], rig_parameters["C2T"], flags=cv.CALIB_ZERO_DISPARITY, alpha=0)

        rectified_stereo_parameters = {"RCRM1":RCRM1, "RCRM2":RCRM2, "RCIM1": RCIM1, "RCIM2":RCIM2, "Q":Q, "ROI1":ROI1, "ROI2":ROI2}
        

        left_map = cv.initUndistortRectifyMap(rig_parameters["CIM1"], rig_parameters["DC1"], rectified_stereo_parameters["RCRM1"], 
                                              rectified_stereo_parameters["RCIM1"], rig_parameters["image_size"], cv.CV_32FC1)

        right_map = cv.initUndistortRectifyMap(rig_parameters["CIM2"], rig_parameters["DC2"],  rectified_stereo_parameters["RCRM2"],
                                               rectified_stereo_parameters["RCIM2"], rig_parameters["image_size"],cv.CV_32FC1)
        
    
        # save rectification maps for future use
        self.rectified_stereo_parameters = rectified_stereo_parameters
        self.left_map = left_map
        self.right_map = right_map
        
        return rectified_stereo_parameters, left_map, right_map
    
    
    # takes a torch array assuming dimensions batchxCxHxW or CxHxW 
    
    def UndistortLeftImage(self, left_img):
        
        left_img_np = np.array(left_img) # converts to numpy 
        shape = left_img_np.shape
        dimensions = len(shape)
        
        # if a single image
        if dimensions == 3:
            
            left_img_np = np.transpose(left_img_np, (1,2,0))
            left_img_undistorted = cv.undistort(left_img_np, self.rig_parameters["CIM1"], self.rig_parameters["DC1"])
            left_img_undistorted = np.transpose(left_img_undistorted, (2,0,1))
            left_img_undistorted = torch.tensor(left_img_undistorted,dtype=torch.uint8) 
            
            return left_img_undistorted
        
        if dimensions == 4:
            
            left_img_np = np.transpose(left_img_np, (0,2,3,1))
            left_imgs_undistorted = np.zeros(left_img_np.shape)
            
            for index in range(shape[0]):
            
                left_img_undistorted = cv.undistort(left_img_np[index], self.rig_parameters["CIM1"], self.rig_parameters["DC1"])
                left_imgs_undistorted[index, :, :, :] = left_img_undistorted
                
            left_imgs_undistorted = np.transpose(left_imgs_undistorted, (0,3,1,2))  
            left_imgs_undistorted = torch.tensor(left_imgs_undistorted, dtype=torch.uint8) 
            
            return left_imgs_undistorted
            
     # takes a torch array assuming dimensions batchxCxHxW or CxHxW     
        
    def UndistortRightImage(self, right_img):
        
        right_img_np = np.array(right_img)
        shape = right_img.shape
        dimensions = len(shape)
        
        # if a single image
        if dimensions == 3:
            
            right_img_np = np.transpose(right_img_np, (1,2,0))
            right_img_undistorted = cv.undistort(right_img_np, self.rig_parameters["CIM2"], self.rig_parameters["DC2"])
            right_img_undistorted = np.transpose(right_img_undistorted, (2,0,1))
            right_img_undistorted = torch.tensor(right_img_undistorted, dtype=torch.uint8) 
            
            return right_img_undistorted
        
        if dimensions == 4:
            
            right_img_np = np.transpose(right_img_np, (0,2,3,1))
            right_imgs_undistorted = np.zeros(right_img_np.shape)
            
            for index in range(shape[0]):
            
                right_img_undistorted = cv.undistort(right_img_np[index], self.rig_parameters["CIM2"], self.rig_parameters["DC2"])
                right_imgs_undistorted[index, :, :, :] = right_img_undistorted
                
            right_imgs_undistorted = np.transpose(right_imgs_undistorted, (0,3,1,2))   
            right_imgs_undistorted = torch.tensor(right_imgs_undistorted, dtype=torch.uint8) 
            
            return right_imgs_undistorted
    
    
     # takes a torch array assuming dimensions batchxCxHxW or CxHxW     
    
    def RectifyImage(self, left_img, right_img=None):
        
        left_img_np = np.array(left_img)
        
        shape = left_img.shape
        dimensions = len(shape)
        
        if dimensions == 3:
            
            # change channel to be consist with openCV 
            left_img_np = np.transpose(left_img_np, (1,2,0))
            left_rect_img = cv.remap(left_img_np, self.left_map[0], self.left_map[1], cv.INTER_LINEAR)
            left_rect_img = np.transpose(left_rect_img, (2,0,1))
            left_rect_img = torch.tensor(left_rect_img, dtype=torch.uint8)
            
            if right_img != None:
                right_img_np = np.array(right_img)
                right_img_np = np.transpose(right_img_np, (1,2,0))
                right_rect_img = cv.remap(right_img_np, self.right_map[0], self.right_map[1], cv.INTER_LINEAR)
                right_rect_img = np.transpose(right_rect_img, (2,0,1))
                right_rect_img = torch.tensor(right_rect_img, dtype=torch.uint8)
    
                return left_rect_img, right_rect_img 
            
            return left_rect_img
        
                
        if dimensions == 4:
            
            # change channel to be consist with openCV 

            left_img_np = np.transpose(left_img_np, (0,2,3,1))
            left_rect_imgs = np.zeros(left_img_np.shape)
            
            if right_img != None:
                
                right_img_np = np.array(right_img)
                right_img_np = np.transpose(right_img_np, (0,2,3,1))
                right_rect_imgs = np.zeros(right_img_np.shape)
                
            for index in range(shape[0]):
                
                left_rect_img = cv.remap(left_img_np[index,:,:,:], self.left_map[0], self.left_map[1], cv.INTER_LINEAR) 
                left_rect_imgs[index,:,:,:] = left_rect_img
                
                if right_img != None:
                    right_rect_img = cv.remap(right_img_np[index,:,:,:], self.right_map[0], self.right_map[1], cv.INTER_LINEAR)
                    right_rect_imgs[index,:,:,:] = right_rect_img
            
            left_rect_imgs = np.transpose(left_rect_imgs, (0,3,1,2))
            left_rect_imgs = torch.tensor(left_rect_imgs, dtype=torch.uint8)
            
            if right_img != None:
                
                right_rect_imgs = np.transpose(right_rect_imgs, (0,3,1,2))
                right_rect_imgs = torch.tensor(right_rect_imgs, dtype=torch.uint8)

                return left_rect_imgs, right_rect_imgs
            
            return left_rect_imgs
        
    
    def PointsToLeftCameraCoordinates(self, points):
        
        points_in_camera_coordinates = torch.matmul(self.rig_parameters["LR"], points) + self.rig_parameters["LT"]
        return points_in_camera_coordinates
    
    def PointsToRectifiedLeftCameraCoordinates(self, points):

        points_in_camera_coordinates = torch.matmul(self.rig_parameters["LR"], points) + self.rig_parameters["LT"]
        points_in_rectified_camera_coordinates = torch.matmul(self.rectified_stereo_parameters["RCRM1"], points_in_camera_coordinates)
        return points_in_rectified_camera_coordinates

    # points of form Scan_Numx3d_CordxPoint_Index, 
    
    def ProjectLidarToLeftImage(self, points):
        
     points = np.array(points)
     
     shape = points.shape
     dimensions = len(shape)
     image_size = self.rig_parameters["image_size"]
     num_points = shape[dimensions-1]
     
     # transform points into left camera coordinate system     
     points_in_camera_coordinates = np.matmul(self.rig_parameters["LR"], points) + self.rig_parameters["LT"]
                
    
     if dimensions == 2:
         
         # turn points into homogenous coordinates
         
         # vec = np.ones((1, num_points))        
         # homogenous_cord = np.concatenate((points_in_camera_coordinates, vec), axis=0)
         homogenous_projected_points = np.matmul(self.rig_parameters["CIM1"], points_in_camera_coordinates)
         
         projected_points = np.copy(homogenous_projected_points)
         
         projected_points[0,:] = projected_points[0,:]/homogenous_projected_points[2,:]
         projected_points[1,:] = projected_points[1,:]/homogenous_projected_points[2,:]
             
         column_index = (np.where(((-.5 < projected_points[0,:]) & (projected_points[0,:]< (image_size[0]-.5)) & (-.5 < projected_points[1,:]) &( projected_points[1,:] < (image_size[1]-.5)) & (projected_points[0,:] != np.nan) & (projected_points[1,:] != np.nan))))[0]
         
         clipped_points = projected_points[:,column_index]
         
         clipped_points_rounded = np.copy(clipped_points)
         clipped_points_rounded[0:2,:] = np.rint(clipped_points_rounded[0:2, :])
         
         x_locations = np.array(clipped_points_rounded[0,:], dtype=np.int64)
         y_locations = np.array(clipped_points_rounded[1,:], dtype=np.int64)
         z_values = np.array(clipped_points_rounded[2,:])
         
         image = np.zeros((image_size[1], image_size[0]))
         image[y_locations, x_locations] = z_values
         
         image = torch.tensor(image)
         clipped_points_rounded = torch.tensor(clipped_points_rounded)
         
         return image, clipped_points_rounded
     
     if dimensions == 3:
         
         # turn points into homogenous coordinates
         
         num_scans = shape[0]
         
         # vec = np.ones((num_scans, 1, num_points))        
         
         # homogenous_cord = np.concatenate((points_in_camera_coordinates, vec), axis=1)
         
         homogenous_projected_points = np.matmul(self.rig_parameters["CIM1"], points_in_camera_coordinates)
         
         projected_points = np.copy(homogenous_projected_points)
         
         projected_points[:,0,:] = projected_points[:,0,:]/homogenous_projected_points[:,2,:]
         projected_points[:,1,:] = projected_points[:,1,:]/homogenous_projected_points[:,2,:]
             
         image = np.zeros((num_scans, image_size[1], image_size[0]))
         
         clipped_points = np.zeros((num_scans, 3, num_points))
             
         column_index = (np.where(((-.5 < projected_points[:,0,:]) & (projected_points[:,0,:]< (image_size[0]-.5)) & (-.5 < projected_points[:,1,:]) &( projected_points[:,1,:] < (image_size[1]-.5)) & (projected_points[:,0,:] != np.nan) & (projected_points[:,1,:] != np.nan))))
             
         clipped_points = projected_points[column_index[0], :, column_index[1]]
         clipped_points_rounded = np.copy(clipped_points)
         clipped_points_rounded[:,0:2] = np.rint(clipped_points_rounded[:,0:2])
         
         x_locations = np.array(clipped_points_rounded[:,0], dtype=np.int64)
         y_locations = np.array(clipped_points_rounded[:,1], dtype=np.int64)
         z_values = np.array(clipped_points_rounded[:,2])
         
         image[column_index[0], y_locations, x_locations] = z_values   
         
         clipped_points_return = np.zeros((num_scans, 3, num_points))
         
         scan_length_max = 0
         
         for scan_index in range(num_scans):
             
             scan_column_index = np.where(column_index[0]==scan_index)
             x_locations_column_index = x_locations[scan_column_index]
             y_locations_column_index = y_locations[scan_column_index]
             z_locations_column_index = z_values[scan_column_index]
             
             scan_length =  np.shape(x_locations_column_index)
             
             clipped_points_return[scan_index, 0, 0:scan_length[0]] = x_locations_column_index
             clipped_points_return[scan_index, 1, 0:scan_length[0]] = y_locations_column_index
             clipped_points_return[scan_index, 2, 0:scan_length[0]] = z_locations_column_index
             
             if scan_length_max < scan_length[0]:
                 scan_length_max = scan_length[0]
                 
         clipped_points_return = clipped_points_return[:,:,:scan_length_max]
                 
         image = torch.tensor(image)
         clipped_points_return = torch.tensor(clipped_points_return)
         
         return image, clipped_points_return
            
            
     
    # points of form Scan_Numx3d_CordxPoint_Index, 
        
    def ProjectLidarToRectifiedLeftImage(self, points):
        
        points = np.array(points)
        
        shape = points.shape
        dimensions = len(shape)
        image_size = self.rig_parameters["image_size"]
        num_points = shape[dimensions-1]
        
        # transform points into left camera coordinate system     
        points_in_camera_coordinates = np.matmul(self.rig_parameters["LR"], points) + self.rig_parameters["LT"]
        
        # transform points into rectified left camera coordinate system
        points_in_rectified_camera_coordinates = np.matmul(self.rectified_stereo_parameters["RCRM1"], points_in_camera_coordinates)             
            

        if dimensions == 2:
            
            # turn points into homogenous coordinates
            
            vec = np.ones((1, num_points))        
            homogenous_cord = np.concatenate((points_in_rectified_camera_coordinates, vec), axis=0)
            homogenous_projected_points = np.matmul(self.rectified_stereo_parameters["RCIM1"], homogenous_cord)
            
            projected_points = np.copy(homogenous_projected_points)
            
            projected_points[0,:] = projected_points[0,:]/homogenous_projected_points[2,:]
            projected_points[1,:] = projected_points[1,:]/homogenous_projected_points[2,:]
                
            column_index = (np.where(((-.5 < projected_points[0,:]) & (projected_points[0,:]< (image_size[0]-.5)) & (-.5 < projected_points[1,:]) &( projected_points[1,:] < (image_size[1]-.5)) & (projected_points[0,:] != np.nan) & (projected_points[1,:] != np.nan))))[0]
            
            clipped_points = projected_points[:,column_index]
            
            clipped_points_rounded = np.copy(clipped_points)
            clipped_points_rounded[0:2,:] = np.rint(clipped_points_rounded[0:2, :])
            
            x_locations = np.array(clipped_points_rounded[0,:], dtype=np.int64)
            y_locations = np.array(clipped_points_rounded[1,:], dtype=np.int64)
            z_values = np.array(clipped_points_rounded[2,:])
            
            image = np.zeros((image_size[1], image_size[0]))
            image[y_locations, x_locations] = z_values
            
            image = torch.tensor(image)
            clipped_points_rounded = torch.tensor(clipped_points_rounded)
            
            return image, clipped_points_rounded
        
        if dimensions == 3:
            
            # turn points into homogenous coordinates
            
            num_scans = shape[0]
            
            vec = np.ones((num_scans, 1, num_points))        
            
            homogenous_cord = np.concatenate((points_in_rectified_camera_coordinates, vec), axis=1)
            
            homogenous_projected_points = np.matmul(self.rectified_stereo_parameters["RCIM1"], homogenous_cord)
            
            projected_points = np.copy(homogenous_projected_points)
            
            projected_points[:,0,:] = projected_points[:,0,:]/homogenous_projected_points[:,2,:]
            projected_points[:,1,:] = projected_points[:,1,:]/homogenous_projected_points[:,2,:]
                
            image = np.zeros((num_scans, image_size[1], image_size[0]))
            
            clipped_points = np.zeros((num_scans, 3, num_points))
                
            column_index = (np.where(((-.5 < projected_points[:,0,:]) & (projected_points[:,0,:]< (image_size[0]-.5)) & (-.5 < projected_points[:,1,:]) &( projected_points[:,1,:] < (image_size[1]-.5)) & (projected_points[:,0,:] != np.nan) & (projected_points[:,1,:] != np.nan))))
                
            clipped_points = projected_points[column_index[0], :, column_index[1]]
            clipped_points_rounded = np.copy(clipped_points)
            clipped_points_rounded[:,0:2] = np.rint(clipped_points_rounded[:,0:2])
            
            x_locations = np.array(clipped_points_rounded[:,0], dtype=np.int64)
            y_locations = np.array(clipped_points_rounded[:,1], dtype=np.int64)
            z_values = np.array(clipped_points_rounded[:,2])
            
            image[column_index[0], y_locations, x_locations] = z_values   
            
            clipped_points_return = np.zeros((num_scans, 3, num_points))
            
            scan_length_max = 0
            
            for scan_index in range(num_scans):
                
                scan_column_index = np.where(column_index[0]==scan_index)
                x_locations_column_index = x_locations[scan_column_index]
                y_locations_column_index = y_locations[scan_column_index]
                z_locations_column_index = z_values[scan_column_index]
                
                scan_length =  np.shape(x_locations_column_index)
                
                clipped_points_return[scan_index, 0, 0:scan_length[0]] = x_locations_column_index
                clipped_points_return[scan_index, 1, 0:scan_length[0]] = y_locations_column_index
                clipped_points_return[scan_index, 2, 0:scan_length[0]] = z_locations_column_index
                
                if scan_length_max < scan_length[0]:
                    scan_length_max = scan_length[0]
                    
            clipped_points_return = clipped_points_return[:,:,:scan_length_max]
                    
            image = torch.tensor(image)
            clipped_points_return = torch.tensor(clipped_points_return)
            
            return image, clipped_points_return
    
    # returns a 4xpoint_num array where the fourth element is the projected disparity value for the given point in 3d space
    # using the inverse of the disparity_to_depth matrix 
    
    
    def DepthtoDisparityImage(self, projected_depth):
        
        Q = self.rectified_stereo_parameters["Q"]
        
        f = Q[2, 3]
        Tx = -1/Q[3,2]
        
        # we consider zero depth values as not having an associated lidar scan point 
        # thus the returned disparity image will only consider nonzero disparity values for error computation
        
        zero_depth_index = (projected_depth == 0)
        
        # set 0 values to 1 for inversion
        projected_depth[projected_depth == 0] = 1
        
        # invert
        projected_depth_inverted = 1/projected_depth
        
        # set zero index back to 0
        
        projected_depth_inverted[zero_depth_index] = 0 
        
        #inverting formula to derive disparity from depth
    
        disparity_image = -f*Tx*projected_depth_inverted
        
        return disparity_image
    
    def DisparityToDepthImage(self, projected_disparity):
        
        Q = self.rectified_stereo_parameters["Q"]
        f = Q[2, 3]
        Tx = -1/Q[3,2]
        
        zero_disparity_index = (projected_disparity == 0)
        
        # make zero disparity 1 for inversion
        projected_disparity[projected_disparity == 0] = 1
        
        # compute distances 
        inverted_disparity = 1/projected_disparity
        
        #set 0 values back to to
        
        inverted_disparity[zero_disparity_index] = 0 
        
        # compute depth image
        depth_image = -f*Tx*inverted_disparity
        
        return depth_image
        

    
    def DisparityError(self, predicted_disparity, true_disparity, pixel_error_limit = 5, max_disparity = -1, remove_disparities_over_max = True):
        
        # make unknown disparity values equal to 0
        
        predicted_disparity[predicted_disparity == -1] = 0 
        
        shape = true_disparity.shape
        dimensions = len(shape)
        
        # if we want to remove parts of the image whos corresponding pixel is outside the image along with disparity values greater than the maximum limit:
       
        if max_disparity != -1:
            
            if dimensions == 2:
            
                (x_num, y_num) = true_disparity.shape
                true_disparity[:, 0:max_disparity] = 0
                
                if remove_disparities_over_max == True:
                    true_disparity[true_disparity > max_disparity] = 0
                    disparity_mask = (true_disparity>0)*(true_disparity < max_disparity)
                    
                else:
                    disparity_mask = (true_disparity>0)
                
            if dimensions == 3:
                
                (batch, x_num, y_num) = true_disparity.shape
                true_disparity[:, :, 0:max_disparity] = 0
                
                if remove_disparities_over_max == True:
                    true_disparity[true_disparity > max_disparity] = 0
                    disparity_mask = (true_disparity>0)*(true_disparity < max_disparity)
                    
                else:
                    disparity_mask = (true_disparity>0)
                    
        else:
            
            disparity_mask = (true_disparity>0)
        
        pixel_count = torch.sum(disparity_mask)    
        disparity_error_image = torch.abs(disparity_mask*predicted_disparity - true_disparity)
        average_error = torch.sum(disparity_error_image)/pixel_count
        
        greater_than_ten = torch.sum(disparity_error_image > pixel_error_limit)
        greater_than_ten_percentage = torch.sum(disparity_error_image > pixel_error_limit)/pixel_count
        
        return average_error, greater_than_ten, greater_than_ten_percentage, disparity_error_image
    
    
    
    def DepthError(self, predicated_distance, true_distance, min_distance=5, max_distance=25):
        
        # remove left part of image whos minumum distance falls outside of the right image
         Q = self.rectified_stereo_parameters["Q"]
         
         f = Q[2, 3]
         Tx = -1/Q[3,2] 
         
         # compute max_disparity corresponding to min_distance
         max_disparity = -f*Tx/min_distance
         max_disparity = int(max_disparity)
         
         # set unknown distances to max distance
         predicated_distance[predicated_distance == -1] = max_distance
         
         shape = predicated_distance.shape
         dimensions = len(shape)
         
         # if we want to remove parts of the image whos corresponding pixel is outside the image along with disparity values greater than the maximum limit:
    
         if dimensions == 2:
        
            (x_num, y_num) =  predicated_distance.shape
            
            # remove sections of image possibly not in image pair 
            
            true_distance[:, 0:max_disparity] = 0
            distance_mask = (max_distance >= true_distance)*(true_distance >= min_distance)
    
            
         if dimensions == 3:
            
            (batch, x_num, y_num) =  predicated_distance.shape
            
            
            true_distance[:, :, 0:max_disparity] = 0
            distance_mask = (max_distance >= true_distance)*(true_distance >= min_distance)
            
            
        # compute distance error
        
         pixel_count = torch.sum(distance_mask)
        
         error_image = torch.abs(distance_mask*(true_distance -  predicated_distance))
         total_error = torch.sum(error_image)
         
         if pixel_count != 0:
             average_error = total_error/pixel_count
             return average_error, total_error, error_image
         else:
             return torch.tensor(0), torch.tensor(0), error_image
             
     
        
         
        # did not have time to implement point generation in torch, so must use numpy - can implement in future
        
    def L2PositionError(self, predicted_disparity, true_disparity, min_distance = 5, max_distance = 25):
        
        
        predicted_locations = self.Generate3dPoints(np.array(predicted_disparity))
        true_locations = self.Generate3dPoints(np.array(true_disparity))
        
        predicted_locations = torch.tensor(predicted_locations)
        true_locations = torch.tensor(true_locations)
        
        distance_mask = (min_distance <= true_locations[:,:,2])*(true_locations[:,:,2] <= max_distance)
        
        squared_difference = (predicted_locations-true_locations)**2
        
        l2_error_image = distance_mask*torch.sqrt(torch.sum(squared_difference, dim=2))
        
        pixel_count = torch.sum(distance_mask)
        
        l2_error_total  = torch.sum(l2_error_image)
        
        average_l2_error = l2_error_total/pixel_count
        
        return average_l2_error, l2_error_total, l2_error_image

         
        
                # average_l2_error, l2_error_total, l2_error_image = Utilities.L2PositionError(predicted_disparity, true_disparity)
                
                # error_figure = Utilities.MeterErrorScatterPlot(left_img, l2_error_image, error_max = 1)
                # error_figure.suptitle(file_name + "." + file_extension + ", l2 error, distance 5-25: " + str(round(average_l2_error.item(), 2)))
                # error_figure.savefig(meters_error_images_overlaid_folder + file_name + ".png")
                
                
                
                
    
    def Generate3dPoints(self, disparity):
        disparity = disparity.astype(np.float32)
        reconstruction_3d = cv.reprojectImageTo3D(disparity, self.rectified_stereo_parameters["Q"])
        return reconstruction_3d
    
    # def Generate3dPointsTorch(self, disparity):
        
    #     dim = len(disparity.shape)        
        
    #     if dim==2:
            
    #         (x, y) = disparity.shape
            
    #         uvd1_points = np.ones(4, x*y)
            
    #         x_array = torch.range(0,x,1)
    #         y_array = torch.range(0,y,1)
            
            
    #         grid_x, grid_y = torch.meshgrid(x_array, y_array)
            
    #         grid_x_flat = torch.flatten(grid_x)
    #         grid_y_flat = torch.flatten(grid_y)
            
            
    #         disparity_array = torch.flatten(disparity)
            
    #         uvd1_points[0,:] = grid_x_flat
    #         uvd1_points[1,:] = grid_y_flat
    #         uvd1_points[2, :] = disparity_array
            
    #         3d_points_hom = self.rectified_stereo_parameters["Q"]*uvd1_points
            
    #         3d_points = 3d_points[0:3,:]_hom/3d_points_hom[3,:]
            
    #         3d_values = torch.reshape(3d_points, (x,y,3)
            
    #         return 3d_points
            
            
            
            
            
            
    
    
    # credit for this function goes to Omar Padierna, 
    
    def Generate3dPlot(self, disparity, image, filename):
        
        points_3d = self.Generate3dPoints(disparity)
        
        # select points with nonzero disparity
        disparity_mask = disparity > 0
        
        masked_points_3d = points_3d[disparity_mask]
        masked_image = image[disparity_mask]
        
        # generate list of RGB colors
        colors = masked_image.reshape(-1,3)
        
        # append colors to end of XYZ coordinates creating XYZRGB for each entry
        vertices = np.hstack([masked_points_3d.reshape(-1,3),colors])
    
        ply_header = '''ply
    		format ascii 1.0
    		element vertex %(vert_num)d
    		property float x
    		property float y
    		property float z
    		property uchar red
    		property uchar green
    		property uchar blue
    		end_header
    		'''
        with open(filename, 'w') as f:
            f.write(ply_header %dict(vert_num=len(vertices)))
            np.savetxt(f,vertices,'%f %f %f %d %d %d')
   
    def DisparityErrorScatterPlot(self, image, error_image, error_bound = 0, disparity_max = 208):
        
        (x, y) = self.rig_parameters["image_size"]
        error_image_np = np.array(error_image)
        locations = np.where(error_image_np > error_bound)
        values = error_image_np[locations]
        
        error_overlaid = plt.figure()
        plt.imshow(image)
        plt.scatter(locations[1], locations[0], c=values, marker="o",  s=6, vmin = error_bound, vmax= disparity_max) # vmax = 250)
        cb = plt.colorbar()
        cb.set_label("disparity error")
        
        return error_overlaid
    
    def MeterErrorScatterPlot(self, image, error_image, error_bound = 0, error_max = 100):
        
        (x, y) = self.rig_parameters["image_size"]
        error_image_np = np.array(error_image)
        locations = np.where(error_image_np > error_bound)
        values = error_image_np[locations]
        
        error_bound = max(error_bound, .1)
        
        error_overlaid = plt.figure()
        plt.imshow(image)
        plt.scatter(locations[1], locations[0], c=values, marker="o",  s=6, norm=matplotlib.colors.LogNorm(error_bound, error_max)) # vmax = 250)
        cb = plt.colorbar()
        cb.set_label("meters")
        
        return error_overlaid

data_folder = "C:\\Users\\lowellm\\Desktop\\forestry_project\\data\\winter_forestry_building\\"


if __name__ == '__main__':
    
    Utilities = StereoRigUtilities()
    Utilities.LoadRigParameters()
    Utilities.CreateRectificationMaps()
    
    
    # if os.path.isdir(data_folder + "\\test_run\\") == False:

    #     os.mkdir(data_folder + "\\test_run\\")
    #     os.mkdir(data_folder + "\\test_run\\rectified_camera\\")
    #     os.mkdir(data_folder + "\\test_run\\rectified_camera\\right\\")
    #     os.mkdir(data_folder + "\\test_run\\rectified_camera\\left\\")
    #     os.mkdir(data_folder + "\\test_run\\SGBM\\")
    #     os.mkdir(data_folder + "\\test_run\\projected_lidar\\")
    
        
    pcd_1 = o3d.io.read_point_cloud(data_folder + "\\lidar\\142.pcd") 
    pcd_2 = o3d.io.read_point_cloud(data_folder + "\\lidar\\1698.pcd")
    
    
    left_img = torchvision.io.read_image(data_folder + "\\camera\\left\\142.jpg")
    right_img = torchvision.io.read_image(data_folder + "\\camera\\right\\142.jpg")
    
    left_img_1 = torchvision.io.read_image(data_folder + "\\camera\\left\\142.jpg")
    
    right_img_1 = torchvision.io.read_image(data_folder + "\\camera\\right\\" + "142" + ".jpg")
    
    left_img_2 = torchvision.io.read_image(data_folder + "\\camera\\left\\" + "1698" + ".jpg")
    right_img_2 = torchvision.io.read_image(data_folder + "\\camera\\right\\" + "1698" + ".jpg")
    
    shape = left_img_1.shape
    
    left_img_stacked = torch.zeros((2, shape[0], shape[1], shape[2]))
    right_img_stacked = torch.zeros((2, shape[0], shape[1], shape[2]))
    
    left_img_stacked[0,:,:,:] = left_img_1
    left_img_stacked[1,:,:,:] = left_img_2
    
    right_img_stacked[0,:,:,:] = right_img_1
    right_img_stacked[1,:,:,:] = right_img_2
    
    
    left_undistorted_image = Utilities.UndistortLeftImage(left_img)
    right_undistorted_image = Utilities.UndistortRightImage(right_img)
    
    left_undistorted_images = Utilities.UndistortLeftImage(left_img_stacked)
    right_undistorted_images = Utilities.UndistortRightImage(right_img_stacked)
    
    
    left_rect_img_1, right_rect_img_1 = Utilities.RectifyImage(left_img_1, right_img_1)
    left_rect_img_2, right_rect_img_2 = Utilities.RectifyImage(left_img_2, right_img_2)
    
    left_rect_img_stacked, right_rect_img_stacked = Utilities.RectifyImage(left_img_stacked, right_img_stacked)
    
    
    # convert pcd points to numpy array
    points_1 = np.asarray(pcd_1.points)
    points_2 = np.asarray(pcd_2.points)
    
    # create 3xpoint_count array
    scan_1 = np.transpose(points_1) 
    scan_2 = np.transpose(points_2)
    
    shape = scan_1.shape
    
    scans_stacked = np.zeros((2, shape[0], shape[1]))
    
    scans_stacked[0,:,:] = scan_1
    scans_stacked[1,:,:] = scan_2
    
    projected_points_left_image_1, projected_points_left_image_points_1 = Utilities.ProjectLidarToLeftImage(scan_1)
    projected_points_left_image_2, projected_points_left_image_points_2 = Utilities.ProjectLidarToLeftImage(scan_2)
    
    projected_points_left_image_stacked, projected_points_left_image_points_stacked = Utilities.ProjectLidarToLeftImage(scans_stacked)
    
    
    projected_points_left_rectified_image_1, projected_points_left_rectified_image_points_1 = Utilities.ProjectLidarToRectifiedLeftImage(scan_1)
    projected_points_left_rectified_image_2, projected_points_left_rectified_image_points_2 = Utilities.ProjectLidarToRectifiedLeftImage(scan_2)
    
    projected_points_left_rectified_image_stacked, projected_points_left_rectified_image_points_stacked = Utilities.ProjectLidarToRectifiedLeftImage(scans_stacked)
    
    disparity_ground_truth_1 = Utilities.DepthtoDisparityImage(projected_points_left_rectified_image_1)
    disparity_ground_truth_2 = Utilities.DepthtoDisparityImage(projected_points_left_rectified_image_2)
    
    disparity_ground_truth_stacked = Utilities.DepthtoDisparityImage(projected_points_left_rectified_image_stacked)
    
    true_depth_from_disparity = Utilities.DisparityToDepthImage(disparity_ground_truth_1)
    
    
    

    
    left_rect_img_np_1 = np.array(left_rect_img_1)
    right_rect_img_np_1 = np.array(right_rect_img_1)
    
    left_rect_img_np_1 = np.transpose(left_rect_img_np_1, (1,2,0))
    right_rect_img_np_1 = np.transpose(right_rect_img_np_1, (1,2,0))
    
    left_rect_img_np_2 = np.array(left_rect_img_2)
    right_rect_img_np_2 = np.array(right_rect_img_2)
    
    left_rect_img_np_2 = np.transpose(left_rect_img_np_2, (1,2,0))
    right_rect_img_np_2 = np.transpose(right_rect_img_np_2, (1,2,0))
    
    # (x, y) = Utilities.rig_parameters["image_size"]
    # depth_image_np = np.array(depth_from_disparity[0,:,:])
    # locations = np.where(depth_image_np > 0)
    # values = depth_image_np[locations]
    
    # error_overlaid = plt.figure()
    # plt.imshow(left_rect_img_np_1)
    # plt.scatter(locations[1], locations[0], c=values, marker="o",  s=6, vmin = 0, vmax=100) # vmax = 250)
    # cb = plt.colorbar()
    # cb.set_label("meters, inverted")
    
    #error_overlaid.show()
    
    
    
    
    imgR_gray_1 = cv.cvtColor(left_rect_img_np_1, cv.COLOR_RGB2GRAY)
    imgL_gray_1 = cv.cvtColor(right_rect_img_np_1, cv.COLOR_RGB2GRAY)
    
    imgR_gray_2 = cv.cvtColor(left_rect_img_np_2, cv.COLOR_RGB2GRAY)
    imgL_gray_2 = cv.cvtColor(right_rect_img_np_2, cv.COLOR_RGB2GRAY)
    
    stereo_matcher = cv.StereoSGBM.create(minDisparity=0, numDisparities = 256, blockSize=10, uniquenessRatio = 3) #  speckleWindowSize = 13, speckleRange = 15) # P1=1, P2=5) #32*3*win_size**2))
    
    predicted_disparity_1 = stereo_matcher.compute(imgL_gray_1,  imgR_gray_1)/16
    predicted_disparity_2 = stereo_matcher.compute(imgL_gray_2,  imgR_gray_2)/16

    
    average_error_1, greater_than_three_1, greater_than_three_percentage_1, disparity_error_image_1 = Utilities.DisparityError(predicted_disparity_1, disparity_ground_truth_1)
    average_error_2, greater_than_three_2, greater_than_three_percentage_2, disparity_error_image_2 = Utilities.DisparityError(predicted_disparity_2, disparity_ground_truth_2)
    
    predicted_depth = Utilities.DisparityToDepthImage(predicted_disparity_1)
    
    average_error, total_error, error_image = Utilities.MeterError(predicted_depth, projected_points_left_rectified_image_1)
    
    # plt.figure()
    # plt.imshow(np.transpose(left_undistorted_image, (1,2,0)))
    # plt.figure()
    # plt.imshow(np.transpose(left_undistorted_image, (1,2,0)))
    # plt.scatter(projected_points_left_image_points_stacked[0,0,:],  projected_points_left_image_points_stacked[0,1,:], c=projected_points_left_image_points_stacked[0, 2,:], marker="o",  s=1, vmin = 0, vmax=100)
    # plt.title("unrectified image: " + str(142) + ".jpg")
    # cb = plt.colorbar()
    # cb.set_label("depth")
    
    left_undistorted = np.array(left_undistorted_images[0,:,:])
    
    left_undistorted = np.transpose(left_undistorted, (1,2,0))
        
    plt.figure()
    plt.imshow(left_undistorted)
    plt.figure()
    plt.imshow(left_undistorted)     
    plt.scatter(projected_points_left_image_points_1[0,:],  projected_points_left_image_points_1[1,:], c=projected_points_left_image_points_1[2,:], marker="o",  s=1, vmin = 0, vmax = 100)
    plt.title("unrectified " + str(142) + ".jpg")
    cb = plt.colorbar()
    cb.set_label("meters")
    
    projected_points_left_image_1
    
    plt.figure()
    plt.imshow(left_rect_img_np_1)
    plt.figure()
    plt.imshow(left_rect_img_np_1)     
    plt.scatter(projected_points_left_rectified_image_points_stacked[0,0,:],  projected_points_left_rectified_image_points_stacked[0,1,:], c=projected_points_left_rectified_image_points_stacked[0, 2,:], marker="o",  s=1, vmin = 0, vmax = 100)
    plt.title(str(142) + ".jpg")
    cb = plt.colorbar()
    cb.set_label("meters")
        
    
    plt.figure()
    plt.imshow(left_rect_img_np_1)
    plt.figure()
    plt.imshow(left_rect_img_np_1)
    plt.scatter(projected_points_left_rectified_image_points_1[0,:],  projected_points_left_rectified_image_points_1[1,:], c=projected_points_left_rectified_image_points_1[2,:], marker="o",  s=1, vmin=0, vmax = 100)
    plt.title(str(142) + ".jpg")    
    cb = plt.colorbar()
    cb.set_label("meters")
    
    
    
    
    left_undistorted = np.array(left_undistorted_image)
    right_undistored = np.array(right_undistorted_image)
    
    left_undistorted = np.transpose(left_undistorted_image, (1,2,0))
    right_undistorted = np.transpose(right_undistorted_image, (1,2,0))
    
    concat_1 = np.concatenate((left_undistorted, right_undistorted), axis=1)    
    plt.figure()
    plt.imshow(concat_1)
    
    
    
    concat_rectified_1 = np.concatenate((left_rect_img_np_1, right_rect_img_np_1), axis=1)
    plt.figure()
    plt.imshow(concat_rectified_1)
    
    plt.figure()
    plt.imshow(predicted_disparity_1)
    plt.figure()
    
    plt.figure()
    plt.imshow(left_rect_img_np_2)
    plt.figure()
    plt.imshow(left_rect_img_np_2)
    plt.scatter(projected_points_left_rectified_image_points_stacked[1,0,:],  projected_points_left_rectified_image_points_stacked[1,1,:], c=projected_points_left_rectified_image_points_stacked[1, 2,:], marker="o",  s=1,vmin=0, vmax=100)
    plt.title(str(1698) + ".jpg")
    cb = plt.colorbar()
    cb.set_label("meters")
    
    plt.figure()
    plt.imshow(left_rect_img_np_2)
    plt.figure()
    plt.imshow(left_rect_img_np_2)
    plt.scatter(projected_points_left_rectified_image_points_2[0,:],  projected_points_left_rectified_image_points_2[1,:], c=projected_points_left_rectified_image_points_2[2,:], marker="o",  s=1, vmin=0, vmax=100)
    plt.title(str(1698) + ".jpg")
    cb = plt.colorbar()
    cb.set_label("meters")
    
    
    concat_rectified_2 = np.concatenate((left_rect_img_np_2, right_rect_img_np_2), axis=1)
    plt.figure()
    plt.imshow(concat_rectified_2)
    
    plt.figure()
    plt.imshow(predicted_disparity_2)
    plt.figure()
    
  

        
    
    

