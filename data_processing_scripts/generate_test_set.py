# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 17:45:37 2022

@author: lowellm
"""

import os
import shutil

data_folder = "C:\\Users\\lowellm\\Desktop\\forestry_project\\data\\winter_forestry_building\\"

os.mkdir(data_folder + "\\calibrated_test_set\\")
os.mkdir(data_folder + "\\calibrated_test_set\\camera\\")
os.mkdir(data_folder + "\\calibrated_test_set\\camera\\left")
os.mkdir(data_folder + "\\calibrated_test_set\\camera\\right")
os.mkdir(data_folder + "\\calibrated_test_set\\true_disparity\\")
os.mkdir(data_folder + "\\calibrated_test_set\\lidar\\")


shutil.copy(data_folder + "\\rectified_camera\\rectified_stereo_parameters.txt", data_folder + "\\calibrated_test_set\\camera\\rectified_stereo_parameters.txt")

for index in range(3,3000):
    
    
    shutil.copy(data_folder + "\\rectified_camera\\left\\" + str(index*10) + ".jpg", data_folder + "\\calibrated_test_set\\camera\\left\\" + str(index-3) + ".jpg")
    shutil.copy(data_folder + "\\rectified_camera\\right\\" + str(index*10) + ".jpg", data_folder + "\\calibrated_test_set\\camera\\right\\" + str(index-3) + ".jpg")
    shutil.copy(data_folder + "\\true_disparity\\" + str(index*10) + ".png", data_folder + "\\calibrated_test_set\\true_disparity\\" + str(index-3) + ".png")
    shutil.copy(data_folder + "\\lidar\\" + str(index*10) + ".pcd", data_folder + "\\calibrated_test_set\\lidar\\" + str(index-3) + ".pcd")
    
