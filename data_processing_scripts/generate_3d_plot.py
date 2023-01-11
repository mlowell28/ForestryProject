# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 18:34:14 2022

@author: lowellm
"""

import cv2 as cv
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.image
import math
import json
from PIL import Image
import torch
import torchvision
from datetime import datetime

import dataset_utils


base_folder = "C:\\Users\\lowellm\\Desktop\\Timberhill_test\\calibrated_test_set\\"


Utilities = dataset_utils.StereoRigUtilities()
Utilities.LoadParameters()

#Function to create point cloud file
def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

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

a = 10