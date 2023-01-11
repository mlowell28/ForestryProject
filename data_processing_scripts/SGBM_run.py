# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 11:19:37 2022

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




data_folder = "C:\\Users\\lowellm\\Desktop\\forestry_project\\data\\winter_forestry_building\\calibrated_test_set\\"

def main():
    
        Utilities = dataset_utils.StereoRigUtilities()
        Utilities.LoadRigParameters()
        Utilities.CreateRectificationMaps()
        
                
        now = datetime.now()
        date_time_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    
        results_folder = data_folder + "SGBM_results_"+date_time_string+"\\"
        disparity_results_folder = results_folder + "\\disparity_results\\"
        reconstruction_results_folder_3d = results_folder + "\\3d_reconstruction_results\\"
        disparity_error_images_folder = results_folder + "\\disparity_error_images\\"
        disparity_error_images_overlaid_folder = results_folder + "\\disparity_error_images_overlaid\\"
        meters_error_images_folder = results_folder + "\\meter_error_images\\"
        meters_error_images_overlaid_folder = results_folder + "\\meter_error_images_overlaid\\"
        
        os.mkdir(results_folder)
        os.mkdir(disparity_results_folder)
        os.mkdir(reconstruction_results_folder_3d)
        os.mkdir(disparity_error_images_folder)
        os.mkdir(disparity_error_images_overlaid_folder)
        os.mkdir(meters_error_images_folder)
        os.mkdir(meters_error_images_overlaid_folder)
        
        (x, y) = Utilities.rig_parameters["image_size"]
        
        with open(results_folder + "SGBM_results.txt", 'w') as f:
         
            
            file_list = os.listdir(data_folder + "\\camera\\left\\")
            
            file_count = len(file_list)
         
            disparity_average_total_error = 0 
            average_greater_than_pixel_bound = 0
            average_greater_than_pixel_bound_percentage = 0
            
            average_meters_total_error_5_25 = 0
            average_meters_total_error_25_50 = 0
            average_meters_total_error_50_100 = 0
            
            max_disparity = 208
            min_disparity = 0
            blockSize = 9
            
            uniquenessRatio=1
            speckleWindowSize=25
            speckleRange=5
            disp12MaxDiff=20
            win_size = 10
            
            P1=8 * 3 * win_size ** 2
            P2=32 * 3 * win_size ** 2
            
            remove_disparities_overmax = False
            pixel_error_bound = 10
            
            print("Max disparity: " + str(max_disparity) + "\n")
            print("SGBM Block Size: " + str(blockSize) + "\n")
            print("Uniqueness Ratio: "+ str(uniquenessRatio) + "\n")
            print("Speckle Window Size: "+ str(speckleWindowSize) + "\n")
            print("Speckle range: "+ str(speckleRange) + "\n")
            print("disp12MaxDiff: "+ str(disp12MaxDiff) + "\n")
            print("P1: " + str(P1)+"\n")
            print("P2: " + str(P2) + "\n")
            
            print("Pixel error bound: " + str(pixel_error_bound) + "\n") 
           
            if remove_disparities_overmax == True:
                print("Removing disparities over max when computing error\n")
            
            f.writelines("Max disparity: " + str(max_disparity) + "\n")
            f.writelines("SGBM Block Size: " + str(blockSize) + "\n")
            f.writelines("Uniqueness Ratio: "+ str(uniquenessRatio) + "\n")
            f.writelines("Speckle Window Size: "+ str(speckleWindowSize) + "\n")
            f.writelines("Speckle range: "+ str(speckleRange) + "\n")
            f.writelines("disp12MaxDiff: "+ str(disp12MaxDiff) + "\n")
            f.writelines("P1: " + str(P1)+"\n")
            f.writelines("P2: " + str(P2) + "\n")
            f.writelines("Pixel error bound: " + str(pixel_error_bound) + "\n")     
            
            if remove_disparities_overmax == True:
                f.writelines("Removing disparities over max when computing pixel error\n")
            

              
            f.writelines("Launching SGBM\n")
            f.writelines("Starting data processing with base folder: " + data_folder  + "\n")            
            
            for index, file in enumerate(file_list):
                
                print("Processing file: " + file + "\n")

                _ = file.split(".")
                
                file_name = _[0]
                file_extension = _[1]
                
                left_img = np.asarray(Image.open(data_folder + "\\camera\\left\\"+ file_name + ".jpg").convert("RGB"))
                right_img = np.asarray(Image.open(data_folder + "\\camera\\right\\"+ file_name + ".jpg").convert("RGB"))
                    
                # load 16 bit uint disparity image
                true_disparity = np.asarray(Image.open(data_folder + "\\true_disparity\\"+ file_name + ".png"))
                
                # divide by 256 to bring back to 256 range
                true_disparity = true_disparity/256
                true_disparity = torch.tensor(true_disparity)                

                # images are in RGB so must convert from RGB2GRAY
                
                left_img_gray = cv.cvtColor(left_img, cv.COLOR_RGB2GRAY)
                right_img_gray = cv.cvtColor(right_img, cv.COLOR_RGB2GRAY)
                
                stereo_matcher = cv.StereoSGBM.create(minDisparity=min_disparity, numDisparities = max_disparity, blockSize=blockSize, uniquenessRatio = uniquenessRatio, speckleWindowSize=speckleWindowSize, speckleRange=speckleRange,
                                                      P1=P1, P2=P2, disp12MaxDiff=disp12MaxDiff) # )
                                
                # divide by 16 as SGBM returns last 4 bits of uint as fractional bits. 
                
                predicted_disparity = stereo_matcher.compute(left_img_gray,  right_img_gray)/16
             
                average_error, pixel_error_count, pixel_error_count_percentage, disparity_error_image = Utilities.DisparityError(predicted_disparity, true_disparity, pixel_error_bound, max_disparity, remove_disparities_overmax)
                
                f.writelines("file: " + file + ", average error: " + str(average_error.item()) + ", pixel error count: " + str(pixel_error_count.item()) + ", pixel error count percentage: " + str(pixel_error_count_percentage.item()) + "\n")
                f.flush()
                
                print("average error: " + str(average_error.item()) + ", pixel error count: " + str(pixel_error_count.item()) + ", pixel error count percentage: " + str(pixel_error_count_percentage.item()) + "\n")
                
  
                disparity_average_total_error += average_error.item()
                average_greater_than_pixel_bound += pixel_error_count.item()
                average_greater_than_pixel_bound_percentage += pixel_error_count_percentage.item()            

                
                # scale by 256 to make last 8 bits decimals
                img_scaled = predicted_disparity*256
                
                # round to int
                img_scaled = np.rint(img_scaled)
                img_scaled = img_scaled.astype(np.uint16)
                
                # create PIL image
                img_scaled = Image.fromarray(img_scaled)
                
                # save as 16 bit PNG file
                
                img_scaled.save(disparity_results_folder + file_name+".png")
                
                # save error image
                disparity_error_image = np.array(disparity_error_image)
                
                # generate plt of error overlayed on image 
                
                disparity_error_figure = Utilities.DisparityErrorScatterPlot(left_img, disparity_error_image, disparity_max = max_disparity)
                disparity_error_figure.suptitle(file_name + "." + file_extension + ", avg disparity error: " + str(round(average_error.item(), 2)))
                disparity_error_figure.savefig(disparity_error_images_overlaid_folder + file_name + ".png")
                
                
                # scale by 256 to make last 8 bits decimals
                disparity_error_image_scaled = disparity_error_image*256
                
                # round to int
                disparity_error_image_scaled = np.rint(disparity_error_image_scaled)
                disparity_error_image_scaled = disparity_error_image_scaled.astype(np.uint16)
                
                # create PIL image
                disparity_error_image_scaled = Image.fromarray(disparity_error_image_scaled)
                
                # save as 16 bit PNG file
                
                disparity_error_image_scaled.save(disparity_error_images_folder + file_name+".png")
                
                # now generate depth images 
                                
                predicted_depth = Utilities.DisparityToDepthImage(predicted_disparity)
                true_depth = Utilities.DisparityToDepthImage(true_disparity)
                 
                # compute 5 to 25 depth error
                average_meters_error_5_25, meters_total_error, meters_error_image = Utilities.DepthError(predicted_depth, true_depth, min_distance = 5, max_distance=25)
                average_meters_total_error_5_25 += average_meters_error_5_25
                
                # error_figure = Utilities.MeterErrorScatterPlot(left_img, meters_error_image, error_max = 25)
                # error_figure.suptitle(file_name + "." + file_extension + ", meter error, distance 5-25: " + str(round(average_meters_error_5_25.item(), 2)))
                # error_figure.savefig(meters_error_images_overlaid_folder + file_name + ".png")
                
                # compute 25 to 50 depth error
                average_meters_error_25_50, meters_total_error, meters_error_image = Utilities.DepthError(predicted_depth, true_depth, min_distance = 25, max_distance=50)
                average_meters_total_error_25_50 += average_meters_error_25_50  
                
                # error_figure = Utilities.MeterErrorScatterPlot(left_img, meters_error_image, error_max = 50)
                # error_figure.suptitle(file_name + "." + file_extension + ", meter error, distance 25-50: " + str(round(average_error.item(), 2)))
                # error_figure.savefig(meters_error_images_overlaid_folder + file_name + ".png")
                
                # compute 50 to 100 depth error
                average_meters_error_50_100, meters_total_error, meters_error_image = Utilities.DepthError(predicted_depth, true_depth, min_distance = 50, max_distance=100)
                average_meters_total_error_50_100 += average_meters_error_50_100  
                
                # error_figure = Utilities.MeterErrorScatterPlot(left_img, meters_error_image, error_max = 100)
                # error_figure.suptitle(file_name + "." + file_extension + ", meter error, distance 50-100: " + str(round(average_error.item(), 2)))
                # error_figure.savefig(meters_error_images_overlaid_folder + file_name + ".png")

                # generate plot of depth error
                
                average_meters_error_5_25, meters_total_error, meters_error_image = Utilities.DepthError(predicted_depth, true_depth, min_distance = 5, max_distance=100)
                
                error_figure = Utilities.MeterErrorScatterPlot(left_img, meters_error_image, error_max = 100)
                error_figure.suptitle(file_name + "." + file_extension + ", depth error, 5m-100m ") #": " + str(round(average_meters_error_5_25.item(), 2)) + str('m'))
                error_figure.savefig(meters_error_images_overlaid_folder + file_name + ".png")
                
                
                f.writelines("file: " + file + ", average depth error 5m-25m: " + str(average_meters_error_5_25.item()) + ", 25m-50m: " + str(average_meters_error_25_50.item())  +  ", error 50m-100m: " + str(average_meters_error_50_100.item()) + "\n")
                f.flush()
                
                print("average depth error, 5m-25m: " + str(average_meters_error_5_25.item()) + ", 25m-50m: " + str(average_meters_error_25_50.item())  +  ", 50m-100m: " + str(average_meters_error_50_100.item()) + "\n")
         
                
                # generate 3d plot
                
                Utilities.Generate3dPlot(predicted_disparity, left_img, reconstruction_results_folder_3d + file_name + ".ply")
    
            disparity_average_total_error = disparity_average_total_error/file_count
            average_greater_than_pixel_bound = average_greater_than_pixel_bound/file_count
            average_greater_than_pixel_bound_percentage = average_greater_than_pixel_bound_percentage/file_count 
        
            average_meters_total_error_5_25 =  average_meters_total_error_5_25/file_count
            average_meters_total_error_25_50 =  average_meters_total_error_25_50/file_count
            average_meters_total_error_50_100 =  average_meters_total_error_50_100/file_count
        
            print("average dataset pixel error: " + str(disparity_average_total_error) + "\n")
            print("average dataset pixel error greater than " + str(pixel_error_bound) + ": " + str(average_greater_than_pixel_bound) + "\n")
            print("average dataset greater than " + str(pixel_error_bound) + " pixel percentage: " + str(average_greater_than_pixel_bound_percentage)+ "\n")
            
                    
            print("average dataset 5-25 meter error: " + str(average_meters_total_error_5_25.item()) + "\n")
            print("average dataset 25-50 meter error: " + str(average_meters_total_error_25_50.item()) + "\n")
            print("average dataset 50-100 meter error: " + str(average_meters_total_error_50_100.item()) + "\n")
            
            f.writelines("average dataset pixel error: " + str(disparity_average_total_error) + "\n")
            f.writelines("average dataset pixel error greater than " + str(pixel_error_bound) + ": " + str(average_greater_than_pixel_bound) + "\n")
            f.writelines("average dataset greater than " + str(pixel_error_bound) + " pixel percentage: " + str(average_greater_than_pixel_bound_percentage)+ "\n")
            
            f.writelines("average dataset 5-25 meter error: " + str(average_meters_total_error_5_25.item()) + "\n")
            f.writelines("average dataset 25-50 meter error: " + str(average_meters_total_error_25_50.item()) + "\n")
            f.writelines("average dataset 50-100 meter error: " + str(average_meters_total_error_50_100.item()) + "\n")
            
            f.close()
            
          

if __name__ == '__main__':
   main()

