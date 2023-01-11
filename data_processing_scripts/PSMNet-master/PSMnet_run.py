from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from models import *
import cv2
from PIL import Image
import matplotlib.pyplot as plt

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
from datetime import datetime
import sys

sys.path.insert(0, "C:\\Users\\lowellm\\Desktop\\forestry_project\\data_processing_scripts\\")

import dataset_utils

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--leftimg', default= "C:\\Users\\lowellm\\Desktop\\StereoRigScripts\\Capture_Program\\left_rect.jpg",
                    help='load model')
parser.add_argument('--rightimg', default= "C:\\Users\\lowellm\\Desktop\\StereoRigScripts\\Capture_Program\\right_rect.jpg",
                    help='load model')                                      
parser.add_argument('--model', default='stackhourglass',
                    help='select model')
parser.add_argument('--maxdisp', type=int, default=256,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.model == 'stackhourglass':
    model = stackhourglass(args.maxdisp)
elif args.model == 'basic':
    model = basic(args.maxdisp)
else:
    print('no model')

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load PSMNet')
    state_dict = torch.load(args.loadmodel)
    model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        if args.cuda:
           imgL = imgL.cuda()
           imgR = imgR.cuda()     

        with torch.no_grad():
            disp = model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp



data_folder = "C:\\Users\\lowellm\\Desktop\\forestry_project\\data\\winter_forestry_building\\calibrated_test_set\\"
#base_folder = "C:\\Users\\lowellm\\Desktop\\winter_data\\winter_forestry_building\\calibrated_test_set_small\\"

def main():
    
        Utilities = dataset_utils.StereoRigUtilities()
        Utilities.LoadRigParameters()
        Utilities.CreateRectificationMaps()
        
                
        now = datetime.now()
        date_time_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    
        results_folder = data_folder + "PSMnet_results_"+date_time_string+"\\"
        disparity_results_folder = results_folder + "\\disparity_results\\"
        reconstruction_results_folder_3d = results_folder + "\\3d_reconstruction_results\\"
        disparity_error_images_folder = results_folder + "\\error_images\\"
        disparity_error_images_overlaid_folder = results_folder + "\\error_images_overlaid\\"
        meters_error_images_folder = results_folder + "\\meter_error_images\\"
        meters_error_images_overlaid_folder = results_folder + "\\meter_error_images_overlaid\\"
        
        os.mkdir(results_folder)
        os.mkdir(disparity_results_folder)
        os.mkdir(reconstruction_results_folder_3d)
        os.mkdir(disparity_error_images_folder)
        os.mkdir(disparity_error_images_overlaid_folder)
        os.mkdir(meters_error_images_folder)
        os.mkdir(meters_error_images_overlaid_folder)
        
        
        with open(results_folder + "PSMnet_results.txt", 'w') as f:
                       
            f.writelines("Launching PSMnet\n")
            f.writelines("Starting data processing with base folder: " + data_folder  + "\n")
            
            file_list = os.listdir(data_folder + "\\camera\\left\\")
            
            file_count = len(file_list)
            
            normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                                'std': [0.229, 0.224, 0.225]}
            infer_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(**normal_mean_var)])
            disparity_average_total_error = 0 
            average_greater_than_pixel_bound = 0
            average_greater_than_pixel_bound_percentage = 0
            
            average_meters_total_error_5_25 = 0
            average_meters_total_error_25_50 = 0
            average_meters_total_error_50_100 = 0
            
            
            max_disparity = 208
            remove_disparities_overmax = False 
            pixel_error_bound = 10
            
            print("Max disparity: " + str(max_disparity) + "\n")
            print("Pixel error bound: " + str(pixel_error_bound) + "\n") 
            
            if remove_disparities_overmax == remove_disparities_overmax:
                print("Removing disparities over max\n") 
                
            
            f.writelines("Max disparity: " + str(max_disparity) + "\n")
            f.writelines("Pixel error bound: " + str(pixel_error_bound) + "\n")   
              
            if remove_disparities_overmax == remove_disparities_overmax:
                f.writelines("Removing disparities over max\n") 
                
            for index, file in enumerate(file_list):
                
                print("Processing file: " + file + "\n")

                _ = file.split(".")
                
                file_name = _[0]
                file_extension = _[1]
    
                left_img = np.asarray(Image.open(data_folder + "\\camera\\left\\"+ file_name + ".jpg").convert("RGB"))
                right_img = np.asarray(Image.open(data_folder + "\\camera\\right\\"+ file_name + ".jpg").convert("RGB"))
                
                # data is in u16 format where last 8 bits are decimals, divide by 256 to bring it back to 0 to 256 range
                
                true_disparity = np.asarray(Image.open(data_folder + "\\true_disparity\\"+ file_name + ".png"))/256
                
                true_disparity = torch.tensor(true_disparity)
                true_disparity = true_disparity.cuda()
                
                imgL = infer_transform(left_img)
                imgR = infer_transform(right_img) 
                
                # pad to width and hight to 16 times
                if imgL.shape[1] % 16 != 0:
                    times = imgL.shape[1]//16       
                    top_pad = (times+1)*16 -imgL.shape[1]
                else:
                    top_pad = 0
        
                if imgL.shape[2] % 16 != 0:
                    times = imgL.shape[2]//16                       
                    right_pad = (times+1)*16-imgL.shape[2]
                else:
                    right_pad = 0    
        
                imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
                imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)
        
                start_time = time.time()
                pred_disp = test(imgL,imgR)
                print('time = %.2f' %(time.time() - start_time))
            
                
                if top_pad !=0 and right_pad != 0:
                    img = pred_disp[top_pad:,:-right_pad]
                elif top_pad ==0 and right_pad != 0:
                    img = pred_disp[:,:-right_pad]
                elif top_pad !=0 and right_pad == 0:
                    img = pred_disp[top_pad:,:]
                else:
                    img = pred_disp
                    
                
                img = torch.tensor(img)
                img = img.cuda()
                average_error, pixel_error_count, pixel_error_count_percentage, disparity_error_image = Utilities.DisparityError(img, true_disparity, pixel_error_bound, max_disparity, remove_disparities_overmax)
                
                f.writelines("file: " + file + ", average error: " + str(average_error.item()) + ", pixel error count: " + str(pixel_error_count.item()) + ", pixel error count percentage: " + str(pixel_error_count_percentage.item()) + "\n")
                f.flush()
                
                print("average error: " + str(average_error.item()) + ", pixel error count: " + str(pixel_error_count.item()) + ", pixel error count percentage: " + str(pixel_error_count_percentage.item()) + "\n")

                
                disparity_average_total_error += average_error.item()
                average_greater_than_pixel_bound += pixel_error_count.item()
                average_greater_than_pixel_bound_percentage += pixel_error_count_percentage.item()            
                
                # save disparity image
                img = np.array(img.cpu())
                
                # scale by 256 to make last 8 bits decimals
                img_scaled = img*256
                
                # round to int
                img_scaled = np.rint(img_scaled)
                img_scaled = img_scaled.astype(np.uint16)
                
                # create PIL image
                img_scaled = Image.fromarray(img_scaled)
                
                # save as 16 bit PNG file
                
                img_scaled.save(disparity_results_folder + file_name+".png")
                
                # save error image
                disparity_error_image = np.array(disparity_error_image.cpu())
                
                # generate plt of error overlayed on image 
                
                error_figure = Utilities.DisparityErrorScatterPlot(left_img, disparity_error_image, disparity_max = max_disparity)
                error_figure.suptitle(file_name + "." + file_extension + ", disparity error: " + str(round(average_error.item(), 2)))
                error_figure.savefig(disparity_error_images_overlaid_folder + file_name + ".png")
                
                # scale by 256 to make last 8 bits decimals
                disparity_error_image_scaled = disparity_error_image*256
                
                # round to int
                disparity_error_image_scaled = np.rint(disparity_error_image_scaled)
                disparity_error_image_scaled = disparity_error_image_scaled.astype(np.uint16)
                
                # create PIL image
                disparity_error_image_scaled = Image.fromarray(disparity_error_image_scaled)
                
                # save as 16 bit PNG file
                
                disparity_error_image_scaled.save(disparity_error_images_folder + file_name+".png")
                
                
                predicted_depth = Utilities.DisparityToDepthImage(img)
                true_depth = Utilities.DisparityToDepthImage(true_disparity)
                 
                true_depth = true_depth.cpu()
                
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
                
                Utilities.Generate3dPlot(img, left_img, reconstruction_results_folder_3d + file_name + ".ply")
    
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






