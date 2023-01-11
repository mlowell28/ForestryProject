# -*- coding: utf-8 -*-
"""exit
Created on Mon Jan 31 22:03:01 2022

@author: Michael
"""

import numpy as np
import math
import socket 
import sys
import os
import multiprocessing
import time

import VLP16CaptureScript
import MultiprocessCameraCapture


mastercam_serial = '20162694'
slavecam_serial = '20162696'


def main(): 
    print("Capture Program")
    print("Enter \"start_capture_continuous\" to start continuous capture.")
    print("Enter \"start_capture_frame\" to capture individual frames.")
    print("Required parameter: folder_name=\"name of folder\".")
    print("Optional parameter: path=\"path to folder\".")
    print("Will place data folder in script directory within collected_data folder if no path given.")
    print("Optional parameter: fps=\"fps\", top fps is 10 for desktop 5 for laptop")
    print("Enter \"exit\" to exit program.")
    
    state = None
    while(state != "exit"):
               
            path = ""
            folder_name = None
            fps = 5
            
            try:
                print("Enter Command")
                input_string = input()
                tokens = input_string.split(' ')
                
                # parse input
                if tokens[0] == "start_capture_continuous" or tokens[0] == "start_capture_frame":
                    
                    # get path
                    for token in tokens:
                        if token.startswith("path="):
                            path = token[len("path="):]
                        
                        if token.startswith("folder_name="):
                            folder_name = token[len("folder_name="):]
                            
                        if token.startswith("fps="):
                            fps = int(token[len("fps="):])
                             
                    if folder_name == None:
                        print("No folder name given, enter again.")
                        
                    else:
                        
                        if tokens[0] == "start_capture_continuous":
                            return_command = continuous_capture(folder_name, path, fps)
                        
                        if tokens[0] == "start_capture_frame":
                            return_command = frame_capture(folder_name, path, fps)
                            
                        if return_command == "exit":
                            state = "exit"
                                                             
                elif tokens[0] == "exit":
                    state = "exit"
                    
                else:
                    print("No valid command entered.")
                    print("Enter \"start_capture_continuous\" to start continuous capture.")
                    print("Enter \"start_capture_frame\" to capture individual frames.")
                    print("required parameter: folder_name=\"name of folder\"")
                    print("optional parameter: path=\"path to folder\"")
                    print("optional parameter: fps=\"fps\", top fps is 10 for desktop 5 for laptop")
                    print("Enter \"exit\" to exit program.")
                
            except ValueError as error:
                print("Format error " + str(error))
                
            except OSError as error:
                print("OS Error " + str(error))
                
            except SystemError as error:
                print("System error" + str(error))
            
            except RuntimeError as error:
                print("Runtime error " + str(error))
                

    print("Exiting program.")
    sys.exit()
             

def continuous_capture(folder_name, path = "", fps=5):
    
    if path != "":
        base_folder = path + folder_name
        lidar_folder = path + folder_name + "\\lidar\\"
        camera_folder = path + folder_name + "\\camera\\"
    else:
        path = os.getcwd() + "\\" 
        
        if os.path.isdir(path+"\\collected_data\\") != True:
            os.mkdir(path + "\\collected_data\\")
 
        base_folder = path + "\\collected_data\\" + folder_name
        lidar_folder = path + "\\collected_data\\" + folder_name + "\\lidar\\"
        camera_folder = path + "\\collected_data\\" + folder_name + "\\camera\\"
        

    os.mkdir(base_folder)
    os.mkdir(lidar_folder)
    os.mkdir(camera_folder)
        
    print("Creating Lidar and Camera Processes.")
    
    UI_to_lidar, lidar_to_UI =  multiprocessing.Pipe()            
    mastercam_to_lidar, lidar_to_mastercam = multiprocessing.Pipe()
    
    lidar_process = multiprocessing.Process(target=VLP16CaptureScript.run_lidar, args=(lidar_folder, fps, lidar_to_UI, lidar_to_mastercam))
    cam_process = multiprocessing.Process(target=MultiprocessCameraCapture.run_cameras, args=(mastercam_serial, slavecam_serial, camera_folder, mastercam_to_lidar))
    
    print("Starting Lidar process.")
    lidar_process.start()
    print("Starting camera process")
    cam_process.start()
    time.sleep(2)
    
    print("Saving data at location: " + base_folder)
    
    # launch capture threads and enter capture loop
    
    print("Launching continuous capture routine.")
    print("Enter \"pause\" to pause.")
    print("Enter \"stop\" to stop capture.")
    print("Enter \"exit\" to exit program.")

    # set processes state to continuous capture
    state = "continuous_capture"
    UI_to_lidar.send("continuous_capture")
    
    while((state!= "stop_capture") and (state!="exit")):
    
        input_string = input()
        
        if input_string == "pause":
            print("Pausing capture.")
            print("Enter \"continue\" to restart.")
            print("Enter \"stop\" to stop the capture sequence.")
            print("Enter \"exit\" to quit program.")
            UI_to_lidar.send("pause")
            state = "pause"
            
        elif input_string == "continue": 
            if state == "conintuous_capture":
                print("Continuous capture already running.")         
            else: 
                print("Continuing capture.")
                UI_to_lidar.send("continuous_capture")
                state = "continuous_capture"  
                   
        elif input_string == "stop":
            print("Stopping capture.")
            UI_to_lidar.send("exit")
            state = "stop_capture"
            
        elif input_string == "exit":
            print("Stopping capture.")
            UI_to_lidar.send("exit")
            state = "exit"
                                   
        else:
            print("Invalid command.")
            if state == "pause":
                print("Capture is paused.")
                print("Enter \"continue\" to restart.")
                print("Enter \"stop\" to stop capture sequence.")
                print("Enter \"exit\" to quit program.")
                
            if state == "continuous_capture":
                print("Capture process is running.")
                print("Enter \"pause\" to pause.")
                print("Enter \"stop\" to stop capture.")
                print("Enter \"exit\" to exit program.")
        
    # wait until processes exit then return 
    lidar_process.join()
    cam_process.join()
    return state
    
def frame_capture(folder_name, path = "", fps=5):
    
    
    if path != "":
        base_folder = path + folder_name
        lidar_folder = path + folder_name + "\\lidar\\"
        camera_folder = path + folder_name + "\\camera\\"
    else:
        path = os.getcwd() + "\\" 
        
        if os.path.isdir(path+"\\collected_data\\") != True:
            os.mkdir(path + "\\collected_data\\")
 
        base_folder = path + "\\collected_data\\" + folder_name
        lidar_folder = path + "\\collected_data\\" + folder_name + "\\lidar\\"
        camera_folder = path + "\\collected_data\\" + folder_name + "\\camera\\"
        

    os.mkdir(base_folder)
    os.mkdir(lidar_folder)
    os.mkdir(camera_folder)
    
    print("Saving data at location: " + base_folder)
    
    # launch capture threads and enter capture loop

    print("Creating lidar and camera processes\n")
    UI_to_lidar, lidar_to_UI =  multiprocessing.Pipe()            
    mastercam_to_lidar, lidar_to_mastercam = multiprocessing.Pipe()
    
    lidar_process = multiprocessing.Process(target=VLP16CaptureScript.run_lidar, args=(lidar_folder, fps, lidar_to_UI, lidar_to_mastercam))
    cam_process = multiprocessing.Process(target=MultiprocessCameraCapture.run_cameras, args=(mastercam_serial, slavecam_serial, camera_folder, mastercam_to_lidar))
    
    print("Starting Lidar process.")
    lidar_process.start()
    print("Starting camera process")
    cam_process.start()
    
    time.sleep(2)

    # set processes state to continuous capture
    
    print("Press enter to save current image.")
    print("Enter \"stop\" to stop capture.")
    print("Enter \"exit\" to exit program.")
    
    frame_number = 0
    state = None
    
    while((state!= "stop_capture") and (state!="exit")):
    
        print("hit enter to capture next frame")
        input_string = input()
        
        if input_string =="":
            print("Capturing frame: " +str(frame_number))
            UI_to_lidar.send("frame_capture")
            frame_number +=1

        elif input_string == "stop":
            print("Stopping capture.")
            state = "stop_capture"
            UI_to_lidar.send("exit")
            
        elif input_string == "exit":
            print("Stopping capture.")
            state = "exit"
            UI_to_lidar.send("exit")
                                   
        else:
            print("Invalid command.")
            print("Press enter to save current image.")
            print("Enter \"stop\" to stop capture.")
            print("Enter \"exit\" to exit program.")
        
            
    # wait until processes exit then return 
    lidar_process.join()
    cam_process.join()
    return state
        
if __name__ == "__main__":
    main()       
       
                        
   
                