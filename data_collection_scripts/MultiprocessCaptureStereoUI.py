# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 22:03:01 2022

@author: Michael
"""

import numpy as np
import math
import sys
import os
import multiprocessing
import time

import MultiprocessCameraCapture



mastercam_serial = '20162694'
slavecam_serial = '20162696'


#global logfile
logfile = None
   
def run_camera_controller(folder_path, control_pipe, cam_pipe, fps = 10):
    pause_time = int((1/fps)*10**9)
    global logfile
    logfile = open(folder_path + "\\CameraControllerLog.txt", "w")    
    logfile.write("target FPS for continuous capture: " + str(fps) + '\n')
    
    try :
        frame_count = 0
        
        # starting state
        state = "pause"
        
      
        while(state != "exit"):
            
            if control_pipe.poll() == True:
                command = control_pipe.recv()
                
                # capture a single frame
                if command == "frame_capture":
                    state = "frame_capture"
                    starting_time = time.time_ns()
                    target_time = starting_time
                    logfile.write(str(time.time_ns()) + ": command received, state changed to frame_capture\n")
                # capture frames continuously
                elif command == "continuous_capture":
                    state = "continuous_capture"
                    starting_time = time.time_ns()
                    target_time = starting_time
                    logfile.write(str(time.time_ns()) + ": command received, state changed to continuous_capture\n")
                    
                # pause capture
                elif command == "pause":
                    state = "pause"
                    logfile.write(str(time.time_ns()) + ": command received, state changed to pause\n")
                # exit process
                elif command == "exit":
                    state = "exit"
                    logfile.write(str(time.time_ns()) + ": command received, state changed to exit\n")
                    
                else:
                    logfile.write(str(time.time_ns()) + ": unknown command received: " + str(command) + '\n')
            
            if state == "frame_capture" or state == "continuous_capture":
                
                # launch camera capture once frame recording starts.
                
                current_time = time.time_ns()  
                if state == "continuous_capture":
                          
                    while(current_time  < target_time):
                        current_time = time.time_ns()
                    
                cam_pipe.send("capture_image")
                logfile.write(str(current_time) + ": Sent capture_image command to mastercam\n")
                logfile.write(str(time.time_ns()) + ": Time Error: " + str(current_time - target_time) + "\n")
            
                frame_count +=1 
                target_time += pause_time
                    
                # after single frame put back in pause state 
                if state == "frame_capture":
                    state = "pause"
            
            elif state == "pause":
                continue
            
        logfile.write(str(time.time_ns()) + ": captured frames: " + str(frame_count) + "\n")
        return_value = frame_count

    finally:
        logfile.write(str(time.time_ns()) + ": closing master camera capture process\n")
        cam_pipe.send("exit")
        logfile.close()
        
        return return_value       

def main(): 
    print("Capture Program")
    print("Enter \"start_capture_continuous\" to start continuous capture.")
    print("Enter \"start_capture_frame\" to capture individual frames.")
    print("Required parameter: folder_name=\"name of folder\".")
    print("Optional parameter: path=\"path to folder\".")
    print("Will place folder in script directory if no path given.")
    print("Enter \"exit\" to exit program.")
    
    state = None
    fps = 10 #default FPS
    
    while(state != "exit"):
               
            path = ""
            folder_name = None
            
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
                            fps = float(token[len("fps="):])
                             
                    if folder_name == None:
                        print("No folder name given, enter again.")
                        
                    else:
                        
                        if tokens[0] == "start_capture_continuous":
                            return_command = continuous_capture(folder_name, path, fps)
                        
                        if tokens[0] == "start_capture_frame":
                            return_command = frame_capture(folder_name, path)
                            
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
             

def continuous_capture(folder_name, path = "", fps=10):
        
    if path != "":
        base_folder = path + folder_name
        camera_folder = path + folder_name + "\\camera\\"
    else:
        path = os.getcwd() + "\\" 
        
        if os.path.isdir(path+"\\collected_data\\") != True:
            os.mkdir(path + "\\collected_data\\")
        
        base_folder = path + "\\collected_data\\" + folder_name
        camera_folder = path+"\\collected_data\\" + folder_name + "\\camera\\" 
 
    os.mkdir(base_folder)
    os.mkdir(camera_folder)
    
    print("Saving data at location: " + base_folder)
    
    # launch capture threads and enter capture loop
    
    print("Launching continuous capture routine.")
    print("Enter \"pause\" to pause.")
    print("Enter \"stop\" to stop capture.")
    print("Enter \"exit\" to exit program.")
        
    print("Camera Processes.")
    
    UI_to_camcontroller, camcontroller_to_UI =  multiprocessing.Pipe()            
    mastercam_to_camcontroller, camcontroller_to_mastercam = multiprocessing.Pipe()
    
    trigger_process = multiprocessing.Process(target=run_camera_controller, args=(camera_folder, camcontroller_to_UI, camcontroller_to_mastercam, fps))
    
    cam_process = multiprocessing.Process(target=MultiprocessCameraCapture.run_cameras, args=(mastercam_serial, slavecam_serial, camera_folder, mastercam_to_camcontroller))
    
    print("Starting Camera processes.")
    
    trigger_process.start()
    cam_process.start()

    # add time for all process to load
    time.sleep(2)

    # set processes state to continuous capture
    state = "continuous_capture"
    UI_to_camcontroller.send("continuous_capture")
    
    while((state!= "stop_capture") and (state!="exit")):
    
        input_string = input()
        
        if input_string == "pause":
            print("Pausing capture.")
            print("Enter \"continue\" to restart.")
            print("Enter \"stop\" to stop the capture sequence.")
            print("Enter \"exit\" to quit program.")
            UI_to_camcontroller.send("pause")
            state = "pause"
            
        elif input_string == "continue": 
            if state == "conintuous_capture":
                print("Continuous capture already running.")         
            else: 
                print("Continuing capture.")
                UI_to_camcontroller.send("continuous_capture")
                state = "continuous_capture"  
                   
        elif input_string == "stop":
            print("Stopping capture.")
            UI_to_camcontroller.send("exit")
            state = "stop_capture"
            
        elif input_string == "exit":
            print("Stopping capture.")
            UI_to_camcontroller.send("exit")
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
    trigger_process.join()
    cam_process.join()
    return state
    
def frame_capture(folder_name, path = ""):
    
    if path != "":
        base_folder = path + folder_name
        camera_folder = path + folder_name + "\\camera\\"
    else:
        path = os.getcwd() + "\\" 
        
        if os.path.isdir(path+"\\collected_data\\") != True:
            os.mkdir(path + "\\collected_data\\")
        
        base_folder = path + "\\collected_data\\" + folder_name
        camera_folder = path+"\\collected_data\\" + folder_name + "\\camera\\" 
 
    os.mkdir(base_folder)
    os.mkdir(camera_folder)
    
    
    print("Saving data at location: " + base_folder)
    
    # launch capture threads and enter capture loop

    UI_to_camcontroller, camcontroller_to_UI =  multiprocessing.Pipe()            
    mastercam_to_camcontroller, camcontroller_to_mastercam = multiprocessing.Pipe()
    
    print("Creating camera processes\n")
    
    trigger_process = multiprocessing.Process(target=run_camera_controller, args=(camera_folder, camcontroller_to_UI, camcontroller_to_mastercam))
    cam_process = multiprocessing.Process(target=MultiprocessCameraCapture.run_cameras, args=(mastercam_serial, slavecam_serial, camera_folder, mastercam_to_camcontroller))
        
    print("Starting Camera processes.")
    
    trigger_process.start()
    cam_process.start()
    
    # add time for all process to load
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
            UI_to_camcontroller.send("frame_capture")
            frame_number +=1

        elif input_string == "stop":
            print("Stopping capture.")
            state = "stop_capture"
            UI_to_camcontroller.send("exit")
            
        elif input_string == "exit":
            print("Stopping capture.")
            state = "exit"
            UI_to_camcontroller.send("exit")
                                   
        else:
            print("Invalid command.")
            print("Press enter to save current image.")
            print("Enter \"stop\" to stop capture.")
            print("Enter \"exit\" to exit program.")
        
            
    # wait until processes exit then return 
    trigger_process.join()
    cam_process.join()
    return state
        
if __name__ == "__main__":
    main()       
       
                        
   
                