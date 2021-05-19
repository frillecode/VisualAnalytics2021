#! /usr/bin/python
"""
Script for loading an image with text, drawing a rectangular ROI around the middle of the image, crop the image to contain only the ROI, perform Canny edge detection and draw green contours around each letter. Each step (expect Canny) will output a .jpg-file.

"""

# Load libraries
import os
import cv2
import numpy as np
import argparse


# Defining function which performs all tasks for edge detection
def edge_detection(image, args):
    """ Function for preproceesing and performing Canny edge detection on an image
    Input:
        image: numpy.ndarray, image data
    """
     
    # 1) Drawing a green rectangle around ROI  
    x1, x2, y1, y2 = [int(str.strip(x)) for x in args['roi_coordinates'].split(',')] #defining coordinates
    # Ensuring that ROI coordinates do not exceed the image size
    if  max(x1,x2) > image.shape[1]:
        sys.exit(f'ERROR: x2 coordinate must not exceed size of image ({image.shape[1]})')
    elif max(y1,y2) > image.shape[0]: 
        sys.exit(f'ERROR: y2 coordinate must not exceed size of image ({image.shape[0]})')
    else:
        image_with_ROI = cv2.rectangle(image.copy(), (x1, y1), (x2, y2), (0,255,0), 3) #drawing rectangle based on coordinates
        cv2.imwrite(os.path.join("out", f"{args['filename_image']}_with_ROI.jpg"), image_with_ROI) #saving roi image


        # 2) Cropping image to contain only the ROI
        image_cropped = image[y1:y2, x1:x2] #slicing based on the coordinates from the ROI
        cv2.imwrite(os.path.join("out", f"{args['filename_image']}_cropped.jpg"), image_cropped) #saving cropped image


        # 3) Apply Canny edge detection
        grey = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY) #turning image to greyscale
        blurred = cv2.GaussianBlur(grey, (5,5), 0) #blurring the image to remove noise. Here, we use Gaussian blur
        canny = cv2.Canny(blurred, #canny edge detection
                          [int(str.strip(x)) for x in args['canny_thresholds'].split(',')][0], #min threshold 
                          [int(str.strip(x)) for x in args['canny_thresholds'].split(',')][1]) #max threshold


        # 4) Draw green contours around letters in the cropped image
        (cnts, _) = cv2.findContours(canny.copy(),        #finding contours
                                 cv2.RETR_EXTERNAL, 
                                 cv2.CHAIN_APPROX_SIMPLE)
        image_letters = cv2.drawContours(image_cropped.copy(), cnts, -1, (0, 255, 0), 1) #drawing contours on original image
        cv2.imwrite(os.path.join("out", f"{args['filename_image']}_letters.jpg"), image_letters) #saving image with contours

                      
                      
def main():
    ap = argparse.ArgumentParser(description="[INFO] This script takes image data and performs Canny edge detection to extract text from the image.")
    # Argument for specifying path to image file
    ap.add_argument("-fi", 
                "--filename_image", 
                required=False, 
                type=str,
                default="_We_Hold_These_Truths__at_Jefferson_Memorial_IMG_4729.jpeg",
                help="str, filename of input image (input file must be in ../../data/assignment3/)") 
    # Argument for specifying coordinates of ROI
    ap.add_argument("-rc",  
                "--roi_coordinates", 
                required=False,  
                type=str,
                default="1400, 2860, 870, 2800",
                help='str, string with coordinates for ROI in the order x1, x2, y1, y2 (e.g.: -rc "1400, 2860, 870, 2800")')
    # Argument for specifying min and max thresholds for Canny edge detection
    ap.add_argument("-ct",  
            "--canny_thresholds", 
            required=False,  
            type=str,
            default="100, 150",
            help='str, string with values for min and max threshold for Canny edge detection (e.g.: -ct  "100, 150")')

    args = vars(ap.parse_args())
    
                     
    # Load image
    image = cv2.imread(os.path.join("..","..","data","project1", f"{args['filename_image']}"))
    

    # Run script
    edge_detection(image, args = args)
    
    
# Define behaviour when called from command line
if __name__=="__main__":
    main()
    print("[INFO] DONE! You can find the results in the output-folder")