# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 09:24:30 2017

@author: Admin
"""

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import pickle
from functions5 import *
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def check_box(new_box):
    
    # finds index of already detected box in the vicinity (dist_thr) of the new box 
    
    dist_thr = 15
    min_distance = 1000
   
    
    n = -1 # index of closest box in the past
    i = 0 # loop index
    min_i = -1 # index of the minimal element
    
    # go over all past boxes
    for box in v.boxes:
        new_boxm = ((new_box[0][0] + new_box[1][0])//2, (new_box[0][1] + new_box[1][1])//2)
        boxm = ((box[0][0] + box[1][0])//2, (box[0][1] + box[1][1])//2)
        # find distance between the center of the new box and an old box
        distance = ((new_boxm[0] - boxm[0])**2 + (new_boxm[1] - boxm[1])**2)**0.5
        if distance < min_distance:
           min_distance = distance
           min_i = i
        i = i + 1   
    if min_distance < dist_thr: 
       # only if the distance is lower than dist_thr return index of the old box
       n = min_i    
       
    return n
 
def draw_labeled_bboxes(img, labels):
    
    # draws boxes in the image
    # only boxes already detected in the past 
    # with  size and probability over the thresholds are drawn
    
    MINXDIST = 80 # minimuim x length
    MINYDIST = 80 # minimuim y length
    
   
    boxes = []
    # update boxprobabilities with moving average filter
    for k in range(np.shape(v.probs)[0]):
        v.probs[k] = 0.8*v.probs[k]
        v.ndet[k] = max(0,v.ndet[k] - 0.2)
     # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        
        distx = abs((bbox[1][0] - bbox[0][0]))
        disty = abs((bbox[1][1] - bbox[0][1]))
        if distx > MINXDIST and disty > MINYDIST:
           # draw only boxes with sides large than MINXDIST and MINYDIST
           n = check_box(bbox)
           if n > -1: 
               # box already identified in the past
               # average box shape
              v.boxes[n] = (((bbox[0][0] + v.boxes[n][0][0])//2, (bbox[0][1] + v.boxes[n][0][1])//2),((bbox[1][0] + v.boxes[n][1][0])//2, (bbox[1][1] + v.boxes[n][1][1])//2))                        
              # update box probability
              v.probs[n] = v.probs[n] + 0.2
              
              # update number of prediction for the box n
              v.ndet[n] = v.ndet[n] + 1
              # draw only boxes with higher than 0.4 probability
              if  v.probs[n] > 0.4 and v.ndet[n] > 4: 
                  cv2.rectangle(img, v.boxes[n][0], v.boxes[n][1], (0,0,255), 6)
              
           
           else:
              boxes.append(bbox)
              # add new box probabilities and number of detections
              v.probs.append(0.2)
              v.ndet.append(1) 
    
    #for p,box in zip(v.probs,v.boxes):
      #  if p > 2 and p/v.n>0.4:
        #   cv2.rectangle(img, box[0], box[1], (0,0,255), 6)              
    v.boxes = boxes
    return img

class Vehicles():
     def __init__(self):       
        self.probs = []  # probabilities of each vehicle detections       
        self.boxes = []  # positions of each vehicle
        self.ndet = []  # number of detections in the past

v = Vehicles()        
#old_boxes = []  # positions of the boxes in the previous frame
#old_boxes1 = [] 
#box_number = 0
trained = 1

def find_cars(image):
    ### Parameters
    
    color_space = 'RGB' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    x_start_stop = [200, None] # Min an
    y_start_stop = [400, 720] # Min and max in y to search in slide_window()
    heat_threshold = 1 # defines how many detections needed in heat map
   

    
    
    
    global trained 
    
    if trained == 1: 
       # load classifier and parameters
       dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
       svc = dist_pickle["svc"]
       X_scaler = dist_pickle["scaler"]
       orient = dist_pickle["orient"]
       pix_per_cell = dist_pickle["pix_per_cell"]
       cell_per_block = dist_pickle["cell_per_block"]
       spatial_size = dist_pickle["spatial_size"]
       hist_bins = dist_pickle["hist_bins"]
    else:
        # train model
        # Divide up into cars and notcars
            
        # Read Cars Images
        images = glob.glob('../vehicles/*/*.png')
        cars = []
        for image1 in images:
            cars.append(image1)
        
        #images = glob.glob('../vehicles/GTI_Left/*.png')
        #for image in images:
        #   cars.append(image)
        
            
        # Read Not Cars Images
        images = glob.glob('../non-vehicles/*/*.png')
        notcars = []
        for image1 in images:
            notcars.append(image1)
        
        
        
        car_features = extract_features(cars, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = extract_features(notcars, color_space=color_space, 
                                spatial_size=spatial_size, hist_bins=hist_bins, 
                                orient=orient, pix_per_cell=pix_per_cell, 
                                cell_per_block=cell_per_block, 
                                hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                hist_feat=hist_feat, hog_feat=hog_feat)
        
        X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = X_scaler.transform(X)
        
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
        
        
        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        
        print('Using:',orient,'orientations',pix_per_cell,
            'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))
        
        # Use a linear SVC 
        svc = LinearSVC()
        
        # Check the training time for the SVC
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        # Check the prediction time for a single sample
        t=time.time()
        trained = 1
         
    draw_image = np.copy(image)
    
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    image = image.astype(np.float32)/255
   
    windows = slide_window(image, x_start_stop, y_start_stop, 
                        xy_window=(88, 88), xy_overlap=(0.5, 0.5))
    
    hot_windows88 = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                             spatial_size=spatial_size, hist_bins=hist_bins, 
                             orient=orient, pix_per_cell=pix_per_cell, 
                             cell_per_block=cell_per_block, 
                             hog_channel=hog_channel, spatial_feat=spatial_feat, 
                             hist_feat=hist_feat, hog_feat=hog_feat)   
    
    windows = slide_window(image, x_start_stop, y_start_stop, 
                        xy_window=(96, 96), xy_overlap=(0.5, 0.5))
    
    hot_windows96 = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)      
    
    windows = slide_window(image, x_start_stop, y_start_stop=y_start_stop, 
                        xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    
    hot_windows128 = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                    
    hot_windows = hot_windows88 + hot_windows96 + hot_windows128
    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 1), thick=6)                    
    #fig = plt.figure()
    #plt.imshow(window_img)
    
    
    
    
    # Read in image similar to one shown above 
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,hot_windows)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,heat_threshold)
    
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    
    # Find final boxes from heatmap using label function
    from scipy.ndimage.measurements import label
    
    labels = label(heatmap)
    #draw_img = draw_labeled_bboxes(np.copy(image), labels)
    
    #fig = plt.figure()
    #plt.subplot(121)
    #plt.imshow(draw_img)
    #plt.title('Car Positions')
    #plt.subplot(122)
    #plt.imshow(heatmap, cmap='hot')
    # plt.title('Heat Map')
    #fig.tight_layout()
    
    #print(labels[1], 'cars found')
    #fig = plt.figure()
    #plt.imshow(labels[0], cmap='gray')
    
    
    # Read in the last image above
    #image = mpimg.imread('../test_images/test6.jpg')
    # Draw bounding boxes on a copy of the image
  
    draw_img = draw_labeled_bboxes(draw_image, labels)
   
    # Display the image
    #fig = plt.figure()
    #plt.imshow(draw_img1)
    
    if trained == 0:
        # save
        
        dist_pickle = {}
        dist_pickle["svc"] = svc
        dist_pickle["scaler"] = X_scaler
        dist_pickle["orient"] = orient
        dist_pickle["pix_per_cell"] = pix_per_cell
        dist_pickle["cell_per_block"] = cell_per_block
        dist_pickle["spatial_size"] = spatial_size
        dist_pickle["hist_bins"] = hist_bins
        pickle.dump( dist_pickle, open( "svc_pickle.p", "wb" ) )
        
    return draw_img
#image = mpimg.imread('../test_images/test6.png')
#find_cars(image)

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
#from IPython.display import HTML

white_output = '../project_video_output.mp4'

##clip1 = VideoFileClip("project_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("../project_video.mp4")
white_clip = clip1.fl_image(find_cars) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)