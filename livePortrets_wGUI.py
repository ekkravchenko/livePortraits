#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import necessary packages and functions
import cv2, dlib, os
import mls as mls
import numpy as np
import math
from utils import getFaceRect, landmarks2numpy, createSubdiv2D, calculateDelaunayTriangles, insertBoundaryPoints
from utils import getVideoParameters, warpTriangle, getRigidAlignment
from utils import teethMaskCreate, erodeLipMask, getLips, getLipHeight, drawDelaunay
from utils import mainWarpField, copyMouth
from utils import hallucinateControlPoints, getInterEyeDistance

""" Select the trackbar parameters"""
# images
imageArray = ['',        # indicates the first frame of the video
              "Frida1.jpg", "Frida2.jpg",
              "MonaLisa.jpg", "Rosalba_Carriera.jpg",
              "IanGillan.jpg", "AfghanGirl.jpg"]
imageIndex = 0 
maximageIndex = len(imageArray) -1
imageTrackbarName = "Images: \n 0: First frame \n 1: Frida Kahlo #1 \n 2: Frida Kahlo #2 \n 3: Mona Lisa \n 4: Rosalba Carriera \n 5: Ian Gillan \n 6: Afghan Girl"

# video
videoArray = ["anger.avi", "smile.avi", "teeth_smile.avi", "surprise.avi"]
videoIndex = 0
maxvideoIndex = len(videoArray) -1
videoTrackbarName = "Videos: \n 0: Anger \n 1: Smile \n 2: Smile with teeth \n 3: Surprise"

# state of the process
OnOFF = 0
maxOnOFF = 1
processTrackbarName = "Process: \n 0 - Still image \n 1 - Animate"

# create a folder for results, if it doesn't exist yet
os.makedirs("video_generated", exist_ok=True) 

""" Get face and landmark detectors"""
faceDetector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "../common/shape_predictor_68_face_landmarks.dat"  # Landmark model location
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)  

""" Function for control window """
# Bug: multi-line titles of the OpenCV trackbars don't work. So instead the correspondence 
# between indices and images is listed on an additional image in control window
controlWindowName = 'Control'
def createControlWindow(controlWindowName, imageTrackbarName, videoTrackbarName, processTrackbarName):
    # create a window
    cv2.namedWindow(controlWindowName)
    # Trackbar to select the image
    cv2.createTrackbar( "Images", controlWindowName, imageIndex, maximageIndex,  updateImageFN)
    # Trackbar to select the video
    cv2.createTrackbar( "Videos", controlWindowName, videoIndex, maxvideoIndex,  updateVideoFN)
    # Trackbar to start/stop the processing
    cv2.createTrackbar("Process", controlWindowName, OnOFF, maxOnOFF, runProcess)
    
    # Write the text with parameters
    position = (30, 30)
    font_scale = 0.75
    color = (255, 255, 255)
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA
    text_size, _ = cv2.getTextSize(imageTrackbarName, font, font_scale, thickness)
    line_height = text_size[1] + 5

    numLines = len(imageTrackbarName.split("\n")) +                len(videoTrackbarName.split("\n")) +                len(processTrackbarName.split("\n")) + 2
    aux_height = 30 + line_height*numLines
    aux_width = 350
    aux_im = np.zeros((30 + 18*line_height, 350,3), np.uint8)

    # write the text with image names
    x, y0 = position # initial postion
    for i, line in enumerate(imageTrackbarName.split("\n")):
        y = y0 + i * line_height
        cv2.putText(aux_im, line, (x,y), font, 
                    font_scale, color, thickness, cv2.LINE_AA) 
    # write the text with video names
    y0 = y + 2*line_height # initial postion: x is the same
    for i, line in enumerate(videoTrackbarName.split("\n")):
        y = y0 + i * line_height
        cv2.putText(aux_im, line, (x,y), font, 
                    font_scale, color, thickness, cv2.LINE_AA) 
    # write the text of process
    y0 = y + 2*line_height # initial postion: x is the same
    for i, line in enumerate(processTrackbarName.split("\n")):
        y = y0 + i * line_height
        cv2.putText(aux_im, line, (x,y), font, 
                    font_scale, color, thickness, cv2.LINE_AA) 
    cv2.imshow(controlWindowName, aux_im)    

""" Trackbar callback functions """     
# Function to select the image
def updateVideoFN( *args ):
    global video_fn, OnOFF
    videoIndex = args[0]
    video_fn = videoArray[videoIndex]
    
    # drop OnOFF to zero and update the "Process trackbar"
    OnOFF = 0
    cv2.createTrackbar("Process", controlWindowName, OnOFF, maxOnOFF, runProcess)
    
    LivePortrets()
    pass

def updateImageFN( *args ):
    # update image filename and drop OnOFF to zero
    global im_fn, OnOFF
    imageIndex = args[0]
    im_fn = imageArray[imageIndex]
        
    # drop OnOFF to zero and update the "Process trackbar"
    OnOFF = 0
    cv2.createTrackbar("Process", controlWindowName, OnOFF, maxOnOFF, runProcess)
    
    LivePortrets()
    pass

# Function to run the process (OnOFF=1) or stop it (OnOFF=0)
def runProcess( *args ):
    global OnOFF
    OnOFF = args[0]
    LivePortrets()
    pass    

""" A "gatekeeper" function """
def LivePortrets():
    # Function that checks the image and video and chooses between  showing the original video, image 
    # or running the main algorithm
    global OnOFF # needed to drop "Process" trackbar to 0-state in the end
    
    # Create a VideoCapture object
    cap = cv2.VideoCapture(os.path.join("video_recorded", video_fn))
    # Check if camera is opened successfully
    if (cap.isOpened() == False): 
        print("Unable to read camera feed")
        
    ## if user selected a first frame
    if im_fn == '': 
        # read the first frame
        ret, frame = cap.read()
        if ret == True:
            # the orginal video doesn't have information about the frame orientation, so  we need to rotate it
            frame = np.rot90(frame, 3).copy()            
            scaleY = 600./frame.shape[0]   # scale the frame to have a 600 pixel height
            frame = cv2.resize(src=frame, dsize=None, fx=scaleY, fy=scaleY, interpolation=cv2.INTER_LINEAR ) 
        
        # create a window and display  frame
        cv2.namedWindow('Live Portrets', cv2.WINDOW_AUTOSIZE)
        cv2.imshow("Live Portrets", frame)
        
        # show the whole video frame-by-frame if Process is ON
        while(OnOFF == 1):
            ret, frame = cap.read()
            if ret == True:
                # the orginal video doesn't have information about the frame orientation, so  we need to rotate it
                frame = np.rot90(frame, 3).copy()
                scaleY = 600./frame.shape[0]   # scale the frame to have a 600 pixel height
                frame = cv2.resize(src=frame, dsize=None, fx=scaleY, fy=scaleY, interpolation=cv2.INTER_LINEAR ) 
                cv2.imshow("Live Portrets", frame)
                if cv2.waitKey(1) & 0xFF == 1:#10:
                    continue
            else:
                # before breaking the loop drop OnOFF to zero and update the "Process trackbar"
                OnOFF = 0
                cv2.createTrackbar("Process", controlWindowName, OnOFF, maxOnOFF, runProcess)
                break                
    ## if user selected one of the images           
    else:
        # read and process the image
        im = cv2.imread(os.path.join("images_scaled", im_fn))
        
        if im is None:
            print("Unable to read the photo")
        else:
            # scale the image to have a 600 pixel height
            scaleY = 600./im.shape[0]
            im = cv2.resize(src=im, dsize=None, fx=scaleY, fy=scaleY, interpolation=cv2.INTER_LINEAR )  
    
        # create a window
        cv2.namedWindow('Live Portrets', cv2.WINDOW_AUTOSIZE)    
        
        # show image or run the algorithm
        if (OnOFF == 0):
            cv2.imshow("Live Portrets", im)
        elif (OnOFF == 1):
            runLivePortrets(im_fn, im, video_fn, cap)

""" Main algorithm """
def runLivePortrets(im_fn, im, video_fn, cap):
    global OnOFF # needed to drop "Process" trackbar to 0-state in the end

    ########## Get the parameters and landmarks of the image #########
    im_height, im_width, im_channels = im.shape

    # detect the face and the landmarks
    newRect = getFaceRect(im, faceDetector)
    landmarks_im = landmarks2numpy(landmarkDetector(im, newRect))

    ###########  Get the parameters of the driving video ##########
    # Obtain default resolutions of the frame (system dependent) and convert from float to integer.
    (time_video, length_video, fps, frame_width, frame_height) = getVideoParameters(cap)

    ############### Create new video ######################
    output_fn = im_fn[:-4] + "_" + video_fn
    out = cv2.VideoWriter(os.path.join("video_generated", output_fn),
                          cv2.VideoWriter_fourcc('M','J','P','G'), fps, (im_width, im_height))

    ############### Initialize the algorithm parameters #################
    frame = [] 
    tform = [] # similarity transformation that alignes video frame to the input image
    srcPoints_frame = []
    numCP = 68 # number of control points
    newRect_frame = []

    # Optical Flow
    points=[]
    pointsPrev=[] 
    pointsDetectedCur=[] 
    pointsDetectedPrev=[]
    eyeDistanceNotCalculated = True
    eyeDistance = 0
    isFirstFrame = True

    ############### Go over frames #################
    while(True):
        ret, frame = cap.read()

        if ret == False:
            # before breaking the loop drop OnOFF to zero and update the "Process trackbar"
            OnOFF = 0
            cv2.createTrackbar("Process", controlWindowName, OnOFF, maxOnOFF, runProcess)
            break  
        else:
            # the orginal video doesn't have information about the frame orientation, so  we need to rotate it
            frame = np.rot90(frame, 3).copy()

            # initialize a new frame for the input image
            im_new = im.copy()          

            ###############    Similarity alignment of the frame #################
            # detect the face (only for the first frame) and landmarks
            if isFirstFrame: 
                newRect_frame = getFaceRect(frame, faceDetector)
                landmarks_frame_init = landmarks2numpy(landmarkDetector(frame, newRect_frame))

                # compute the similarity transformation in the first frame
                tform = getRigidAlignment(landmarks_frame_init, landmarks_im)    
            else:
                landmarks_frame_init = landmarks2numpy(landmarkDetector(frame, newRect_frame))
                if tform == []:
                    print("ERROR: NO SIMILARITY TRANSFORMATION")

            # Apply similarity transform to the frame
            frame_aligned = np.zeros((im_height, im_width, im_channels), dtype=im.dtype)
            frame_aligned = cv2.warpAffine(frame, tform, (im_width, im_height))

            # Change the landmarks locations
            landmarks_frame = np.reshape(landmarks_frame_init, (landmarks_frame_init.shape[0], 1, landmarks_frame_init.shape[1]))
            landmarks_frame = cv2.transform(landmarks_frame, tform)
            landmarks_frame = np.reshape(landmarks_frame, (landmarks_frame_init.shape[0], landmarks_frame_init.shape[1]))

            # hallucinate additional control points
            if isFirstFrame: 
                (subdiv_temp, dt_im, landmarks_frame) = hallucinateControlPoints(landmarks_init = landmarks_frame, 
                                                                                im_shape = frame_aligned.shape, 
                                                                                INPUT_DIR="", 
                                                                                performTriangulation = True)
                # number of control points
                numCP = landmarks_frame.shape[0]
            else:
                landmarks_frame = np.concatenate((landmarks_frame, np.zeros((numCP-68,2))), axis=0)

            ############### Optical Flow and Stabilization #######################
            # Convert to grayscale.
            imGray = cv2.cvtColor(frame_aligned, cv2.COLOR_BGR2GRAY)

            # prepare data for an optical flow
            if (isFirstFrame==True):
                [pointsPrev.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
                [pointsDetectedPrev.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
                imGrayPrev = imGray.copy()

            # pointsDetectedCur stores results returned by the facial landmark detector
            # points stores the stabilized landmark points
            points = []
            pointsDetectedCur = []
            [points.append((p[0], p[1])) for p in landmarks_frame[68:,:]]
            [pointsDetectedCur.append((p[0], p[1])) for p in landmarks_frame[68:,:]]

            # Convert to numpy float array
            pointsArr = np.array(points, np.float32)
            pointsPrevArr = np.array(pointsPrev,np.float32)

            # If eye distance is not calculated before
            if eyeDistanceNotCalculated:
                eyeDistance = getInterEyeDistance(landmarks_frame)
                eyeDistanceNotCalculated = False

            dotRadius = 3 if (eyeDistance > 100) else 2
            sigma = eyeDistance * eyeDistance / 400
            s = 2*int(eyeDistance/4)+1

            #  Set up optical flow params
            lk_params = dict(winSize  = (s, s), maxLevel = 5, criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))
            pointsArr, status, err = cv2.calcOpticalFlowPyrLK(imGrayPrev,imGray,pointsPrevArr,pointsArr,**lk_params)
            sigma = 100

            # Converting to float and back to list
            points = np.array(pointsArr,np.float32).tolist()   

            # Facial landmark points are the detected landmark and additional control points are tracked landmarks  
            landmarks_frame[68:,:] = pointsArr
            landmarks_frame = landmarks_frame.astype(np.int32)

            # getting ready for the next frame
            imGrayPrev = imGray        
            pointsPrev = points
            pointsDetectedPrev = pointsDetectedCur

            ############### End of Optical Flow and Stabilization #######################

            # save information of the first frame for the future
            if isFirstFrame: 
                # hallucinate additional control points for a still image
                landmarks_list = landmarks_im.copy().tolist()
                for p in landmarks_frame[68:]:
                    landmarks_list.append([p[0], p[1]])
                srcPoints = np.array(landmarks_list)
                srcPoints = insertBoundaryPoints(im_width, im_height, srcPoints) 

                lip_height = getLipHeight(landmarks_im)            
                (_, _, maskInnerLips0, _) = teethMaskCreate(im_height, im_width, srcPoints)    
                mouth_area0=maskInnerLips0.sum()/255  

                # get source location on the first frame
                srcPoints_frame = landmarks_frame.copy()
                srcPoints_frame = insertBoundaryPoints(im_width, im_height, srcPoints_frame)  

                # Write the original image into the output file
                out.write(im_new)                  

                # Display the original image           
                cv2.imshow('Live Portrets', im_new)
                # stop for a short while
                if cv2.waitKey(1) & 0xFF == 1:#10:
                    continue
                # Go out of the loop if OnOFF trackbar was changed to 0
                if OnOFF==0:
                    break

                # no need in additional wraps for the first frame
                isFirstFrame = False
                continue

            ############### Warp field #######################               
            dstPoints_frame = landmarks_frame
            dstPoints_frame = insertBoundaryPoints(im_width, im_height, dstPoints_frame)

            # get the new locations of the control points
            dstPoints = dstPoints_frame - srcPoints_frame + srcPoints   

            # get a warp field, smoothen it and warp the image
            im_new = mainWarpField(im,srcPoints,dstPoints,dt_im)       

            ############### Mouth cloning #######################
            # get the lips and teeth mask
            (maskAllLips, hullOuterLipsIndex, maskInnerLips, hullInnerLipsIndex) = teethMaskCreate(im_height, im_width, dstPoints)
            mouth_area = maskInnerLips.sum()/255        

            # erode the outer mask based on lipHeight
            maskAllLipsEroded = erodeLipMask(maskAllLips, lip_height)
            
            # smooth the mask of inner region of the mouth
            maskInnerLips = cv2.GaussianBlur(np.stack((maskInnerLips,maskInnerLips,maskInnerLips), axis=2),(3,3), 10)

            # clone/blend the moth part from 'frame_aligned' if needed (for mouth_area/mouth_area0 > 1)
            im_new = copyMouth(mouth_area, mouth_area0,
                                landmarks_frame, dstPoints,
                                frame_aligned, im_new,
                                maskAllLipsEroded, hullOuterLipsIndex, maskInnerLips)           

            # Write the frame into the file 'output.avi'
            out.write(im_new)

            # Display the resulting frame    
            cv2.imshow('Live Portrets', im_new)
            
            # stop for a short while
            if cv2.waitKey(1) & 0xFF == 1:#10:
                continue
                
            # Go out of the loop if OnOFF trackbar was changed to 0
            if OnOFF==0:
                break               
    # When everything is done, release the video capture and video write objects
    cap.release()
    out.release()
          

""" Main part """
#define the driving video and the image of the face in neutral pose
im_fn = imageArray[imageIndex]
video_fn = videoArray[videoIndex]
LivePortrets()

# create control window with a trackbar
createControlWindow(controlWindowName, imageTrackbarName, videoTrackbarName, processTrackbarName)

cv2.waitKey(0)
cv2.destroyAllWindows() 


