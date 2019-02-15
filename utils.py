# import necessary packages
import sys, cv2, dlib, time, os
import numpy as np
from skimage import transform


def getFaceRect(im, faceDetector):
    faceRects = faceDetector(im, 0)
    if len(faceRects)>0:
        faceRect = faceRects[0]
        newRect = dlib.rectangle(int(faceRect.left()),int(faceRect.top()), 
                             int(faceRect.right()),int(faceRect.bottom()))
    else:
        newRect = dlib.rectangle(0,0,im.shape[1],im.shape[0])
    return newRect

def insertBoundaryPoints(width, height, np_array):
    ## Takes as input non-empty numpy array
    np_array = np.append(np_array,[[0, 0]],axis=0)
    np_array = np.append(np_array,[[0, height//2]],axis=0)
    np_array = np.append(np_array,[[0, height-1]],axis=0)    
    np_array = np.append(np_array,[[width//2, 0]],axis=0)
    np_array = np.append(np_array,[[width//2, height-1]],axis=0)        
    np_array = np.append(np_array,[[width-1, 0]],axis=0)
    np_array = np.append(np_array,[[width-1, height//2]],axis=0)   
    np_array = np.append(np_array,[[width-1, height-1]],axis=0)
    return np_array

def createSubdiv2D(size, landmarks):
    '''
        Input
    size[0] is height
    size[1] is width    
    landmarks as in dlib-detector output
        Output
    subdiv -- Delaunay Triangulation        
    '''   
    # Rectangle to be used with Subdiv2D
    rect = (0, 0, size[1], size[0])
    
    # Create an instance of Subdiv2D
    subdiv = cv2.Subdiv2D(rect)
    
    # Insert points into subdiv
    for p in landmarks :
        subdiv.insert((p[0], p[1])) 
      
    return subdiv


        
def drawDelaunay(img, dt, points, delaunayColor = (255,255,255)) :
    size = img.shape
    r = (0, 0, size[1], size[0])

    # Will convert triangle representation to three vertices pt1, pt2, pt3
    for t in dt :
        pt1 = (points[t[0],0], points[t[0],1])
        pt2 = (points[t[1],0], points[t[1],1])
        pt3 = (points[t[2],0], points[t[2],1])
        # Draw triangles that are completely inside the image
        if rectContains(r, pt1) and rectContains(r, pt2) and rectContains(r, pt3) :
            cv2.line(img, pt1, pt2, delaunayColor, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunayColor, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunayColor, 1, cv2.LINE_AA, 0)
            
            
def getVideoParameters(cap):
    #  Obtains default resolutions of the frames (system dependent) and convert the resolutions from float to integer.
    time = int(cap.get(cv2.CAP_PROP_POS_AVI_RATIO))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # = 3953
    fps    = int(cap.get(cv2.CAP_PROP_FPS))          # = 29    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return (time,length, fps, width, height) 


# Creates masks and hulls for outer and inner line of lips
def teethMaskCreate(height, width,landmarks_video):    
    ## Create a black image to be used as a mask for the lips
    maskAllLips = np.zeros((height, width), dtype = np.uint8)
    maskInnerLips = np.zeros((height, width), dtype = np.uint8)

    # Create a convex hull using the points of the left and right eye
    outerLipsIndex = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    innerLipsIndex = [60, 61, 62, 63, 64, 65, 66, 67]

    hullOuterLipsIndex = []
    for i in range(0,len(outerLipsIndex)):
        hullOuterLipsIndex.append((landmarks_video[outerLipsIndex[i]][0], landmarks_video[outerLipsIndex[i]][1]))
    cv2.fillConvexPoly(maskAllLips, np.int32(hullOuterLipsIndex), 255)

    hullInnerLipsIndex = []
    for i in range(0,len(innerLipsIndex)):
        hullInnerLipsIndex.append((landmarks_video[innerLipsIndex[i]][0], landmarks_video[innerLipsIndex[i]][1]))
    cv2.fillConvexPoly(maskInnerLips, np.int32(hullInnerLipsIndex), 255)
    
    return (maskAllLips, hullOuterLipsIndex, maskInnerLips, hullInnerLipsIndex)

def getLipHeight(landmarks):
    #upperLipIndex = [50, 51, 52]
    #lowerLipIndex = [58, 57,56]
    #return landmarks[lowerLipIndex[0]][1] - landmarks[upperLipIndex[0]][1]
    return landmarks[58][1] - landmarks[50][1]

def erodeLipMask(maskAllLips, lipHeight, coeff = 0.1):
    # erode the outer mask
    erosionSize = int(coeff*lipHeight)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*erosionSize+1, 2*erosionSize+1),(erosionSize, erosionSize))
    maskAllLipsEroded = cv2.erode(maskAllLips, element)    
    return maskAllLipsEroded


# Get the points on the lips on the outer (points_outLips) and iner side (points_inLips) and the triangulation (dt_lips)
def getLips(points):
    # Create a convex hull using the points of the inner and outer lips
    outerLipsIndex = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    innerLipsIndex = [60, 61, 62, 63, 64, 65, 66, 67]  
    inFlag = False

    points_inLips = points[innerLipsIndex,:]
    points_outLips = points[outerLipsIndex,:]

    # Rectangle to be used with Subdiv2D
    if inFlag:
        rect_new = (np.min(points_inLips[:,0]), np.min(points_inLips[:,1]), 
                    np.max(points_inLips[:,0]), np.max(points_inLips[:,1]))
    else:
        rect_new = (np.min(points_outLips[:,0]), np.min(points_outLips[:,1]), 
                    np.max(points_outLips[:,0]), np.max(points_outLips[:,1]))

    # Create an instance of Subdiv2D
    subdiv_new = cv2.Subdiv2D(rect_new)
    
    if inFlag: 
        # Insert points into subdiv
        for i in range(0, len(innerLipsIndex)) :
            subdiv_new.insert((points_inLips[i,0], points_inLips[i,1]))     

        # get the initial indices of the points in lips triangulation
        dt_lips = np.array(calculateDelaunayTriangles(subdiv_new, points_inLips))
        for ii in range(0, dt_lips.shape[0]):
            for jj in range(0, dt_lips.shape[1]):
                dt_lips[ii,jj] = innerLipsIndex[dt_lips[ii,jj]]
    else:
        # Insert points into subdiv
        for i in range(0, len(outerLipsIndex)) :
            subdiv_new.insert((points_outLips[i,0], points_outLips[i,1]))     
        # get the initial indices of the points in lips triangulation
        dt_lips = np.array(calculateDelaunayTriangles(subdiv_new, points_outLips))
        for ii in range(0, dt_lips.shape[0]):
            for jj in range(0, dt_lips.shape[1]):
                dt_lips[ii,jj] = outerLipsIndex[dt_lips[ii,jj]]        
        
    return (points_inLips, points_outLips, dt_lips)
        

# Turn the dlib-landmarks into numpy array    
def landmarks2numpy(landmarks_init):
    landmarks = landmarks_init.parts()
    points = []
    for ii in range(0, len(landmarks)):
        points.append([landmarks[ii].x, landmarks[ii].y])
    return np.array(points)


# Estimate the similarity transformation of landmarks_frame to landmarks_im
# Uses 3 points: a tip of the nose and corners of the eyes
def getRigidAlignment(landmarks_frame, landmarks_im):
    # video frame
    video_lmk = [[np.int(landmarks_frame[30][0]), np.int(landmarks_frame[30][1])], 
                 [np.int(landmarks_frame[36][0]), np.int(landmarks_frame[36][1])],
                 [np.int(landmarks_frame[45][0]), np.int(landmarks_frame[45][1])] ]

    # Corners of the eye in normalized image
    img_lmk = [ [np.int(landmarks_im[30][0]), np.int(landmarks_im[30][1])], 
                [np.int(landmarks_im[36][0]), np.int(landmarks_im[36][1])],
                [np.int(landmarks_im[45][0]), np.int(landmarks_im[45][1])] ]

    # Calculate similarity transform
    #tform = cv2.estimateRigidTransform(np.array([video_lmk]), np.array([img_lmk]), False)
    tform = transform.estimate_transform('similarity', np.array(video_lmk), np.array(img_lmk)).params[:2,:]
    return tform            


# Creates a mask of shape (im.shape[0],im.shape[1],3) that for each pixels contains 
# the index of the triangle (in Delanay Triangulation) it belongs to.
# Also creates a 2x3xlen(dt_im) that contains the affine matrix for each triangle
def getTriMaskAndMatrix(im,srcPoints,dstPoints,dt_im):
    maskIdc = np.zeros((im.shape[0],im.shape[1],3), np.int32)
    matrixA = np.zeros((2,3, len(dt_im)), np.float32)
    for i in range(0, len(dt_im)):
        t_src = []
        t_dst = []
        for j in range(0, 3):
            t_src.append((srcPoints[dt_im[i][j]][0], srcPoints[dt_im[i][j]][1]))
            t_dst.append((dstPoints[dt_im[i][j]][0], dstPoints[dt_im[i][j]][1]))        
        # get an inverse transformatio: from t_dst to t_src
        Ai_temp = cv2.getAffineTransform(np.array(t_dst, np.float32), np.array(t_src, np.float32))
        matrixA[:,:,i] = Ai_temp - [[1,0,0],[0,1,0]]
        # fill in a mask with triangle number
        cv2.fillConvexPoly(img=maskIdc, points=np.int32(t_dst), color=(i,i,i), lineType=8, shift=0) 
    return (maskIdc,matrixA)


# Get an initial warp field of shape (im.shape[0],im.shape[1],2) which for each pixel keeps it's x and y offsets
def getWarpInit(matrixA, maskIdc,numPixels,xxyy):
    maskIdc_resh = maskIdc[:,:,0].reshape(numPixels)
    warpField = np.zeros((maskIdc.shape[0],maskIdc.shape[1],2), np.float32)
    warpField = warpField.reshape((numPixels,2))
    
    # wrap triangle by triangle
    for i in range(0, matrixA.shape[2]):
        xxyy_masked = []
        xxyy_masked = xxyy[maskIdc_resh==i,:]
        # don't process empty array
        if xxyy_masked.size == 0:
            continue
        warpField_temp = cv2.transform(xxyy_masked, np.squeeze(matrixA[:,:,i]))
        warpField[maskIdc_resh==i,:] =  np.reshape(warpField_temp, (warpField_temp.shape[0], 2))  

        # reshape to original image shape 
    warpField = warpField.reshape((maskIdc.shape[0],maskIdc.shape[1],2))
    return warpField


# Creates an warped image from original image (im), its control points (srcPoints),
# destination location of the control points Points) and Delanuy Triangulation (dt_im)
def mainWarpField(im,srcPoints,dstPoints,dt_im):
    yy,xx = np.mgrid[0:im.shape[0], 0:im.shape[1]]
    numPixels = im.shape[0] * im.shape[1]
    xxyy = np.reshape(np.stack((xx.reshape(numPixels),yy.reshape(numPixels)), axis = 1), (numPixels, 1, 2))           

    # get a mask with triangle indices
    (maskIdc, matrixA) = getTriMaskAndMatrix(im,srcPoints,dstPoints,dt_im)
      
    # compute the initial warp field (offsets) and smooth it
    warpField = getWarpInit(matrixA, maskIdc,numPixels,xxyy)  #size: im.shape[0]*im.shape[1]*2
    warpField_blured = smoothWarpField(dstPoints, warpField)
    
    # get the corresponding indices instead of offsets and make sure theu are in the image range
    warpField_idc = warpField_blured + np.stack((xx,yy), axis = 2)
    warpField_idc[:,:,0] = np.clip(warpField_idc[:,:,0],0,im.shape[1]-1)  #x
    warpField_idc[:,:,1] = np.clip(warpField_idc[:,:,1],0,im.shape[0]-1)  #y
    
    # fill in the image with corresponding indices
    im_new2 = im.copy()
    im_new2[yy,xx,:] = im[np.intc(warpField_idc[:,:,1]), np.intc(warpField_idc[:,:,0]),:]

    return im_new2


# Smoothes the warp field (offsets) depending oon the distance from a face
def smoothWarpField(dstPoints, warpField):
    warpField_blured = warpField.copy()

    # calculate facial mask
    faceHullIndex = cv2.convexHull(np.array(dstPoints[:68,:]), returnPoints=False)
    faceHull = []
    for i in range(0, len(faceHullIndex)):
        faceHull.append((dstPoints[faceHullIndex[i],0], dstPoints[faceHullIndex[i],1]))
    maskFace = np.zeros((warpField.shape[0],warpField.shape[1],3), dtype=np.uint8)  
    cv2.fillConvexPoly(maskFace, np.int32(faceHull), (255, 255, 255))     

    # get distance transform
    dist_transform = cv2.distanceTransform(~maskFace[:,:,0], cv2.DIST_L2,5)
    max_dist = dist_transform.max()

    # initialize a matrix with distance ranges ang sigmas for smoothing
    maxRadius = 0.05*np.linalg.norm([warpField.shape[0], warpField.shape[1]]) # 40 pixels for 640x480 image
    thrMatrix = [[0, 0.1*max_dist, 0.1*maxRadius],
                 [0.1*max_dist, 0.2*max_dist, 0.2*maxRadius],
                 [0.2*max_dist, 0.3*max_dist, 0.3*maxRadius],
                 [0.3*max_dist, 0.4*max_dist, 0.4*maxRadius],
                 [0.4*max_dist, 0.5*max_dist, 0.5*maxRadius],
                 [0.5*max_dist, 0.6*max_dist, 0.6*maxRadius],
                 [0.6*max_dist, 0.7*max_dist, 0.7*maxRadius],
                 [0.7*max_dist, 0.8*max_dist, 0.8*maxRadius],
                 [0.8*max_dist, 0.9*max_dist, 0.9*maxRadius],
                 [0.9*max_dist, max_dist + 1, maxRadius]]
    for entry in thrMatrix:
        # select values in the range (entry[0], entry[1]]
        mask_range = np.all(np.stack((dist_transform>entry[0], dist_transform<=entry[1]), axis=2), axis=2)
        mask_range = np.stack((mask_range,mask_range), axis=2)

        warpField_temp = cv2.GaussianBlur(warpField, (0, 0), entry[2])        
        warpField_blured[mask_range] = warpField_blured[mask_range]       
    return warpField_blured


# Seamlessly clones and alpha-blends the mouth if it's significantly open
def copyMouth(mouth_area, mouth_area0, landmarks_frame, dstPoints,frame_aligned, im_new2,maskAllLipsEroded, hullOuterLipsIndex, maskInnerLips):
    # initialize the image that will contain the wrapped mouth
    im_new = im_new2.copy()
    
    # get new lips triangulation
    (_, _, dt_lips) = getLips(dstPoints)
    
    if (mouth_area > mouth_area0):
        # wrap the mouth from an aligned frame (frame_aligned) to an image (im_new)
        for i in range(0, len(dt_lips)):
            t_frame_lips = []
            t_im_lips = []
            for j in range(0, 3):
                t_frame_lips.append((landmarks_frame[dt_lips[i][j]][0], 
                                     landmarks_frame[dt_lips[i][j]][1]))
                t_im_lips.append((dstPoints[dt_lips[i][j]][0], 
                                  dstPoints[dt_lips[i][j]][1]))        
            warpTriangle(frame_aligned, im_new, t_frame_lips, t_im_lips)   

        ###       Seamlessly clone the teeth to an empty image        
        # find center of the mask to be cloned with the destination image
        r = cv2.boundingRect(np.float32([hullOuterLipsIndex]))    
        center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))

        # Clone seamlessly from an image with a wraped mouth (im_new) on a wraped image  (im_new2)
        im_seamClone = cv2.seamlessClone(np.uint8(im_new), np.uint8(im_new2.copy()),  np.uint8(maskAllLipsEroded), 
                                     center, cv2.NORMAL_CLONE)
        # initialize masks for alpha-blending
        if (mouth_area > 2*mouth_area0):
            mask_seamClone = maskInnerLips
            mask2 = (255,255,255) - maskInnerLips
        else:
            mask_seamClone = maskInnerLips/2
            mask2 = (255,255,255) - maskInnerLips                
        
        # Perform alpha-blending: I=αF+(1−α)B                
        temp1 = np.multiply(im_seamClone,(mask_seamClone*(1.0/255)))
        temp2 = np.multiply(im_new2,(mask2*(1.0/255)))
        im_new2 = (temp1 + temp2).astype(np.uint8)
    return im_new2  


# Hallucinate additional (out-of-face) control points by aligning the template to the obtined landmarks
# Returns subdivition (subdiv_temp) and Delanauy triangulation (dt_temp) on an aligned template 
# and original image landmarks (landmarks_out) with additional control points
def hallucinateControlPoints(landmarks_init, im_shape, INPUT_DIR="", performTriangulation = False):
    # load the image, clone it, and setup the mouse callback function
    template_im = cv2.imread(os.path.join(INPUT_DIR,"templates","brene-brown.jpg"))

    # load template control points
    templateCP_fn = os.path.join(INPUT_DIR, 'features','brene_controlPoints.txt')
    templateCP_init = np.int32(np.loadtxt(templateCP_fn, delimiter=' '))    
    
    # align the template to the frame
    tform = getRigidAlignment(templateCP_init, landmarks_init)
    
    # Apply similarity transform to the frame
    templateCP = np.reshape(templateCP_init, (templateCP_init.shape[0], 1, templateCP_init.shape[1]))
    templateCP = cv2.transform(templateCP, tform)
    templateCP = np.reshape(templateCP, (templateCP_init.shape[0], templateCP_init.shape[1]))
    
    # omit the points outside the image range
    templateCP_Constrained = omitOOR(templateCP, im_shape)

    # hallucinate additional keypoint on a new image 
    landmarks_list = landmarks_init.tolist()
    for p in templateCP_Constrained[68:]:
        landmarks_list.append([p[0], p[1]])
    landmarks_out = np.array(landmarks_list)
        
    subdiv_temp = None
    dt_temp = None
    if performTriangulation:
        srcTemplatePoints = templateCP_Constrained.copy()
        srcTemplatePoints = insertBoundaryPoints(im_shape[1], im_shape[0], srcTemplatePoints)
        subdiv_temp = createSubdiv2D(im_shape, srcTemplatePoints)  
        dt_temp = calculateDelaunayTriangles(subdiv_temp, srcTemplatePoints) 
    
    return (subdiv_temp, dt_temp, landmarks_out)


# Omit the out-of-face control points in the template which fall out of image range after alignment
def omitOOR(landmarks_template, shape):
    outX = np.logical_or((landmarks_template[:,0] < 0),  (landmarks_template[:,0] >= shape[1]))
    outY = np.logical_or((landmarks_template[:,1] < 0), (landmarks_template[:,1] >= shape[0]))
    outXY = np.logical_or(outX,outY)
    landmarks_templateConstrained = landmarks_template.copy()
    landmarks_templateConstrained = landmarks_templateConstrained[~outXY]
    return landmarks_templateConstrained


def getInterEyeDistance(landmarks):
    leftEyeLeftCorner = (landmarks[36,0], landmarks[36,1])
    rightEyeRightCorner = (landmarks[45,0], landmarks[45,1])
    distance = int(cv2.norm(np.array(rightEyeRightCorner) - np.array(leftEyeLeftCorner)))
    return distance

##========================================================================================================================================
##                                        TAKEN FROM THE COURSE MATERIALS
##=======================================================================================================================================
# Check if a point is inside a rectangle (Mallik)
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True


# Calculate Delaunay triangles for set of points.
# Returns the vector of indices of 3 points for each triangle
def calculateDelaunayTriangles(subdiv, points):
    # Get Delaunay triangulation
    triangleList = subdiv.getTriangleList()

    # Find the indices of triangles in the points array
    delaunayTri = []

    for t in triangleList:
        # The triangle returned by getTriangleList is
        # a list of 6 coordinates of the 3 points in
        # x1, y1, x2, y2, x3, y3 format.
        # Store triangle as a list of three points
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        #if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
        # Variable to store a triangle as indices from list of points
        ind = []
        # Find the index of each vertex in the points list
        for j in range(0, 3):
            for k in range(0, len(points)):
                if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                    ind.append(k)
                    # Store triangulation as a list of indices
        if len(ind) == 3:
            delaunayTri.append((ind[0], ind[1], ind[2]))
    return delaunayTri


# Apply affine transform calculated using srcTri and dstTri to src and output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None,
             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ((1.0, 1.0, 1.0) - mask)
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
    