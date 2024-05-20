import cv2
import numpy as np

from collections import defaultdict
from utilities import drawPoints, resizeImg, createMatrix, normalizePoints, drawLines, denormalizePoints
import scipy.spatial as spatial
import scipy.cluster as clstr


def preprocessImg(img):
    '''
        Convert to image to grayscale and blur the image to remove noise

        Parameters:
            img - image to preprocess
        
        Return value:
            blur - the preprocessed image
    '''
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0, 0) #remove noise before edge detection

    return blur

def detectEdges(img):
    '''
        Detect the edges in the image using Canny edge detection

        Parameters:
            img - the image to be processed
            
        Return value:
            array of edges
    '''
    img = img.astype(np.uint8)
    return cv2.Canny(img, 80, 200, 3)

def detectLines(edges):
    '''
        Detect lines from the given array of edges

        Parameters:
            edges - array of detected edges
        
        Return value:
            array of detected lines
    '''
    # lines = cv2.HoughLines(dilated, 1.5, np.pi / 180, 450)
    return cv2.HoughLines(edges, 1, np.pi / 180, 120, 0, 0)

def sortLines(lines):
    '''
        Separate the detected horizontal and vertical lines

        Parameters:
            lines - array of detected lines

        Return value:
            h - array of horizontal lines
            v - aray of vertical lines
    '''
    h = []
    v = []

    for i in range(lines.shape[0]):
        rho = lines[i][0][0]
        theta = lines[i][0][1]

        if theta < np.pi / 4 or theta > np.pi -np.pi/4:
            v.append([rho, theta])
        else:
            h.append([rho, theta])

    return h, v

def removeSimilarLines(lines, threshold):
    '''
        Remove similar lines from array lines to reduce the intersections in the image.

        Parameters:
            lines - array of [rho, theta] of the lines
            threshold - the number of rho (pixels) difference to consider a line similar 
    '''
    for i in lines:
        length = len(lines)
        if i == []:
            continue
        for j in range(length-1):
            if i == lines[j] or i==[] or lines[j] == []: 
                continue
            elif i[0] - threshold <= lines[j][0] <= i[0] + threshold:
                lines[j] = []
    
    return [item for item in lines if item != []]

def getLineIntersections(horizontal, vertical):
    '''
        Get the intersections of the horizontal and vertical lines

        Parameters:
            horizontal - array of horizontal lines
            vertical - array of vertical lines
        
        Return value:
            array of points (intersections)
    '''
    points = []
    for rho1, theta1 in horizontal:
        for rho2, theta2 in vertical:
            A = np.array([
                [np.cos(theta1), np.sin(theta1)],
                [np.cos(theta2), np.sin(theta2)]
            ])
            b = np.array([[rho1], [rho2]])
            point = np.linalg.solve(A, b)
            point = int(np.round(point[0])), int(np.round(point[1]))
            points.append(point)
    return np.array(points)

def clusterIntersections(points, max_dist=40):
    '''
        Group the given points into clusters
        Parameters:
            points - array of points to cluster
            max_dist - maximum distance threshold used to form clusters
        Return value:
            result - array of centroids of the clusters
    '''

    # pairwise distance of each point
    Y = spatial.distance.pdist(points)
    # single linkage clustering
    Z = clstr.hierarchy.single(Y)
    # form flat clusters based on the threshold
    T = clstr.hierarchy.fcluster(Z, max_dist, 'distance')
    clusters = defaultdict(list)
    # group points into clusters
    for i in range(len(T)):
        clusters[T[i]].append(points[i])
    clusters = clusters.values()
    # compute centroid of clusters 
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), clusters)

    result = []
    for point in clusters:
        result.append([point[0], point[1]])
    return result

def checkColors(img, mat, threshold=20):
    '''
        Checks if 3 consecutive colors of the detected board are the same.
        Parameters:
            img - image of the board
            mat - 2d array of the detected points
            threshold - maximum difference between colors to consider that colors are different
        Return value:
            flat - 1d array of points from the chess board.
    '''
    phase = 0
    while phase < 4:
        if phase == 0: # top
            box1 = getMeanColor(img, [mat[0][0], mat[0][1], mat[1][0], mat[1][1]])
            box2 = getMeanColor(img, [mat[0][1], mat[0][2], mat[1][1], mat[1][2]])
            box3 = getMeanColor(img, [mat[0][2], mat[0][3], mat[1][2], mat[1][3]])
        elif phase == 1: # bottom
            box1 = getMeanColor(img, [mat[-1][0], mat[-1][1], mat[-2][0], mat[-2][1]])
            box2 = getMeanColor(img, [mat[-1][1], mat[-1][2], mat[-2][1], mat[-2][2]])
            box3 = getMeanColor(img, [mat[-1][2], mat[-1][3], mat[-2][2], mat[-2][3]])
        elif phase == 2: # left
            box1 = getMeanColor(img, [mat[0][0], mat[0][1], mat[1][0], mat[1][1]])
            box2 = getMeanColor(img, [mat[1][0], mat[1][1], mat[2][0], mat[2][1]])
            box3 = getMeanColor(img, [mat[2][0], mat[2][1], mat[3][0], mat[3][1]])
        elif phase == 3: # right
            box1 = getMeanColor(img, [mat[-1][-1], mat[-1][-2], mat[-2][-1], mat[-2][-2]])
            box2 = getMeanColor(img, [mat[-2][-1], mat[-2][-2], mat[-3][-1], mat[-3][-2]])
            box3 = getMeanColor(img, [mat[-3][-1], mat[-3][-2], mat[-4][-1], mat[-4][-2]])

        diff1_2 = np.sqrt((box1[0]-box2[0])**2 + (box1[1]-box2[1])**2 + (box1[2]-box2[2])**2)
        diff2_3 = np.sqrt((box2[0]-box3[0])**2 + (box2[1]-box3[1])**2 + (box2[2]-box3[2])**2)
        
        if diff1_2 > threshold and diff2_3 > threshold:
            phase += 1
        else:
            if phase == 0: # remove top row
                mat = np.array(mat[1:])
            elif phase == 1: # remove bottom row
                mat = np.array(mat[:-1])
            elif phase == 2: # remove left column
                mat = np.array([subarray[1:] for subarray in mat])
            elif phase == 3: # remove right column
                mat = np.array([subarray[:-1] for subarray in mat])

    flat = mat[0]
    for i in mat[1:]:
        flat = np.vstack((flat, i))

    return flat

def getMeanColor(image, polygon_points):
    '''
        Get the mean color of a polygon in the image

        Parameters:
            image - image of the chessboard
            polygon_points - array of points to create a polygon

        Return value:
            mean_color - the BGR values of the polygon
    '''
    # Create a mask for the polygon
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(np.array(polygon_points))
    cv2.fillPoly(mask, [hull], 255)
    
    # Apply the mask to the image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Calculate the mean color
    mean_color = cv2.mean(masked_image, mask=mask)[:3]
    
    return mean_color

def detectBoard(img):
    '''
        Detect the points of the board

        Parameters:
            img - the image of the board
        
        Return value:
            clusters - 1d matrix of the points
            scale - how much the image used for getting the points were bigger than the original image
    '''
    frame, scale = resizeImg(img) 
    processed = preprocessImg(frame)
    _, thresh_img = cv2.threshold(processed, 120, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)

    edges = detectEdges(morph_img)
    lines = detectLines(edges)
    
    if lines is not None:
        h, v = sortLines(lines)

        h = removeSimilarLines(h, 30)
        v = removeSimilarLines(v, 30)

        points = getLineIntersections(h, v)
        matrix = createMatrix(points)

        if len(points) >= 81:
            points = checkColors(morph_img, matrix)
            drawPoints(frame, points, (255,0,0), 3)
            clusters = clusterIntersections(points)
            if len(clusters) == 81:
                return normalizePoints(np.array(clusters).astype('float32'), frame.shape)
            elif len(clusters) > 81:
                matrix = createMatrix(np.array(clusters).astype('int32'))
                clusters = checkColors(morph_img, matrix)
                return normalizePoints(np.array(clusters).astype('float32'), frame.shape)
            else:
                print("No chessboard detected.")
        else:
            print("Insufficient points.")
    else:
        print("No chessboard detected.")
    
    return None