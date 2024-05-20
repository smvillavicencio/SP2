import cv2
import numpy as np

def resizeImg(img, width=2048):
    '''
        Resize an image while maintaining its proportionality

        Parameters:
            img - image to be resized
            width - width of the resized image (default = 2048)
    
        Return values:
            img - resized image
            scale - the scale of the image compared to the original image
    '''
    h, w, _ = img.shape
    
    if w == width:
        return img, 1
    
    scale = width/w
    img = cv2.resize(img, (width, int(h*scale)))

    return img, scale

def drawPoints(img, points, color, size):
    '''
        Draw points on the image

        Parameters:
            img - image where points will be drawn
            points - array of points
            color - color of points to be drawn
            size - size of points to be drawn
    '''
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 2, color, size)

def drawLines(img, lines, color):
    '''
        Draw lines on the image

        Parameters:
            img - image where lines will be drawn
            lines - array of lines
            color - color of lines to be drawn
    '''
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 2000*(-b)), int(y0 + 2000*(a)))
        pt2 = (int(x0 - 2000*(-b)), int(y0 - 2000*(a)))
        cv2.line(img, pt1, pt2, color, 3, cv2.LINE_AA)

def createMatrix(points):
    '''
        Create a 2d matrix of points arranged from left to right, top to bottom. Each row and column in the matrix is a line.

        Parameters:
            points - array of line intersections
        
        Return value:
            2d matrix of points
    '''
    arr = []
    to_search = points[:]

    while len(to_search) > 0:
        a = sorted(to_search, key=lambda x: x[0] + x[1])[0]
        b = sorted(to_search, key=lambda x: x[0] - x[1])[-1]
    
        row_points = []
        remaining_points = []
        for k in to_search:
            d = k.size  # diameter of the keypoint (might be a threshold)
            dist = np.linalg.norm(np.cross(np.subtract(k, a), np.subtract(b, a))) / np.linalg.norm(b)   # distance between keypoint and line a->b
            if d/2 > dist:
                row_points.append(k)
            else:
                remaining_points.append(k)
        
        arr.insert(len(arr), sorted(row_points, key=lambda x: x[0]))
        to_search = remaining_points
    
    return np.array(arr)

def normalizePoints(points, dim):
    '''
      Unit scaling the points (0-1) 

      Parameters:
        points - 1d array of points
        dim - dimension of the image
      
      Return value:
        points - the normalized 1d array
    '''
    for i in points:
      i[0] /= dim[1]
      i[1] /= dim[0]
    return points

def denormalizePoints(points, dim):
    '''
      Change the scale of points to the image size.

      Parameters:
        points - 1d array of points
        dim - dimension of the image
      
      Return value:
        points - the denormalized 1d array
    '''
    for i in points:
      i[0] *= dim[1]
      i[1] *= dim[0]
    
    return points

def moveStackToSAN(moveStack):
    '''
        Convert the movestack to Standard Algebraic Notation

        Parameters:
            moveStack - stack of moves made in the game
        
        Return value:
            san - string of the Standard Algebraic Notation
    '''
    san = ""

    for idx, move in enumerate(moveStack):
        if idx % 2 == 0:
            san += f"{str((idx//2)+1)}. "
        san += f"{move} "
    
    return san

def addNewLine(text, maxCharsPerLine=97):
    '''
        Split a long string of text to multiple lines

        Parameters:
            text - string to split into multiple lines
            maxCharsPerLine - character limit per line
        Return value:
            string with added \n
    '''
    words = text.split()
    lines = []
    currentLine = ''
    
    for word in words:
        if len(currentLine) + len(word) <= maxCharsPerLine:
            currentLine += word + ' '
        else:
            lines.append(currentLine.strip())
            currentLine = word + ' '
    
    # Add the last line
    lines.append(currentLine.strip())
    
    return '\n'.join(lines)