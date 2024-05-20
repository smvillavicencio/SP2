import numpy as np

def convertPiece(value):
    pieceMap = {
        0: "b",
        1: "k",
        2: "n",
        3: "p",
        4: "q",
        5: "r",
        6: "B",
        7: "K",
        8: "N",
        9: "P",
        10: "Q",
        11: "R",
    }

    return pieceMap[value]

def isPointInsidePolygon(point, polygon):
    '''
        Uses the winding number algorithm to determine if the point is inside the polygon
        Source: https://www.dgp.toronto.edu/~mac/e-stuff/point_in_polygon.py

        Parameters:
            point - the point to check
            polygon - array of points of the polygon

        Return value:
            wn - the winding number, if it is equal to 0, it means point is outside the polygon 
    '''
    def is_left(P0, P1, P2):
        return (P1[0] - P0[0]) * (P2[1] - P0[1]) - (P2[0] - P0[0]) * (P1[1] - P0[1])
    
    wn = 0   # the winding number counter
    # repeat the first vertex at end
    polygon = tuple(polygon[:]) + (polygon[0],)
    # loop through all edges of the polygon
    for i in range(len(polygon)-1):     # edge from polygon[i] to polygon[i+1]
        if polygon[i][1] <= point[1]:        # start y <= point[1]
            if polygon[i+1][1] > point[1]:     # an upward crossing
                if is_left(polygon[i], polygon[i+1], point) > 0: # point left of edge
                    wn += 1           # have a valid up intersect
        else:                      # start y > point[1] (no test needed)
            if polygon[i+1][1] <= point[1]:    # a downward crossing
                if is_left(polygon[i], polygon[i+1], point) < 0: # point right of edge
                    wn -= 1           # have a valid down intersect
    return wn

def locatePiecesOnBoard(pieces, chessboard, whitePos):
    '''
        Locate the pieces on an 8x8 chessboard

        Parameters:
            pieces - array of detected pieces
            chessboard - array of detected intersections
            whitePos - 0 or 1 depending on the position of the white pieces (POV of the web cam)
        
        Return value:
            8x8 matrix with the located pieces 
    '''
    board = np.full((8, 8), fill_value='_', dtype=str)

    left = min([row[0][0] for row in chessboard])
    right = max([row[-1][0] for row in chessboard])
    top = min([point[1] for point in chessboard[0]])
    bot = max([point[1] for point in chessboard[-1]])

    sortedPieces = sorted(sorted(pieces, key=lambda x: x[1][0]), key=lambda x: x[1][1]) # sort by y-coordinate first then by x-coordinate
    pieceIndex = 0

    for row in range(len(chessboard)-1):
        for col in range(len(chessboard[row])-1):
            if board[row][col] == '_':
                # corners of the square
                uLeft = chessboard[row][col]
                dLeft = chessboard[row+1][col]
                uRight = chessboard[row][col+1]
                dRight = chessboard[row+1][col+1]

                while pieceIndex < len(sortedPieces):
                    value, point = sortedPieces[pieceIndex]
                    x, y = point

                    if left <= x <= right and top <= y <= bot: # if part of the chess board
                        if isPointInsidePolygon(point, [uRight, uLeft, dLeft, dRight]):
                            board[row][col] = convertPiece(value)
                            sortedPieces.pop(pieceIndex)
                            break
                        else:
                            pieceIndex += 1
                    else:
                        sortedPieces.pop(pieceIndex)
                pieceIndex = 0

    return np.flip(board.T, int(whitePos))
  
def boardToFen(board):
    '''
        Create the FEN string from a given board

        Parameters:
            board - 8x8 matrix to convert

        Return value:
            FEN string to repersent the position 
    '''
    fen = ''
    emptyCounter = 0

    for row in board:
        for cell in row:
            if cell == '_':
                emptyCounter += 1
            else:
                if emptyCounter > 0:
                    fen += str(emptyCounter)
                    emptyCounter = 0
                fen += cell
        if emptyCounter > 0:
            fen += str(emptyCounter)
            emptyCounter = 0 
        fen += '/'
    return fen[:-1] # remove the last slash

def getMove(board, fen, moveStack):
    '''
        Get the move made from the current FEN

        Parameters:
            board - state of the board before the latest move
            fen - FEN string of the current board
            moveStack - stack of the moves made throughout the game
        
        Return values:
            board - the current board state after determining the move
            moveStack - updated moveStack
    '''
    for i in board.legal_moves:
      move = board.san(i)
      board.push(i)
      if board.board_fen() == fen:
        moveStack.append(move)
        return board, moveStack
      _ = board.pop()

    