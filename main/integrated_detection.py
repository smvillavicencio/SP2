import cv2
import numpy as np
from chessboard import display
import chess

from detect_board import detectBoard
from detect_pieces import detectPieces
from position import locatePiecesOnBoard, boardToFen, getMove
from utilities import drawPoints, denormalizePoints, resizeImg, createMatrix, moveStackToSAN


def detect(source, whitePos):
	print("Opening camera...")
	cap = cv2.VideoCapture(source)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 750)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

	board = None
	boardFound = False
	boardReady = False
	moveStack = []
	result = None
	game = display.start('')


	while True:
		try:
			_, frame = cap.read()
			frame2, scale = resizeImg(frame)

			if board is not None:
				drawPoints(frame, board, (255,0,0), 1)

			cv2.imshow("Stream", frame)

			if not boardFound:
				board = detectBoard(frame2) 
				if board is not None:
					boardFound = True
					board = denormalizePoints(board, frame.shape)
					matrix = createMatrix(np.array(board))
					print("Chessboard found!")
					print("Press SPACE to redetect board")
					print("Press 1 when board is ready")

			if boardFound:
				if cv2.waitKey(30) == 32: # redetect board on space
					boardFound = False
					# newBoard = detectBoard(frame2) 
					# if newBoard is not None:
					#     board = newBoard
					#     board = denormalizePoints(board, frame.shape)
					#     matrix = createMatrix(np.array(board))
						
				
				if not boardReady and cv2.waitKey(30) == 49: # start detection of pieces on 1
					boardReady = True
					cBoard = chess.Board()
					display.update(cBoard.board_fen(), game)
					print("Board ready!")
					print("Press 2 to end turn")
					print("Press D if draw was agreed")
					print("Press W if white resigned")
					print("Press B if black resigned")

				if boardReady and result == None:
					if cv2.waitKey(30) == 50: # press 2 after a player moved 
						print("Detecting pieces...")
						currentPieces = detectPieces(frame)
						chessboard = locatePiecesOnBoard(currentPieces, matrix, whitePos)
						currentFen = boardToFen(chessboard)
						cBoard, moveStack = getMove(cBoard, currentFen, moveStack)
						display.update(cBoard.board_fen(), game)

						if moveStack[-1][-1] == '#':
							if len(moveStack) % 2 == 1:
								result = '1-0'
								print("White wins")
							else:
								result = '0-1'
								print("Black wins")
							return result, moveStackToSAN(moveStack)
						elif len(list(cBoard.legal_moves)) == 0:
							result = '1/2-1/2'
							print("Draw")
							return result, moveStackToSAN(moveStack)

					elif cv2.waitKey(30) == ord('d'): #draw on D
						result = '1/2-1/2'
						print("Draw was agreed")
						return result, moveStackToSAN(moveStack)

					elif cv2.waitKey(30) == ord('w'): #white resign on W
						result = '0-1'
						print("Black wins by white resignation")
						return result, moveStackToSAN(moveStack)
					
					elif cv2.waitKey(30) == ord('b'): #black resign on B
						result = '1-0'
						print("White wins by black resignation")
						return result, moveStackToSAN(moveStack)

			display.check_for_quit()
			if cv2.waitKey(30) == 27: # exit on escape/ongoing
				result = "*"
				return result, moveStackToSAN(moveStack)

		except Exception as e:
			print(e)

	cap.release()
	cv2.destroyAllWindows()