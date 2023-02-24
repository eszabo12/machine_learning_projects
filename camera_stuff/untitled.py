


def hasWon(player, board):
	#player will be a char, 'X' or 'O'

#test on an example where the top row is the same, 'o' has won
#the leftmost column is the same

	#three in a column
	for x in range(3):
		#boolean for the same
		same = true
		for y in range(3):
			element = board[x][y]
			if (element != player):
				same = false
			#this will be 'X' or 'O'
			if y == 0:
				continue
			if element != board[x][y-1]:
				same = false
		if (same == true):
			return true

	#three in a column
	for y in range(3):
		same = true
		for x in range(3):
			element = board[x][y]
			if (element != player):
				same = false
			if x == 0:
				continue
			if element != board[x-1][y]:
				same = false
		if (same == true):
			return true
			#this will be 'X' or 'O'
			
	#three in a diagonal
	if (board[0][0] == player == board[1][1] == board[2][2]):
		return true
	if (board[2][0] == player == board[1][1] == board[0][2]):
		return true
	return false








