import numpy as np

class Checkers():
    """
    1 represents red piece
    2 represents red crown piece
    -1 represents black piece
    -2 represents black crown piece
    0 represents empty squares
    """
    def __init__(self):
        self.size = 8
        self.board = self.getInitBoard()
        self.action_space = self.getInitActions()

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        self.board = np.zeros((self.size, self.size))
        for y in range(0, 3):
            for x in range(self.size):
                if (x + y) % 2 != 0:
                    self.board[y, x] = 1
        for y in range(5, self.size):
            for x in range(self.size):
                if (x + y) % 2 != 0:
                    self.board[y, x] = -1
        return self.board

    def withinBounds(self, loc):
        bounds = False
        if 0 <= loc[0] < self.size and 0 <= loc[1] < self.size:
            bounds = True
        return bounds

    def getActions(self, loc):
        actions = []
        actions.append((loc, (loc[0] - 1, loc[1] - 1)))
        actions.append((loc, (loc[0] - 1, loc[1] + 1)))
        actions.append((loc, (loc[0] - 2, loc[1] - 2)))
        actions.append((loc, (loc[0] - 2, loc[1] + 2)))
        actions.append((loc, (loc[0] + 1, loc[1] - 1)))
        actions.append((loc, (loc[0] + 1, loc[1] + 1)))
        actions.append((loc, (loc[0] + 2, loc[1] - 2)))
        actions.append((loc, (loc[0] + 2, loc[1] + 2)))
        return actions

    def getInitActions(self):
        count = 0
        actionToIndex = {}
        indexToAction = {}
        for y in range(self.size):
            for x in range(self.size):
                actions = self.getActions((y, x))
                for action in actions:
                    end = action[1]
                    if self.withinBounds(end):
                        actionToIndex[action] = count
                        indexToAction[count] = action
                        count += 1
        return (actionToIndex, indexToAction)

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (self.size, self.size)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return len(self.action_space[0])

    def midpoint(self, start, end):
        """
        Input:
            start: current coord
            end: coord to move to
        Returns:
            mid: midpoint between start and end
        """
        y_mid = int(abs(start[0] + end[0])/2)
        x_mid = int(abs(start[1] + end[1])/2)
        mid = (y_mid, x_mid)
        return mid

    def isJumpMove(self, start, end):
        y_diff = abs(start[0] - end[0])
        x_diff = abs(start[1] - end[1])
        jump = False
        if y_diff > 1 and x_diff > 1:
            jump = True
        return jump

    def getNextState(self, board, player, action):
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            board: board after applying action
            player: player who plays in the next turn (should be -player)
        """
        start, end = action[0], action[1]
        ## Moves piece
        piece = board[start]
        board[start] = 0
        board[end] = piece
        ## Handles crown condition
        if (piece == 1 and end[0] == self.size-1) or (piece == -1 and end[0] == 0):
            board[end] = board[end]*2
        ## Handles capture condition
        if self.isJumpMove(start, end):
            mid = self.midpoint(start, end)
            board[mid] = 0
        else:
            player = -player
        return board, player

    def getValidMoves(self, board, player):
        """
        Input:
            board: current board
            player: current player

        Returns:
            valids: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        ## Gets location of pieces
        pieces = []
        for y in range(self.size):
            for x in range(self.size):
                square = board[y, x]
                if square in (player, player*2):
                    pieces.append((y, x))
        ## Adds valid actions for each piece to list
        valids = []
        for piece in pieces:
            actions = self.getActions(piece)
            for action in actions:
                start, end = action[0], action[1]
                ## Discard out of bounds action
                if not self.withinBounds(end):
                    continue
                ## Discard if square is not empty
                if board[end] != 0:
                    continue
                ## Discard invalid jump moves
                mid = self.midpoint(start, end)
                if self.isJumpMove(start, end) and (board[mid] not in (-player, -player*2)):
                    continue
                ## Discard attempted crown moves when not crowned
                if board[piece] == 1 and (end[0] - start[0] < 0):
                    continue
                if board[piece] == -1 and (end[0] - start[0] > 0):
                    continue
                valids.append(action)
        return valids

    def getCanonicalForm(self, board, player):
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player.
        """
        if player == 1:
            return board.flatten()
        else:
            return -np.rot90(board, 2).flatten()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    env = Checkers()
    env.getInitBoard()
    board = env.board
    counter = 0

    player = 1
    while True:
        counter += 1

        actions = env.getValidMoves(board, player)
        if len(actions) == 0:
            print(f"Episode len: {counter}, Looser: {player}")
            break
        idx = np.random.randint(len(actions))
        action = actions[idx]

        board, player = env.getNextState(board, player, action)