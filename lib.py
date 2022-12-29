import numpy as np
import time
from keras.models import clone_model


episode_end = False

def softmax(x, temperature=1):
    return np.exp(x / temperature) / np.sum(np.exp(x / temperature))


class Node(object):

    def __init__(self, board=None, parent=None, gamma=0.9):
        """
        Game Node for Monte Carlo Tree Search
        Args:
            board: the chess board
            parent: the parent node
            gamma: the discount factor
        """
        self.children = {}  # Child nodes
        self.board = board  # Chess board
        self.parent = parent
        self.values = []  # reward + Returns
        self.gamma = gamma
        self.starting_value = 0

    def update_child(self, move, Returns):
        """
        Update a child with a simulation result
        Args:
            move: The move that leads to the child
            Returns: the reward of the move and subsequent returns

        Returns:

        """
        child = self.children[move]
        child.values.append(Returns)

    def update(self, Returns=None):
        """
        Update a node with observed Returns
        Args:
            Returns: Future returns

        Returns:

        """
        if Returns:
            self.values.append(Returns)

    def select(self, color=1):
        """
        Use Thompson sampling to select the best child node
        Args:
            color: Whether to select for white or black

        Returns:
            (node, move)
            node: the selected node
            move: the selected move
        """
        assert color == 1 or color == -1, "color has to be white (1) or black (-1)"
        if self.children:
            max_sample = np.random.choice(color * np.array(self.values))
            max_move = None
            for move, child in self.children.items():
                child_sample = np.random.choice(color * np.array(child.values))
                if child_sample > max_sample:
                    max_sample = child_sample
                    max_move = move
            if max_move:
                return self.children[max_move], max_move
            else:
                return self, None
        else:
            return self, None

    def simulate(self, model, board, depth=0, max_depth=4, random=False, temperature=1):
        """
        Recursive Monte Carlo Playout
        Args:
            model: The model used for bootstrap estimation
            env: the chess environment
            depth: The recursion depth
            max_depth: How deep to search
            temperature: softmax temperature

        Returns:
            Playout result.
        """
        board_in = board.fen()
        layer_board = layerboard(board)
        if board.turn and random:
            move = np.random.choice([x for x in board.generate_legal_moves()])
        else:
            successor_values = []
            for move in board.generate_legal_moves():
                episode_end, reward = step(board, move)
                result = board.result()

                if (result == "1-0" and board.turn) or (
                        result == "0-1" and not board.turn):
                    board.pop()
                    layer_board = layerboard(board)
                    break
                else:
                    if board.turn:
                        sucval = reward + self.gamma * np.squeeze(
                            model.predict(np.expand_dims(layer_board, axis=0)))
                    else:
                        sucval = np.squeeze(np.random.randint(-5, 5) / 5)
                    successor_values.append(sucval)
                    board.pop()
                    layer_board = layerboard(board)

            if not episode_end:
                if board.turn:
                    move_probas = softmax(np.array(successor_values), temperature=temperature)
                    moves = [x for x in board.generate_legal_moves()]
                else:
                    move_probas = np.zeros(len(successor_values))
                    move_probas[np.argmax(successor_values)] = 1
                    moves = [x for x in board.generate_legal_moves()]
                if len(moves) == 1:
                    move = moves[0]
                else:
                    move = np.random.choice(moves, p=np.squeeze(move_probas))

        episode_end, reward = step(board, move)

        if episode_end:
            Returns = reward
        elif depth >= max_depth:  # Bootstrap the Monte Carlo Playout
            Returns = reward + self.gamma * np.squeeze(model.predict(np.expand_dims(layer_board, axis=0)))
        else:  # Recursively continue
            Returns = reward + self.gamma * self.simulate(model, board, depth=depth + 1,temperature=temperature)

        board.pop()
        np.random.randint(-5, 5) / 5

        board_out = board.fen()
        assert board_in == board_out

        if depth == 0:
            return Returns, move
        else:
            noise = np.random.randn() / 1e6
            return Returns + noise


def predictmove(model, board):
    tree = Node(board)
    layer_board = layerboard(board)
    state = np.expand_dims(layer_board.copy(), axis=0)
    state_value = model.predict(state)
    start_mcts_after = -1
    tree = mcts(tree, board, model)
    # Step the best move
    max_move = None
    max_value = np.NINF
    for move, child in tree.children.items():
        sampled_value = np.mean(child.values)
        if sampled_value > max_value:
            max_value = sampled_value
            max_move = move
    return max_move
    
def layerboard(board):
    """
    Initalize the numerical representation of the environment
    Returns:

    """
    mapper = {"p": 0, "r": 1, "n": 2, "b": 3, "q": 4, "k": 5, "P": 0, "R": 1, "N": 2, "B": 3, "Q": 4, "K": 5}
    layer_board = np.zeros(shape=(8, 8, 8))
    for i in range(64):
        row = i // 8
        col = i % 8
        piece = board.piece_at(i)
        if piece == None:
            continue
        elif piece.symbol().isupper():
            sign = 1
        else:
            sign = -1
        layer = mapper[piece.symbol()]
        layer_board[layer, row, col] = sign
        layer_board[6, :, :] = 1 / board.fullmove_number
    if board.turn:
        layer_board[6, 0, :] = 1
    else:
        layer_board[6, 0, :] = -1
    layer_board[7, :, :] = 1
    return layer_board

def mcts(node, board, model, gamma = 0.9,search_time=1, min_sim_count=10):
        """
        Run Monte Carlo Tree Search
        Args:
            node: A game state node object

        Returns:
            the node with playout sims

        """

        starttime = time.time()
        sim_count = 0
        board_in = board.fen()
        layer_board = layerboard(board)

        # First make a prediction for each child state
        for move in board.generate_legal_moves():
            if move not in node.children.keys():
                node.children[move] = Node(board, parent=node)

            episode_end, reward = step(board,move)

            if episode_end:
                successor_state_value = 0
            else:
                successor_state_value = np.squeeze(
                    model.predict(np.expand_dims(layer_board, axis=0), verbose=0)
                )

            child_value = reward + gamma * successor_state_value

            node.update_child(move, child_value)
            board.pop()
            layer_board = layerboard(board)
        if not node.values:
            node.values = [0]

        while starttime + search_time > time.time() or sim_count < min_sim_count:
            depth = 0
            color = 1
            node_rewards = []

            # Select the best node from where to start MCTS
            while node.children:
                node, move = node.select(color=color)
                if not move:
                    # No move means that the node selects itself, not a child node.
                    break
                else:
                    depth += 1
                    color = color * -1  # switch color
                    episode_end, reward = step(board,move)  # Update the environment to reflect the node
                    node_rewards.append(reward)
                    # Check best node is terminal

                    if board.result() == "1-0" and depth == 1:  # -> Direct win for white, no need for mcts.
                        board.pop()
                        layer_board = layerboard(board)
                        node.update(1)
                        node = node.parent
                        return node
                    elif episode_end:  # -> if the explored tree leads to a terminal state, simulate from root.
                        while node.parent:
                            board.pop()
                            layer_board = layerboard(board)
                            node = node.parent
                        break
                    else:
                        continue

            # Expand the game tree with a simulation
            Returns, move = node.simulate(clone_model(model),
                                          board,
                                          depth=0)
            layer_board = layerboard(board)

            if move not in node.children.keys():
                node.children[move] = Node(board, parent=node)

            node.update_child(move, Returns)

            # Return to root node and backpropagate Returns
            while node.parent:
                latest_reward = node_rewards.pop(-1)
                Returns = latest_reward + gamma * Returns
                node.update(Returns)
                node = node.parent

                board.pop()
                layer_board = layerboard(board)
            sim_count += 1

        board_out = board.fen()
        assert board_in == board_out

        return node

def step(board, action):
        """
        Run a step
        Args:
            action: python chess move
        Returns:
            epsiode end: Boolean
                Whether the episode has ended
            reward: float
                Difference in material value after the move
        """
        layer_board = layerboard(board)
        piece_balance_before = get_material_value(layer_board)
        board.push(action)
        layer_board = layerboard(board)
        piece_balance_after = get_material_value(layer_board)
        auxiliary_reward = (piece_balance_after - piece_balance_before) * 0.01
        result = board.result()
        reward = 0
        episode_end = False
        if result == "*":
            reward = 0
            episode_end = False
        elif result == "1-0":
            reward = 1
            episode_end = True
        elif result == "0-1":
            reward = -1
            episode_end = True
        elif result == "1/2-1/2":
            reward = 0
            episode_end = True
        reward += auxiliary_reward

        return episode_end, reward
def get_material_value(layer_board):
        """
        Sums up the material balance using Reinfield values
        Returns: The material balance on the board
        """
        pawns = 1 * np.sum(layer_board[0, :, :])
        rooks = 5 * np.sum(layer_board[1, :, :])
        minor = 3 * np.sum(layer_board[2:4, :, :])
        queen = 9 * np.sum(layer_board[4, :, :])
        return pawns + rooks + minor + queen