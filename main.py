from Game import Checkers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.fc1 = nn.Linear(64, 512)
		self.actionProbs = nn.Linear(512, 340)
		self.stateValue = nn.Linear(512, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		pi = F.softmax(self.actionProbs(x), dim=1)
		v = self.stateValue(x)
		return pi, v

def selectAction(env, actions, pi):
	mask = torch.zeros(pi.shape)
	for a in actions:
		mask[0][env.action_space[0][a]] = 1

	# Zeroing out invalid actions and rescaling valids
	pi = pi*mask
	# pi = pi/pi.sum()

	m = Categorical(pi)
	action_idx = m.sample()
	action = env.action_space[1][action_idx.item()]
	return action, m.log_prob(action_idx)

def flip_coord(coord):
		"""
		Takes in coordinate tuple (y, x)
		Returns coordinate from the perspective of the other side
		"""
		coord = (abs(coord[0] - 8 + 1), abs(coord[1] - 8 + 1))
		return coord

def flip_action(action):
	"""
	Takes action tuple (start_coord, end_coord)
	Flips start and end coord
	"""
	action = (flip_coord(action[0]), flip_coord(action[1]))
	return action

def generateEpisode(env, policy):
	board = env.getInitBoard()
	player = 1
	sequence = []
	probs = []
	values = []
	count = 0
	while True:
		count += 1
		actions = env.getValidMoves(board, player)
		if len(actions) == 0:
			looser = player
			break
		if player == -1:
			for i in range(len(actions)):
				actions[i] = flip_action(actions[i])

		state = env.getCanonicalForm(board, player)
		state = torch.tensor(state).unsqueeze(0).float()
		pi, v = policy(state)
		action, prob = selectAction(env, actions, pi)

		sequence.append(player)
		probs.append(prob)
		values.append(v)

		if player == -1:
			action = flip_action(action)
		board, player = env.getNextState(board, player, action)

	sequence = torch.tensor(sequence).unsqueeze(-1)
	probs = torch.stack(probs)
	values = torch.cat(values)
	return sequence, probs, values, looser, count

def getReturns(sequence, looser):
	returns = torch.zeros(sequence.shape)

	gamma = 0.98
	PENALTY = -0.15
	W, L = torch.tensor([[1.0]]), torch.tensor([[-1.0]])
	for t in reversed(range(len(sequence))):
		player = sequence[t]
		if player == looser:
			returns[t] = L
			L = gamma*L + PENALTY
		else:
			returns[t] = W
			W = gamma*W + PENALTY
	return returns


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	env = Checkers()
	policy = Policy()
	opt = torch.optim.Adam(policy.parameters(), lr=0.00001)
	mse = nn.MSELoss(reduction='none')

	hist = []
	epLen = []

	for episode in range(1000):
		sequence, probs, values, looser, count = generateEpisode(env, policy)
		epLen.append(count)
		returns = getReturns(sequence, looser)
		loss = -((returns - values)*probs + mse(returns, values)).sum()
		loss.backward()
		opt.step()
		opt.zero_grad()
		hist.append(loss.item())
		print(f"Episode: {episode}, Episode len: {count}, Loss: {loss.item()}")

	plt.plot(hist)
	plt.show()
	plt.plot(epLen)
	plt.show()