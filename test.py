import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from Game import Checkers

class Policy(nn.Module):
	def __init__(self):
		super(Policy, self).__init__()
		self.fc1 = nn.Linear(64, 512)
		self.fc2 = nn.Linear(512, 512)
		self.probs = nn.Linear(512, 340)
		self.value = nn.Linear(512, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		pi = F.softmax(self.probs(x), dim=1)
		v = self.value(x)
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

def train(env, policy, numEpisodes):
	MOVE_PENALTY = -0.01
	GAMMA = 0.80
	hist = []
	for episode in range(numEpisodes):
		env.getInitBoard()
		board = env.board
		trainExamples = []
		player = 1
		count = 0

		skip = False
		while True:
			count += 1
			if len(trainExamples) > 1000:
				skip = True
				break

			actions = env.getValidMoves(board, player)
			if len(actions) == 0:
				looser = player
				break

			state = env.getCanonicalForm(board, player)
			state = torch.tensor(state).float().unsqueeze(0)

			pi, value = policy(state)

			action, action_prob = selectAction(env, actions, pi)

			trainExamples.append([state, value, action_prob, player])

			board, player = env.getNextState(board, player, action)
	# [print(action_prob, player) for _, _, action_prob, player in trainExamples]
	# 	if not skip:
	# 		loss = []
	# 		W, L = torch.tensor([[1.0]]), torch.tensor([[-1]])
	# 		for x in trainExamples[::-1]:
	# 			state, value, prob, player = x[0], x[1], x[2], x[3]
	# 			if player == looser:
	# 				error = -L*prob
	# 				loss.append(error)
	# 				L = MOVE_PENALTY + L
	# 			else:
	# 				error = -W*prob
	# 				loss.append(error)
	# 				W = MOVE_PENALTY + W

				

	# 		loss = torch.stack(loss).mean()
	# 		print(f"Ep: {episode}, Ep Len: {count}, Loss: {loss.item()}", end='\n\n')
	# 		hist.append(loss.item())

	# 		opt.zero_grad()
	# 		loss.backward()
	# 		opt.step()


	# torch.save(policy.state_dict(), 'policy.pt')
	# return hist

policy = Policy()
mse = nn.MSELoss()
opt = torch.optim.Adam(policy.parameters(), lr=0.00001)
env = Checkers()

hist = train(env, policy, 1)
# plt.plot(hist)
# plt.show()