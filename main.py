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
		self.probs = nn.Linear(512, 340)
		self.value = nn.Linear(512, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		pi = F.softmax(self.probs(x), dim=1)
		v = self.value(x)
		return (pi, v)

policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=0.00005)
mse = nn.MSELoss()
env = Checkers()

thing = []
for episode in range(1000):
	env.getInitBoard()
	board = env.board
	counter = 0

	trainExamples = []
	player = 1
	while True:
		counter += 1

		actions = env.getValidMoves(board, player)
		if len(actions) == 0:
			looser = player
			print(f"Episode len: {counter}, Looser: {looser}")
			break

		state = env.getCanonicalForm(board, player)
		state = torch.tensor(state).float().unsqueeze(0)

		pi, v = policy(state)
		mask = torch.zeros(pi.shape)
		for a in actions:
			mask[0][env.action_space[0][a]] = 1

		pi = pi*mask
		pi = pi/pi.sum()

		m = Categorical(pi)
		action_idx = m.sample()
		prob = m.log_prob(action_idx)
		trainExamples.append([prob, v, player, None])

		action = env.action_space[1][action_idx.item()]
		board, player = env.getNextState(board, player, action)

	trainExamples = [(x[0], x[1], (torch.tensor([[-1.0]]) if x[2] == looser else torch.tensor([[1.0]]))) for x in trainExamples]

	loss = []
	for x in trainExamples:
		# (z-v)^2 - (z-v)*pi
		loss.append(mse(x[2], x[1]) - (x[2]-x[1])*prob)

	loss = torch.stack(loss).sum()
	loss.backward()

	error = loss.item()/counter
	thing.append(error)
	print(error)
	opt.step()

plt.plot(thing)
plt.show()