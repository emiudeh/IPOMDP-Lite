'''
        This file/code is only utilized for testing to ensure that the planner is behaving reasonable.
        It's a random test case I created for personal testing and isn't part of the planner logic.
'''

import numpy as np
from BeliefMDP import BeliefMDP
from Pomdp import Pomdp
from IPomdp import IPomdp

states = np.array([0.8, 0, 0.2, 0])
observations = np.array([0.8, 0, 0.2, 0.0 ])
actions = np.array([0, 1])

transitions = np.array([
    [[[0.05, 0.125, 0.7, 0.125],[0.125, 0.05, 0.125, 0.7]], [[0.125, 0.7, 0.125, 0.05],[0.7, 0.125, 0.05, 0.125]]],
    [[[0.125, 0.05, 0.125, 0.7],[0.05, 0.125, 0.7, 0.125]], [[0.7, 0.125, 0.05, 0.125],[0.125, 0.7, 0.125, 0.05]]],
    [[[0.7, 0.125, 0.05, 0.125],[0.125, 0.7, 0.125, 0.05]], [[0.125, 0.05, 0.125, 0.7],[0.05, 0.125, 0.7, 0.125]]],
    [[[0.125, 0.7, 0.125, 0.05],[0.7, 0.125, 0.05, 0.125]], [[0.05, 0.125, 0.7, 0.125],[0.125, 0.05, 0.125, 0.7]]]]
)

observe = np.array([
    [[0,15, 0.05, 0.075, 0.725],[0,8, 0.1, 0.03, 0.07]],
    [[0.1, 0.1, 0.6, 0.2],[0.15, 0.7, 0.075, 0.075]],
    [[0.2, 0.6, 0.1, 0.1],[0.05, 0.075, 0.725, 0.15]],
    [[0.5, 0.3, 0.1, 0.1],[0.1, 0.1, 0.2, 0.6]]]
)

reward = np.array([
    [[[-1],[50]], [[-1], [-1]]],
    [[[50],[-1]], [[-1], [-1]]],
    [[[-1],[-1]], [[50], [-1]]],
    [[[-1],[-1]], [[-1], [50]]]]
)

pomdp = Pomdp(states, observations, actions, transitions, observe, reward)

opponent = BeliefMDP(pomdp)
opponent.reward = np.swapaxes(opponent.reward,1,2)
opponent.transitions = np.swapaxes(opponent.transitions,1,2)
op_model = np.zeros((states.shape[0], actions.shape[0])) 

k = 3
horizon = 4
assumed_policy = np.full((states.shape[0], actions.shape[0]), 0.5)
for i in range (0, k):
    levelk = opponent.get_counter_policy(assumed_policy, horizon)
    op_model += levelk
    if i < k-1:
        assumed_policy = BeliefMDP(pomdp).get_counter_policy(levelk, horizon)


op_model = op_model/k
print("\n\nDeduced Opponent MDP model: \n", op_model)
# print(op_model)
# # print(np.zeros((states.shape[0], actions.shape[0])))

agent = IPomdp(pomdp)
print()
print(agent.backup(op_model, 3))
print("\n")
