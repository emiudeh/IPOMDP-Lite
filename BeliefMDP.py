import numpy as np

class BeliefMDP:
    # TODO: Evaluate how you're passing the parameters
    def __init__(self, pomdp, discount = None):
        self.states = pomdp.states
        self.agent_actions = pomdp.agent_actions
        self.opponent_actions = pomdp.opponent_actions
        self.transitions = pomdp.transitions
        self.reward = pomdp.reward
        if discount == None:
            self.discount = 0.95
        else:
            self.discount = discount

    # def solve_nested_MDP(self, reasoning_level):
    #     # policy representative of level1 uniform reasoning of the opponent

    #     opponent_policy_for_level = np.full((self.states, self.opponent_actions), 1/self.opponent_actions.size)
    #     policy_list = []
    #     for level in range (reasoning_level):
    #         level_policy = self.get_counter_policy(opponent_policy_for_level)
    #         policy_list.append(level_policy)
    #         if level != reasoning_level - 1:
    #             opponent_policy_for_level = self.get_counter_policy(level_policy)
                
    #     ret_policy = np.zeros((self.states, self.agent_actions))
    #     for s in self.states:
    #         for u in self.agent_actions:
    #             temp = 0
    #             for p in policy_list:
    #                 temp = temp + p[s][u]
    #             ret_policy[s][u] = temp/len(policy_list)
    #     return ret_policy


    def get_counter_policy(self, opponent_policy, horizon):
        ret_policy = np.zeros((self.states.shape[0], self.agent_actions.shape[0]))
        # TODO: reconsider swapping the iteration sequence (ie, actions over states)
        for s,_ in enumerate(self.states):
            for u,_ in enumerate(self.agent_actions):
                for v,_ in enumerate(self.opponent_actions):
                    ret_policy[s][u] = ret_policy[s][u] + opponent_policy[s][v] * (self.reward[s][u][v] + self.discount * self.sum_over_trans_value(s, u, v, opponent_policy, horizon))
            ret_policy[s] = ret_policy[s]/np.sum(ret_policy[s])
        return ret_policy

            
    # returns the Q value of a state
    # s = state
    def get_state_value(self, s, opponent_policy, horizon):
        if horizon == 0:
            return 0
        val_list = []
        for u,_ in enumerate(self.agent_actions):
            val_sum = 0
            for v,_ in enumerate(self.opponent_actions):
                val_sum = val_sum + opponent_policy[s][v] *  (self.reward[s][u][v] + self.discount * self.sum_over_trans_value(s, u, v, opponent_policy, horizon))
            val_list.append(val_sum)
        return max(val_list)


    def sum_over_trans_value(self, s, u, v, opponent_policy, horizon):
        val = 0
        for s_prime,_ in enumerate(self.states):
            val = val + (self.transitions[s][u][v][s_prime] * self.get_state_value(s_prime, opponent_policy, horizon - 1))
        return val



