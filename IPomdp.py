import numpy as np

class IPomdp:
    # TODO: Evaluate how you're passing the parameters
    # def __init__(self, states=None, agent_actions=None, opponent_actions=None, observations=None, transitions=None, observe=None, reward=None, opponent_model=None, discount=None):
    def __init__(self, pomdp, opponent_model=None, discount=None):
        self.states = pomdp.states
        self.observations = pomdp.observations
        self.agent_actions = pomdp.agent_actions
        self.opponent_actions = pomdp.opponent_actions
        self.transitions = pomdp.transitions
        self.observe = pomdp.observe
        self.reward = pomdp.reward
        if discount == None:
            self.discount = 0.95
        else:
            self.discount = discount

    def backup(self, opponent_policy, horizon):
        opt_action = 0
        max_value = 0
        # TODO: reconsider swapping the iteration sequence (ie, actions over states)
        for s_index,s_value in enumerate(self.states):
            for u_index,_ in enumerate(self.agent_actions):
                action_value = 0
                for v_index,_ in enumerate(self.opponent_actions):
                    immediate_R = self.reward[s_index][u_index][v_index] * opponent_policy[s_index][v_index] * s_value
                    action_value = action_value + immediate_R + self.discount * self.sum_over_trans_value(s_index, u_index, v_index, opponent_policy, horizon)

                if action_value > max_value:
                    max_value = action_value
                    opt_action = u_index

        return opt_action

    # def get_immediate_reward(self, u_index, opponent_policy):
    #     ret = 0
    
    #     for s_index, s_value in enumerate(self.states):
    #         for v_index,_ in enumerate(self.opponent_actions):
    #             ret = ret + 
    #     return ret


    # returns the Q value of a state
    # s = state
    def get_state_value(self, s, opponent_policy, horizon):
        if horizon == 0:
            return 0
        max_value = 0

        for u_index,_ in enumerate(self.agent_actions):
            val_sum = 0
            for v_index,_ in enumerate(self.opponent_actions):
                val_sum = val_sum + opponent_policy[s][v_index] * (self.reward[s][u_index][v_index] 
                    + self.discount * self.sum_over_trans_value(s, u_index, v_index, opponent_policy, horizon))
            if val_sum > max_value:
                max_value = val_sum
        return max_value

    
    # Essentially multiplying transition probability by expected value of next state.
    def sum_over_trans_value(self, s, u_index, v_index, opponent_policy, horizon):
        ret = 0
        for s_prime_index,_ in enumerate(self.states):
            summation_holder = 0
            for o_index,_ in enumerate(self.observations):
                summation_holder = summation_holder + (self.observe[s_prime_index][u_index][o_index] 
                    * self.get_state_value(s_prime_index, opponent_policy, horizon - 1))
            ret = ret + (self.transitions[s][u_index][v_index][s_prime_index] * summation_holder)
        return ret

