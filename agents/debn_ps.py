# -*- coding: utf-8 -*-

import random
import numpy as np
from collections import deque
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchsummary import summary

_INTERACTION = namedtuple('Interaction', ('percept', 'action', 'reward'))


########################################################################
#       DEBN
# ---------------
#
# Defines a feedforward DNN that represents a DEBN approximating merit values.
#
#


class DEBN(nn.Module):
    """
    A feedforward NN implementing a DEBN.

    Parameters:
        dim_percept:    int
                        the dimension of the percept space
        dim_action:     int
                        the dimension of the action space
        dim_hidden_:    list of int
                        the dimensions of hidden layers
        dropout_rate_   list of float
                        the probability of a hidden neuron to be dropped in each hidden layer
    """
    def __init__(self, dim_percept, dim_action, dim_hidden_=[32], dropout_rate_=[0.], train_output_weights = False):
        super(DEBN, self).__init__()
        #initialisation
        init_kaiman = True

        #input layer
        self.visible = nn.Linear(dim_percept+dim_action, dim_hidden_[0])
        self.visible.apply(self._init_weights) if init_kaiman else None

        self.b_input = nn.Linear(dim_percept+dim_action, 1)
        nn.init.constant_(self.b_input.bias, 0.)
        self.b_input.bias.requires_grad = False

        #hidden layers
        self.hidden = nn.ModuleList()
        for l in range(len(dim_hidden_)-1):
            self.hidden.append(nn.Linear(dim_hidden_[l], dim_hidden_[l+1]))

        for l in range(len(dim_hidden_) - 1):
            self.hidden[l].apply(self._init_weights) if init_kaiman else None

        #output layer
        self.output = nn.Linear(dim_hidden_[-1]+1, 1)
        if not(train_output_weights):
            nn.init.constant_(self.output.weight, 1.)
            self.output.weight.requires_grad = False

        #dropout layers
        self.dropout = nn.ModuleList()
        for l in range(len(dropout_rate_)):
            self.dropout.append(nn.Dropout(p=dropout_rate_[l]))

        # ##checkflags
        # print("MODEL-SUMMARY:")
        # summary(self,(1, dim_percept+dim_action))


    def _init_weights(self, layer):
        """
        Initializes weights with kaiming_uniform method for ReLu from PyTorch.
        """
        if type(layer) == nn.Linear:
            #nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.kaiming_uniform_(layer.weight, a=0, mode='fan_in', nonlinearity='relu')

    def forward(self, percept, action):

        out = torch.cat((percept, action), dim=1)

        bias = self.b_input(out)
        out = F.relu(self.visible(out))
        for l in range(len(self.hidden)):
            out = self.dropout[l](out)
            out = F.relu(self.hidden[l](out))
        out = self.dropout[-1](out)
        out = torch.cat((out,bias),1)
        out = self.output(out)
        return out


#########################################################
#   DEBN Agent
# ----------------
#
# Defines the agent that interacts with the environment.
#
#

class DEBNAgent():
    """
    The agent that uses a DEBN (represented by a feedforward NN) to predict  merit values.
    Parameters:
        dim_percept     int
                        the dimension of the percept space, i.e. #percept neurons
        dim_action:     int
                        the dimension of the action space, i.e. #action neurons
        all_actions:    torch.Tensor
                        all possible actions collected in a list, each of tensor.size([1,m])
                        ideally in a (discrete and componetized) one-hot encoding
        dim_hidden:     list of int
                        the dimensions of hidden layers
        dropout_rate:   list of float
                        the probability of a hidden neuron to be dropped in each layer
        target_update:  int
                        the number of trials that have to be passed before the
                        policy network is copied to the target network
        device:         str
                        the device on which the DEBN is run, either "cpu" or "cuda"
        learning_rate:  float
                        learning rate >=0. for optimizer
        capacity:       int
                        the maximum size of the memory on which the DEBN is trained
        gamma:           float
                        the gamma parameter
        batch_size:     int
                        the size >1 of the batches sampled from the memory after each
                        trial to train the DEBN
        replay_time:    int
                        number of interactions with the environment before experience replay
                        is invoked
    """
    def __init__(self, dim_percept, dim_action, all_actions, dim_hidden=[32], dropout_rate=[0.], target_update=50, device="cpu", learning_rate=0.01,
                 capacity=1000, gamma=0.9, batch_size=100, replay_time=10, episodic = True, train_output_weights = None):
        #ERRORS
        if batch_size <= 1:
            raise ValueError("Invalid batch size: batch_size={}.".format(batch_size))
        if device != "cpu":
            raise NotImplementedError("GPU usage is not supported yet: device=\"{}\".".format(device)
                                     )
        if learning_rate < 0.:
            raise ValueError("Invalid learning rate: learning_rate={}.".format(learning_rate))
        if len(dim_hidden) != len(dropout_rate):
            raise ValueError("Number of hidden layers \"{}\" should equal".format(len(dim_hidden))+\
                             " number of dropout layers \"{}\".".format(len(dropout_rate))
                            )
        if gamma > 1. or gamma < 0.:
            raise ValueError("Invalid gamma value: gamma=\"{}\".".format(gamma))
        ##############

        #the policy network for choosing actions
        if train_output_weights is None:
            train_output_weights = False
            if episodic:
                train_output_weights = True
        self.policy_net = DEBN(dim_percept, dim_action, dim_hidden_=dim_hidden,
                              dropout_rate_=dropout_rate, train_output_weights=train_output_weights).to(device)

        #PARAMETERS
        #parameters as in class description
        self.all_actions = all_actions
        self.target_update = target_update
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_count = 0
        self.replay_time = replay_time
        self.gamma_ps = 1
        self.episodic = episodic

        #INTERNAL VARS
        #the optimizer for the policy network
        self._optimizer = optim.Adam(filter(lambda p: p.requires_grad, 
                                            self.policy_net.parameters()),
                                     lr=learning_rate, amsgrad=True#, weight_decay=1e-5
                                    )
        #the target network for learning
        self._target_net = DEBN(dim_percept, dim_action, dim_hidden_=dim_hidden,
                               dropout_rate_=dropout_rate, train_output_weights=train_output_weights).to(device)
        self._target_net.load_state_dict(self.policy_net.state_dict())
        #the memory of the agent
        self._memory = deque(maxlen=capacity)
        if self.episodic:
            #the percepts and actions of the current trial
            self._trial_data = deque()
            #the rewards of the current trial
            self._trial_rewards = torch.empty(0)
        #previous percept
        self._prev_s = None
        #previous action
        self._prev_a = None
        if not(self.episodic):
            #save the latest interaction in buffer_memory before calculating the accumulated reward
            self._buffer_memory = deque(maxlen=2*replay_time)
        #number of interactions
        self._num_interactions = 0


    def deliberate_and_learn(self, percept, action, reward, eta_e, done, softmax_e=1):
        """
        This is the primary function that represents the agent's reaction to an interaction with
        the environment.
        Following an interaction with the environment, the agent receives a percept (representing
        the current state of the environment) and a reward.
        (1) The reward is saved alongside the previous percept and preceding action to the trial
            data.
        (2) At the end of a trial, the accumulated data from the trial is saved to the agent's
            memory and the agent is trained. In addition, the agent is trained whenever
            X (=replay_time) number of interactions have passed since the last training.
            The network receives a tensor of torch.size([n+m]) where
            n=#percept neurons and m=#action neurons (ideally in one-hot encoding)
        (3) During a trial, the agent chooses and returns an action in response to the current
            percept.
            This action should be in a one-hot encoding.
        Parameters:
            percept:    torch.Tensor
                        the current percept issued by the environment as a tensor
                        of torch.size([1,n]) which should be rescaled such that
                        components are real values in [0,1] for optimal performance
                        of the DEBN
            reward:     float
                        the current reward >1.e-20 issued by the environment
            done:       bool
                        True if the trial has ended, False otherwise
        Returns:
            action:     torch.Tensor
                        the action that is to be performed as a tensor of torch.size([1,m]),
                        ideally in one-hot encoding
        """

        #(1)
        if self._prev_s is not None:
            self._save_data(reward, eta_e)

        #(2)
        self._num_interactions += 1
        if self._num_interactions % self.replay_time == 0:
            self._experience_replay()

        if done:
            self._save_memory()
            self._experience_replay()

            self._prev_a = None
            self._prev_s = None

            return None
			
        #(3)
        if action is None:
            action = self._choose_action(percept, softmax_e)


        self._prev_a = action
        self._prev_s = percept

        return action

    def deliberate(self, percept, softmax_e):
        """
        This is the primary function that represents the agent's choice of an action.
         During a trial, the agent chooses and returns an action in response to the current
            percept.
            This action should be in a one-hot encoding.

        Parameters:
            percept:    torch.Tensor
                        the current percept issued by the environment as a tensor
                        of torch.size([1,n]) which should be rescaled such that
                        components are real values in [0,1] for optimal performance
                        of the DEBN

        """

        action = self._choose_action(percept, softmax_e)

        return action

    def learn(self, percept, action, reward, done):
        """
        Following is the learning process of the agent:
        (1) The reward is saved alongside the previous percept and preceding action to the trial
            data.
        (2) At the end of a trial, the accumulated data from the trial is saved to the agent's
            memory and the agent is trained. In addition, the agent is trained whenever
            X (=replay_time) number of interactions have passed since the last training.
            The network receives a tensor of torch.size([n+m]) where
            n=#percept neurons and m=#action neurons (ideally in one-hot encoding)

        Parameters:

            percept:    torch.Tensor
                        the current percept issued by the environment as a tensor
                        of torch.size([1,n]) which should be rescaled such that
                        components are real values in [0,1] for optimal performance
                        of the DEBN
            reward:     float
                        the current reward >1.e-20 issued by the environment
            done:       bool
                        True if the trial has ended, False otherwise

        Returns:

            action:     torch.Tensor
                        the action that is to be performed as a tensor of torch.size([1,m]),
                        ideally in one-hot encoding

        """
        self._prev_a = action
        self._prev_s = percept

        self._num_interactions += 1

        if self._prev_a is not None:
            self._save_buffer_data(reward)

        #check if enough new entries are in the buffer_memory before calculating glow and adding elements to the memory:
        if self._num_interactions % self.replay_time and len(self._buffer_memory) >= 2*self.replay_time:
            self._save_memory()
            self._experience_replay()

        if done:
            self._prev_a = None
            self._prev_s = None
            self._num_interactions = 0.

        return None

    def _save_buffer_data(self, reward):
        """
        saves, state, action and reward in the buffer memory
        Parameters:
             reward: reward of current epoisode

        Returns: None
        """
        data = (self._prev_s, self._prev_a, reward)
        self._buffer_memory.append(data)
        return None

    def reward_glow(self, n):
        """
        Caclulates the contirbutions of glow from future interactions and adds them to the memory so that the network can be trained on
        these episodes

        Parameters:
            n: integer set to n and then it calculates all the accumulated reward for the entries in n to self.replay_time
               in the self._buffer_memory

        Returns:
            r: (float) accumulated reward

        """
        if n == self.replay_time:
            accumulated_reward = 0.
            for i in range(self.replay_time):
                glow_contribution = self._buffer_memory[self.replay_time + i][2] * (self.gamma) ** i
                if glow_contribution > 10**(-20):
                    accumulated_reward += glow_contribution
            r = accumulated_reward
            reward_tensor = torch.Tensor([[r]])
            data = _INTERACTION(self._buffer_memory[n][0], self._buffer_memory[n][1], reward_tensor)
            self._memory.append(data)
            return r
        else:
            r = self._buffer_memory[n][2] + self.gamma * self.reward_glow(n + 1)
            reward_tensor = torch.Tensor([[r]])
            data = _INTERACTION(self._buffer_memory[n][0], self._buffer_memory[n][1], reward_tensor)
            self._memory.append(data)
            return r


    def _save_data(self, reward, eta_e):
        """
        Saves data of the current instance of the agent-environment interaction.
        The data is distributed as follows.
            (previous percept, previous action) --> _trial_data
            reward --> _trial_rewards
        gamma is taken into account for the reward.
        Parameters:
            reward: float
                    the current reward issued by the environment
        Returns:
            None
        """

        data = (self._prev_s, self._prev_a)
        self._trial_data.append(data)
        r_val = reward
        if reward != 0:
            for i in range(len(self._trial_rewards)-1, -1, -1):
                r_val = r_val*eta_e
                if r_val < 1.e-10:
                    break
                self._trial_rewards[i] += r_val
        self._trial_rewards = torch.cat((self._trial_rewards, torch.Tensor([[reward]])))

        return None


    def _save_memory(self):
        """
        Saves the data of the current trial to the agent's memory from which it is trained.
        The memory is structured as follows, (percept, action, reward).
        The _trial_data is then emptied.
        Parameters:
            None
        Returns:
            None
        """

        global _INTERACTION
        if self.episodic:
            for i, data in enumerate(self._trial_data):
                data = _INTERACTION(data[0], data[1], self._trial_rewards[i])
                self._memory.append(data)

            self._trial_data = deque()
            self._trial_rewards = torch.empty(0)

        else:
            self.reward_glow(0)

        return None


    def _experience_replay(self):
        """
        Trains the DEBN on batches of the form (s, a, r, s , a) sampled from its memory.
        (1) The policy network is copied to the target network after experience
            replay has been invoked X (=target_update) times since the last update.
        (2) A batch is sampled from the memory. The s, a, r, s and a are batched together
            so all can be trained together.
        (3) The merit values M(s,a) are predicted by the policy and target network.
        (4) The policy network is updated to predict M(s,a)+r
            (where r is already taking gamma into account).
            Therefore,
            (4.1) the loss is calculated and
            (4.2) the model is optimized with an Adam optimizer.
        Parameters:
            None
        Returns:
            None
        """

        if len(self._memory) < self.batch_size:
            return None

        #(1)
        self.target_count += 1
        if self.target_count % self.target_update == 0:
            self._target_net.load_state_dict(self.policy_net.state_dict())

        #(2)
        global _INTERACTION

        batch = random.sample(self._memory, self.batch_size)
        batch = _INTERACTION(*zip(*batch))

        batch_percept = torch.cat(batch.percept)
        batch_action = torch.cat(batch.action)
        batch_reward = torch.cat(batch.reward)
        batch_reward = batch_reward.resize_(len(batch_reward), 1)

        #(3)
        m_values = self.policy_net(batch_percept, batch_action)
        target_m_values = self._target_net(batch_percept, batch_action).detach()

        #(4)
        #(4.1)
        loss =  F.mse_loss(m_values, (target_m_values+batch_reward-self.gamma_ps*m_values.detach()))
        #(4.2)
        self._optimizer.zero_grad()
        loss.backward()

        # #Gradient clipping:
        # for param in self.policy_net.parameters():
        #     if not param.grad is None:
        #         param.grad.data.clamp_(-100, 100)

        self._optimizer.step()

        return None

    def _choose_action(self, percept, softmax_e = 1., return_prob_vals:bool = False):
        """
        Chooses an action according to the policy networks h-value predictions.

        Parameters:
            percept:    torch.Tensor
                        the current percept issued by the environment

        Returns:
            action:     torch.Tensor
                        the action that is to be performed as a tensor of torch.size([m])
        """
        m_values = np.array([])
        for action in self.all_actions:
            with torch.no_grad():
                m_val = self.policy_net(percept, action).item()
            m_values = np.append(m_values, m_val)
        #print(m_values)
        rescale = np.amax(m_values)
        prob_values = np.exp(softmax_e*(m_values-rescale))

        prob_values = prob_values/np.sum(prob_values)
        if return_prob_vals : return prob_values
        else :
            #print(m_values)
            a_pos = np.random.choice(range(len(self.all_actions)), p=prob_values)
            action = self.all_actions[a_pos]
            return action