import gym
import numpy as np
# https://www.youtube.com/watch?v=6Bz7SdMahFg&ab_channel=DibyaChakravorty
# this is for an inventory management problem

class InventoryEnv(gym.Env):

    def __init__(self):
        ''' Must define an self.obs_space and self.action_space here'''
        #Define action space : bounds, space type and shape
        #Bound : shelf space is limited
        self.max_cap = 4000
        #Space type : Better to use Box than Discrete since Discrete will lead to too many output nodes in the NN 
        # Shape : rllib cannot handle scalar actions, so turn it into a numpy array
        self.action_space = gym.spaces.Box(low=np.array([0]), high=np.array([self.max_cap]))
        #Observation Space (state space) : bounds, space type and shape
        
        #
        pass

    def reset(self):
        '''Returns : observation of the initial state
            Reset the env to initial state so that the new episode can be started 
        '''
        #

        pass

    def step(self):
        '''Returns : the next obs, reward, done and optionally additional info'''
        pass

    def render(self):
        '''
        Returns none 
        renders the premade environments provided by openai gym library
        '''
        pass

    def close(self):
        '''Returns none
            This method is optional and used to clean up all the resources
        '''

        pass
    
    def seed(self):
        '''Returns list of seeds 
        used to set seed for the random number generator for obtaining deterministic behavior
        '''
        
        pass