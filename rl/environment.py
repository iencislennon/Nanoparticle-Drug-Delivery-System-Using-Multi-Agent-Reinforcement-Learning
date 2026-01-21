import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from physics.diffusion import Brownian_Motion
from enviroment.tumor import Tumor_model
from phase1_demo import diffusion_demo
from enviroment.barriers import Endothelial_Barrier
class NanoparticleEnv:
    """
        rl environment for nanoparticle navigation
    """

    def __init__(self, n_agents=1):
        self.n_agents = n_agents
        self.n_actions = 8 # 6 directions (x,-x,y,-y,z,-z) + release + wait
        # tumor model from phase 1
        self.tumor = Tumor_model(
            center=[0,0,0],
            radius=5e-3
        )

        self.brownian = Brownian_Motion(
            particle_radius=50e-9,
            viscosity=0.0035
        )

        self.dt = 0.1
        self.max_steps = 1000
        self.current_step = 0

        # state то набор параметров, описывающих 
        # текущее положение агента в среде. 
        # Оно характеризует ситуацию, 
        # в которой находится агент в данный момент
        
        # state
        self.positions = None
        self.velocities = None
        self.drug_remaining = None
        
        self.barrier_normal = Endothelial_Barrier(is_tumor=False)
        self.barrier_tumor = Endothelial_Barrier(is_tumor=True)


    def reset(self):
        """
            Returns
            observation: initial observation
        """

        # initializing positions of agents

        self.positions = np.array([[4.5e-3, 0, 0]] * self.n_agents)
        self.velocities = np.zeros((self.n_agents ,3))
        self.drug_remaining = np.ones(self.n_agents)
        self.current_step = 0

        #observation
        obs = self._get_observation()
        return obs
    
    def _get_observation(self):
        """
            observation vector for agents
        """
        obs = []

        for i in range(self.n_agents):
            pos = self.positions[i] # positions
            vel = self.velocities[i] # velocities

            # get tumor properties
            props = self.tumor.properties(pos)

            # observation vector
            agent_observation = np.concatenate(
                [
                    pos / 1e-3,
                    vel / 1e-3,
                    [props['Oxygen (mmHg)'] / 40.0]
                ]
            )

            obs.append(agent_observation)

        return np.array(obs)
    def step(self, actions):
        """
            arguments:
                actions: list of agent actions
            returns:
                observation, reward, done, info
        """
        rewards = np.zeros(self.n_agents)

        for i in range(self.n_agents):
            action = actions[i]

            if action == 6:
                props= self.tumor.properties(self.positions[i])
                    
                # release drug
                if props['is_tumor'] and self.drug_remaining[i] > 0:
                    zone = self.tumor.get_zone(self.positions[i])
                    if zone == 'necrotic':
                        rewards[i] += 50
                    elif zone == 'hypoxic':
                        rewards[i] += 100
                    elif zone == 'proliferative':
                        rewards[i] += 75
                    self.drug_remaining[i] = 0

                else:
                   rewards[i] = -10
            # wait for brownian motion
            elif action == 7:
                pass

            elif action <6:
                directions = np.array([
                    [1, 0, 0],   # +X
                    [-1, 0, 0],  # -X
                    [0, 1, 0],   # +Y
                    [0, -1, 0],  # -Y
                    [0, 0, 1],   # +Z
                    [0, 0, -1]   # -Z

                ])

                # applying small force in chosen direction
                force_magnitude = 1e-12
                self.velocities[i] += directions[action] * force_magnitude / self.brownian.mass * self.dt

            new_pos, new_vel = self.brownian.step(
                self.positions[i],
                self.velocities[i],
                self.dt
            )

            self.positions[i] = new_pos
            self.velocities[i] = new_vel

            # a little punishment for each atep
            rewards[i] -= 0.1
        self.current_step += 1

        done = self.current_step >= self.max_steps

        info = {
            'step': self.current_step,
            'positions': self.positions.copy(),
            'drug_remaining': self.drug_remaining.copy()
        }

        obs = self._get_observation()
        return obs, rewards, done, info
