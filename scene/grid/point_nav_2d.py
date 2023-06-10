# my custom point navigation 2d world
import numpy as np 
import pygame 
import pytest 

import gymnasium as gym 
from gymnasium import spaces

class PointNav2D(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self,
        render_mode=None):
        self.window_size = 512

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.observation_space = spaces.Box(-0.5, 0.5, shape=(4,), dtype=float)
        self.action_space = spaces.Box(-0.1, 0.1, shape=(2,), dtype=float)
        self.window = None 
        self.clock = None 

    def _get_obs(self):
        return np.array([self._agent_location[0], self._agent_location[1],
            self._target_location[0], self._target_location[1]])

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location,
                ord=2,
            )
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = np.array([0.0, 0.0])
        self._target_location = np.array([0.5, 0.5])
        
        # return the observation
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info

    def step(self, action):
        # move the agent, assure the agent is in the grid world
        self._agent_location = np.clip(
            self._agent_location + action, -0.5, 0.5
        )
        # an episode is done when the agent reaches the target
        distance = np.linalg.norm(
                        self._agent_location - self._target_location,
                        ord=2,
                    )
        terminated = distance < 0.01
        reward = -distance
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            # init pygame window
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None and self.render_mode == 'human':
            # init pygame clock
            self.clock = pygame.time.Clock()

        # draw the background
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        item_size = 10
        offset = np.array([self.window_size // 2,self.window_size // 2])
        item_offset = np.array([item_size // 2, item_size // 2])

        # draw target in red rectangle
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                self.window_size * np.array([
                    self._target_location[1],
                    -self._target_location[0]
                ]) + offset - item_offset,
                (item_size, item_size),
            )
        )

        # draw the agent in blue circle

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            self.window_size * np.array([
                    self._agent_location[1],
                    -self._agent_location[0]
                ]) + offset,
            item_size // 2,
        )

        if self.render_mode == 'human':
            # update the window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata['render_fps'])
        else: # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas), axes=(1, 0, 2)),
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def debug(self):
        return get_maze_debug(self.maze)


