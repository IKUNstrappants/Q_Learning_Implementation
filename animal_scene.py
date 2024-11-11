from zoo import hunter, prey, omega_predator
import pygame
import numpy as np
from utilities import *
class grassland():
    def __init__(self, num_hunter=1, num_prey=10, num_OmegaPredator=5, size=100, hunter_n_action=25, prey_n_action=1):
        self.hunters = {}
        self.preys = {}
        self.OmegaPredators = {}
        self.size=size
        self.window = None
        self.render_mode = "human"
        self.clock = None
        self.window_size = 1024  # The size of the PyGame window
        for i in range(num_hunter):
            self.hunters[i] = hunter(field=self, id=i, action_space=hunter_n_action)
        for i in range(num_prey):
            self.preys[i+num_hunter+10] = prey(field=self, id=i+num_hunter+10, action_space=prey_n_action)
        for i in range(num_OmegaPredator):
            self.OmegaPredators[i+num_hunter+10+num_prey+10] = omega_predator(field=self, id=i+num_hunter+10+num_prey+10)

    def _get_all_entities(self):
        return list(self.hunters.values()) + list(self.preys.values()) + list(self.OmegaPredators.values())

    def _update_possessed_entities(self):
        for entity in self._get_all_entities():
            if entity.possessed:
                entity.walk()

    def reset(self):
        for hunterID in self.hunters.keys():
            self.hunters[hunterID].reset()
        for preyID in self.preys.keys():
            self.preys[preyID].reset()
        for OmegaPredatorID in self.OmegaPredators.keys():
            self.OmegaPredators[OmegaPredatorID].reset()

    def _render_entity(self, entity, canvas, pix_square_size):
        screen_position = pix_square_size * (entity.location.numpy() + self.size) / 2
        screen_coord = lambda x: pix_square_size * (x + self.size) / 2
        color_code = {
            0: np.array([0, 0, 0]),     # barrier
            1: np.array([255, 0, 0]),   # predator
            2: np.array([0, 255, 0]),   # prey
            4: np.array([255, 255, 0]),   # omega_predator
        }
        # color = color_code[entity.type * 10 + (1 if entity.alive else 0)]
        pygame.draw.circle(surface=canvas, color=color_code[entity.type], center=screen_coord(entity.location.numpy()), radius=7, width=0 if entity.alive else 4)
        if entity.view_cache is not None:
            for i in range(len(entity.ray[0])):
                angle  = entity.ray[0][i]
                length = entity.ray[1][i]
                direction = rotateVector(entity.forward, angle)
                # print(entity.view_cache[i, :])
                type, distance = tuple(entity.view_cache[i, :])
                distance = 1.0 / distance
                ray_color = color_code[type.item()]
                pygame.draw.line(
                    canvas,
                    color_code[entity.type] * 0.5,
                    screen_coord(entity.location.numpy()),
                    screen_coord((entity.location + direction * length).numpy()),
                    width=3,
                )
                if type != 0:
                    pygame.draw.line(
                        canvas,
                        ray_color,
                        screen_coord(entity.location.numpy()),
                        screen_coord((entity.location + direction * distance.item()).numpy()),
                        width=3,
                    )

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = self.window_size / self.size # The size of a single grid square in pixels

        for entity in self._get_all_entities():
            self._render_entity(entity, canvas, pix_square_size)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            # self.clock.tick(3)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        return
        if self.window is not None:
            pygame.display.quit()
            self.window = None
