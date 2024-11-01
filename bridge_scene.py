import pygame
import numpy as np
from utilities import *
from zoo import walker

class bridge_scene():
    def __init__(self, size=100):
        self.size=size
        self.window = None
        self.render_mode = "human"
        self.clock = None
        self.agent = walker(self, ray=[[-30, -15, -5,  0,  5, 15, 30],
                                       [ 10,  10, 10, 10, 10, 10, 10]])
        self.window_size = 512  # The size of the PyGame window

    def reset(self):
        self.agent.reset()

    def _render_entity(self, entity, canvas, pix_square_size):
        screen_position = pix_square_size * (entity.location.numpy() + self.size) / 2
        color_code = {
            0: np.array([0, 0, 0]),     # barrier
            1: np.array([255, 0, 0]),   # predator
            2: np.array([0, 255, 0]),   # prey
        }
        # color = color_code[entity.type * 10 + (1 if entity.alive else 0)]
        pygame.draw.circle(surface=canvas, color=np.array([0, 255, 0]), center=screen_position, radius=7, width=0)
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
                    np.array([0, 255, 0]) * 0.5,
                    screen_position,
                    pix_square_size * ((entity.location + direction * length * 2).numpy() + self.size) / 2,
                    width=3,
                )
                if type != 0:
                    pygame.draw.line(
                        canvas,
                        ray_color,
                        screen_position,
                        pix_square_size * ((entity.location + direction * distance.item() * 2).numpy() + self.size) / 2,
                        width=3,
                    )

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((0, 0, 0))
        pix_square_size = self.window_size / self.size # The size of a single grid square in pixels

        points = np.array(list(zip(self.agent.perception.surface.exterior.xy))).reshape(2, -1).T
        # print(points, points.shape)
        # print(pix_square_size * (points[ 0 ] + self.size) / 2)
        # print(points[0])
        for i in range(points.shape[0]-1):
            pygame.draw.line(
                canvas,
                np.array([255, 0, 0]),
                pix_square_size * (points[ i ]) + np.array([0, self.window_size / 2]),
                pix_square_size * (points[i+1]) + np.array([0, self.window_size / 2]),
                width=3,
            )
        self._render_entity(self.agent, canvas, pix_square_size)

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
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

if __name__ == "__main__":
    bs = bridge_scene()
    while True:
        bs._render_frame()