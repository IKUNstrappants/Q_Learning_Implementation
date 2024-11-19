import random

import torch

from ideas import *
from ideas import IdleAI
from utilities import *
from gym_examples_main.gym_examples.envs import AnimalEnv, BridgeEnv
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points


x = torch.rand(256, 2)

class animal():
    def __init__(self, field, AI=IdleAI, id=0, ray=None,
                 location=torch.zeros(2, device=device(), dtype=torch.double),
                 forward=torch.tensor([1.,0.], dtype=torch.double, device=device()),
                 possess=False,
                 action_space=25):

        self.location = location.double()
        self.forward = forward.double()
        self.brain = Brain(AI = AI)
        self.turn = 0
        self.move = 0
        self.ray = ray
        self.field = field
        self.size = 0
        # self.type values
        # 1: hunter   2: prey     3: barrier      4: omega_predator
        self.type = 0
        self.id = id
        self.score = 0
        self.possessed = possess
        self.lifetime = 0
        self.alive = True
        self.perception = AnimalEnv(animal=self, action_space=action_space)
        self.view_cache = None
        self._reset_upon_death = False

    def reset(self):
        pass

    def cheatWalk(self):
        targets = []
        if self.type == 4:
            targets = list(self.field.hunters.values())
        elif self.type == 1:
            targets = list(self.field.preys.values())
        elif self.type == 2:
            return
        distance = 40.0
        # print(self.type)
        closest_ = targets[0]
        for target in targets:
            if target == self or target.alive==False:
                pass
            dist = vectorDistance(self.location, target.location)
            if self.type == 1 or self.type == 4:
                if dist <= distance:
                    distance = dist
                    closest_ = target
        if distance < 30.:
            dist_vec = self.location - closest_.location
            self.location += (dist_vec / vectorDistance(self.location, closest_.location) * (-1 if (self.type==1 or self.type==4) else 1)).flatten()

    def vision(self, angle, length, targets):
        # print("get vision")
        ray = rotateVector(self.forward, angle)
        type, distance = 0., 10000.
        for target in targets:
            dist = vectorDistance(target.location, self.location)
            if 0 <= torch.dot(target.location-self.location, ray) <= target.size and dist <= length and dist < distance:
                type = target.type
                distance = dist
        distance = 1.0 / (distance + 1)
        return torch.tensor([[type, distance]]).to(device())

    def view(self):
        targets = self.field._get_all_entities()
        available = []
        for target in targets:
            if target.id == self.id or target.alive==False:
                continue
            # print(target, target.location)
            dist = vectorDistance(target.location, self.location)
            if dist < 21.:
                available.append(target)
        view_cache = []
        for index in range(len(self.ray[0])):
            view_cache.append(self.vision(self.ray[0][index], self.ray[1][index], available))
        self.view_cache = torch.cat(view_cache).to(torch.float32)
        return self.view_cache

    def walk(self, movement=None, action=None):
        if not self.possessed:
            size = self.field.size
            move = movement[0] * 5.
            turn = movement[1] * 15.
            self.forward = rotateVector(self.forward, turn)
            self.location = (self.location + self.forward * move + size) % (size*2) - size
        else:
            self.cheatWalk()
    
    #new add
    def continuous_walk(self, action):
        size = self.field.size
        if not self.possessed:
            move = action[0].item()
            turn = action[1].item()
            self.forward = rotateVector(self.forward, turn)
            self.location = (self.location + self.forward * move + size) % (size*2) - size
        else:
            self.cheatWalk()

    def die(self):
        self.alive = False
        self.score -= 10
        if self._reset_upon_death:
            self.reset()

class hunter(animal):
    def __init__(self, field, AI=PredatorAI, id=0, action_space=25):
        super().__init__(field, AI, id=id,
                         ray=[[-150, -30, -15, -5,  0,  5, 15, 30, 150, 180],
                              [  15,  10,  15, 20, 20, 20, 15, 10,  15,  20]],
                         possess=False, action_space=action_space)
        self.size = 1.
        self.type = 1
        self._reset_upon_death = False

    def walk(self, movement=None, action=None):
        super().walk(movement, action)

    def reset(self):
        size = self.field.size
        self.turn = 0
        self.move = 0
        location, forward = (torch.rand(2, dtype=torch.double, device=device())*size*2 - size,
                             rotateVector(torch.tensor([1., 0], device=device()), random.random() * 360))
        self.location = location
        self.forward = forward
        self.score = 0
        self.alive = True

    def hunt(self):
        targets = self.field.preys.values()
        kills = 0.
        for target in targets:
            if not target.alive or target == self:
                pass
            else:
                dist = vectorDistance(target.location, self.location)
                if dist < 3:
                    target.die()
                    kills += 1.
        return kills

class omega_predator(animal):
    def __init__(self, field, id):
        super().__init__(field=field, possess=True, id=id)
        self.type = 4
        self.size = 5

    def reset(self):
        size = self.field.size
        self.location = torch.rand(2, dtype=torch.double, device=device())*size*2 - size

class prey(animal):
    def __init__(self, field, AI=PreyAI, id=0, action_space=25):
        super().__init__(field, AI, id=id,
                         ray=[[-70, -30, -15,  0,  15, 30, 70],
                              [  7,   7,   7,  7,   7,  7,  7]],
                         possess=True,
                         action_space=action_space)
        self.size = 3.
        self.type = 2
        self._reset_upon_death = True

    def reset(self):
        size = self.field.size
        self.turn = 0
        self.move = 0
        location, forward = (torch.rand(2, dtype=torch.double, device=device())*size*2 - size,
                             rotateVector(torch.tensor([1., 0], device=device()), random.random() * 360))
        self.location = location
        self.forward = forward
        self.score = 0
        self.alive = True

    def walk(self, movement=None, action=None):
        super().walk()
        self.score += 0.02

class barrier(animal):
    def __init__(self, field, AI=IdleAI, id=0):
        super().__init__(field, AI, id=id,
                         ray=[])
        self.size = 3.
        self.type = 3.

class walker():
    def __init__(self, bridge, AI=IdleAI, id=0,
                 ray=None,
                 location=torch.zeros(2),
                 forward=torch.tensor([1., 0.], dtype=torch.double),
                 possess=False):
        self.location = location.double()
        self.forward = forward.double()
        self.brain = Brain(AI=AI)
        self.turn = 0
        self.move = 0
        self.ray = ray
        self.bridge = bridge
        self.size = 0
        self.type = 0
        self.id = id
        self.score = 0
        self.possessed = possess
        self.lifetime = 0
        self.alive = True
        self.perception = BridgeEnv(agent=self)
        self.view_cache = None
        self._reset_upon_death = False

    def reset(self):
        pass

    def tp(self, location):
        self.location = location
        self.forward = torch.tensor([1, 0])

    def vision(self, angle, length):
        ray_direction = rotateVector(self.forward, angle)
        ray_start = Point(self.location)
        ray = LineString([ray_start, ray_direction])

        # Find the nearest point on the polygon to the ray
        nearest_point = nearest_points(ray, self.perception.surface.exterior)[1]

        # Calculate the distance from the ray start to the nearest point on the polygon
        distance = min(ray_start.distance(nearest_point), length)

        return torch.tensor([1.0 / distance]).to(device())

    def view(self):
        view_cache = []
        for index in range(len(self.ray[0])):
            view_cache.append(self.vision(self.ray[0][index], self.ray[1][index]))
        self.view_cache = torch.cat(view_cache)
        return self.view_cache

    def walk(self, action):
        # print("begin walk")
        self.score -= self.perception._action_to_penalty[action]
        movement = self.perception._action_to_direction[action]
        move = movement[0]
        turn = movement[1]
        self.forward = rotateVector(self.forward, turn)
        self.location = min(max(self.location + self.forward * move, 0), self.perception.bridge_length)



if __name__ == "__main__":
    pass