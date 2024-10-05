import random
from ideas import *
from utilities import *
from gym_examples_main.gym_examples.envs import AnimalEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = torch.rand(256, 2)

class animal():
    def __init__(self, field, AI=IdleAI, id=0, ray=None,
                 location=torch.zeros(2),
                 forward=torch.tensor([1.,0.], dtype=torch.double),
                 possess=False):

        self.location = location.double()
        self.forward = forward.double()
        self.brain = Brain(AI = AI)
        self.turn = 0
        self.move = 0
        self.ray = ray
        self.field = field
        self.size = 0
        self.type = 0
        self.id = id
        self.score = 0
        self.possessed = possess
        self.lifetime = 0
        self.alive = True
        self.perception = AnimalEnv(animal=self)
        self.view_cache = None
        self._reset_upon_death = False

    def reset(self):
        pass

    def cheatWalk(self):
        targets = list(self.field.hunters.values()) + list(self.field.preys.values())
        distance = 99.0
        closest_ = targets[0]
        for target in targets:
            if target == self or target.alive==False:
                pass
            target_to_self = self.location - target.location
            dist = vectorDistance(target_to_self, 0)
            if target.type==1 and self.type==2:
                if dist <= distance:
                    distance = dist
                    closest_ = target
            elif target.type==2 and self.type==1:
                if dist <= distance:
                    distance = dist
                    closest_ = target
        if distance<99.:
            dist_vec = self.location - closest_.location
            self.location += dist_vec / vectorDistance(dist_vec, 0) * (1 if self.type==2 else -1)

    def vision(self, angle, length, targets):
        # print("get vision")
        ray = rotateVector(self.forward, angle)
        type, distance = 0., 10000.
        for target in targets:
            dist = vectorDistance(target.location, self.location)
            if torch.dot(target.location-self.location, ray)<=target.size and dist <= length and dist < distance:
                type = target.type
                distance = dist
        return torch.tensor([[type, 1.0 / distance]]).to(device)

    def view(self):
        targets = self.field._get_all_entities()
        available = []
        for target in targets:
            if target.id == self.id or target.alive==False:
                continue
            # print(target, target.location)
            dist = vectorDistance(target.location, self.location)
            if 1. < dist < 21.:
                available.append(target)
        view_cache = []
        for index in range(len(self.ray[0])):
            view_cache.append(self.vision(self.ray[0][index], self.ray[1][index], available))
        self.view_cache = torch.cat(view_cache)
        return self.view_cache

    def walk(self, movement, action):
        size = self.field.size
        # print("begin walk")
        self.score -= self.perception._action_to_penalty[action]
        if not self.possessed:
            move = movement[0] * 5.
            turn = movement[1] * 15.
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
    def __init__(self, field, AI=PredatorAI, id=0):
        super().__init__(field, AI, id=id,
                         ray=[[-30, -15, -5,  0,  5, 15, 30],
                              [ 10,  15, 20, 20, 20, 15, 10]],
                         possess=False)
        self.size = 4.
        self.type = 1
        self._reset_upon_death = False

    def walk(self, movement, action):
        super().walk(movement, action)
        self.score -= 0.00

    def reset(self):
        size = self.field.size
        self.turn = 0
        self.move = 0
        location, forward = torch.rand(2, dtype=torch.double)*size*2 - size, rotateVector(torch.tensor([1., 0]), random.random() * 360)
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
        # if kills>0: print(f"kills = {kills}")
        self.score += kills
        return kills


class prey(animal):
    def __init__(self, field, AI=PreyAI, id=0):
        super().__init__(field, AI, id=id,
                         ray=[[-70, -30, -15,  0,  15, 30, 70],
                              [  7,   7,   7,  7,   7,  7,  7]],
                         possess=True,
                         )
        self.size = 3.
        self.type = 2
        self._reset_upon_death = True

    def reset(self):
        size = self.field.size
        self.turn = 0
        self.move = 0
        location, forward = torch.rand(2, dtype=torch.double)*size*2 - size, rotateVector(torch.tensor([1., 0]), random.random() * 360)
        self.location = location
        self.forward = forward
        self.score = 0
        self.alive = True

    def walk(self, movement, action):
        super().walk(movement, action)
        self.score += 0.02

class barrier(animal):
    def __init__(self, field, AI=IdleAI, id=0):
        super().__init__(field, AI, id=id,
                         ray=[])
        self.size = 3.
        self.type = 3.

