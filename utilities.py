import torch
import math

def rotateVector(vector, angle):
    phi = torch.tensor(angle * math.pi / 180, dtype=torch.double, device=vector.device)
    s = torch.sin(phi)
    c = torch.cos(phi)
    rot = torch.stack([torch.stack([c, -s]),
                       torch.stack([s, c])])
    return torch.flatten(vector[None, :].double() @ rot.t().double())

def vectorDistance(x1, x2):
     return torch.cdist(x1[None, :], x2[None, :], p=2.0)

def device():
    return torch.device(
        # "cuda" if torch.cuda.is_available() else
        # "mps" if torch.backends.mps.is_available() else
        "cpu"
    )