# TODO : implement transfer functions for commonly used optical components. 
#   e.g. lens, window, aperture etc.

import torch 
import numpy as np
import matplotlib.pyplot as plt

class RectangularAperture(torch.nn.Module):
    def __init__(self, 
                 lx: float, ly: float, 
                 rx: float, ry: float,
                 nx: int, ny: int):
        super().__init__()
        self.lx = lx
        self.ly = ly
        x = torch.arange(-lx/2, lx/2, lx/nx, requires_grad=False)
        y = torch.arange(-ly/2, ly/2, ly/ny, requires_grad=False)
        tx = torch.zeros(nx, requires_grad=False)
        ty = torch.zeros(ny, requires_grad=False)
        tx[torch.where((x < rx) & (x > -rx))] = 1
        ty[torch.where((y < ry) & (y > -ry))] = 1
        txx, tyy = torch.meshgrid(ty, tx, indexing='ij')
        self.t = txx * tyy

    def forward(self, u: torch.Tensor):
        return u * self.t

    def plot(self):
        plt.figure()
        plt.imshow(self.t.numpy(), \
                   extent=[-self.lx/2, self.lx/2, -self.ly/2, self.ly/2])


class CircAperture(torch.nn.Module):
    def __init__(self, 
                 lx: float, ly: float, 
                 r: float, 
                 nx: int, ny: int):
        super().__init__()
        self.lx = lx
        self.ly = ly
        x = torch.arange(-lx/2, lx/2, lx/nx, requires_grad=False)
        y = torch.arange(-ly/2, ly/2, ly/ny, requires_grad=False)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        rrsq = yy**2 + xx**2
        self.t = torch.zeros(ny, nx)
        self.t[rrsq <= r**2] = 1

    def forward(self, u: torch.Tensor):
        return u * self.t

    def plot(self):
        plt.figure()
        plt.imshow(self.t.numpy(), \
                   extent=[-self.lx/2, self.lx/2, -self.ly/2, self.ly/2])


class ThinLens(torch.nn.Module):
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 f: float, wl: float):
        super().__init__()
        self.lx = lx
        self.ly = ly
        x = torch.arange(-lx/2, lx/2, lx/nx, requires_grad=False)
        y = torch.arange(-ly/2, ly/2, ly/ny, requires_grad=False)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        rrsq = yy**2 + xx**2
        k = 2 * np.pi / wl
        self.t = torch.exp(-1j * k * rrsq / 2 / f)
        pass
    
    def forward(self, u: torch.Tensor):
        return u * self.t 

    def plot(self):
        pass
