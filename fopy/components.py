# TODO : implement transfer functions for commonly used optical components. 
#   e.g. lens, window, aperture etc.

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

class RectangularAperture:
    def __init__(self, 
                 lx: float, ly: float, 
                 rx: float, ry: float,
                 nx: int, ny: int):
        super().__init__()
        self.lx = lx
        self.ly = ly
        x = jnp.arange(-lx/2, lx/2, lx/nx)
        y = jnp.arange(-ly/2, ly/2, ly/ny)
        tx = np.zeros(nx)
        ty = np.zeros(ny)
        tx[np.where((x < rx) & (x > -rx))] = 1
        ty[np.where((y < ry) & (y > -ry))] = 1
        txx, tyy = jnp.meshgrid(ty, tx, indexing='ij')
        self.t = jnp.asarray(txx * tyy)

    def __call__(self, u: jnp.ndarray):
        return u * self.t

    def plot(self):
        plt.figure()
        plt.imshow(self.t, extent=[-self.lx/2, self.lx/2, -self.ly/2, self.ly/2])


class CircAperture:
    def __init__(self, 
                 lx: float, ly: float, 
                 r: float, 
                 nx: int, ny: int):
        super().__init__()
        self.lx = lx
        self.ly = ly
        x = jnp.arange(-lx/2, lx/2, lx/nx)
        y = jnp.arange(-ly/2, ly/2, ly/ny)
        yy, xx = jnp.meshgrid(y, x, indexing='ij')
        rrsq = yy**2 + xx**2
        t = np.zeros((ny, nx))
        t[rrsq <= r**2] = 1
        self.t = jnp.asarray(t)

    def __call__(self, u: jnp.ndarray):
        return u * self.t

    def plot(self):
        plt.figure()
        plt.imshow(self.t, extent=[-self.lx/2, self.lx/2, -self.ly/2, self.ly/2])


class ThinLens:
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 f: float, wl: float):
        super().__init__()
        self.lx = lx
        self.ly = ly
        x = jnp.arange(-lx/2, lx/2, lx/nx)
        y = jnp.arange(-ly/2, ly/2, ly/ny)
        yy, xx = jnp.meshgrid(y, x, indexing='ij')
        rrsq = yy**2 + xx**2
        k = 2 * jnp.pi / wl
        self.t = jnp.exp(-1j * k * rrsq / 2 / f)
        pass
    
    def __call__(self, u: jnp.ndarray):
        return u * self.t 

    def plot(self):
        pass
