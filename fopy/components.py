# TODO : implement transfer functions for commonly used optical components. 
#   e.g. lens, window, aperture etc.

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from . import fields

class RectangularAperture:
    def __init__(self, rx: float, ry: float):
        self.rx = rx    # lenght of aperture along the x dimension
        self.ry = ry    # length of aperture along the y dimension

    def __call__(self, u: fields.Field):
        lx, ly = u.get_dim()
        nx, ny = u.n_samples()
        x = jnp.arange(-lx/2, lx/2, lx/nx)
        y = jnp.arange(-ly/2, ly/2, ly/ny)
        tx = np.zeros(nx)
        ty = np.zeros(ny)
        tx[np.where((x < self.rx/2) & (x > -self.rx/2))] = 1
        ty[np.where((y < self.ry/2) & (y > -self.ry/2))] = 1
        txx, tyy = np.meshgrid(ty, tx, indexing='ij')
        t = jnp.asarray(txx * tyy)
        return u * t

class CircAperture:
    def __init__(self, r: float):
        super().__init__()
        self.r = r      # aperture radius

    def __call__(self, u: fields.Field):
        lx, ly = u.get_dim()
        nx, ny = u.n_samples()
        x = jnp.arange(-lx/2, lx/2, lx/nx)
        y = jnp.arange(-ly/2, ly/2, ly/ny)
        yy, xx = jnp.meshgrid(y, x, indexing='ij')
        rrsq = yy**2 + xx**2
        t = np.zeros((ny, nx))
        t[rrsq <= self.r**2] = 1
        t = jnp.asarray(t)
        return u * t

class ThinLens:
    def __init__(self, r: float, zf: float):
        self.r = r          # aperture radius of the lens
        self.zf = zf        # focal length
    
    def __call__(self, u: fields.Field, plot=False) -> fields.Field:

        wl = u.get_wl()
        k = 2 * jnp.pi / wl

        lx, ly = u.get_dim()
        nx, ny = u.n_samples()
        dx, dy = u.get_sampling()

        x = jnp.arange(-nx//2, nx//2) * dx
        y = jnp.arange(-ny//2, ny//2) * dy
        yy, xx = jnp.meshgrid(y, x, indexing='ij')
        rrsq = yy**2 + xx**2

        # define aperture of the lens
        ta = np.zeros((ny, nx))
        ta[rrsq <= self.r**2] = 1
        ta = jnp.asarray(ta)

        # define phase trasnfer function of the lens
        tf = jnp.exp(-1j * (k/2/self.zf) * rrsq)

        if plot:
            plt.figure()
            plt.subplot(2,2,1)
            plt.imshow(np.abs(tf), extent=(-lx/2, lx/2, ly/2, -ly/2))

            plt.subplot(2,2,2)
            plt.imshow(np.angle(tf), extent=(-lx/2, lx/2, ly/2, -ly/2))

            plt.subplot(2,2,3)
            plt.imshow(np.abs(ta),extent=(-lx/2, lx/2, ly/2, -ly/2) )

            plt.subplot(2,2,4)
            plt.imshow(np.angle(ta), extent=(-lx/2, lx/2, ly/2, -ly/2))
        return u * ta * tf

    def plot(self):
        pass
