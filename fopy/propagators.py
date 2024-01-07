# TODO:Implement fresnel propagation kernel.
# TODO:Implement fraunhoffer propagation kernel.

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt


class Fresnel:
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 z: float, wl: float) -> None:

        super().__init__()
        dx = lx/nx 

        if dx >= (wl * z / lx):
            self.model = FresnelTf(lx, ly, nx, ny, z, wl)
        else:
            self.model = FresnelIr(lx, ly, nx, ny, z, wl)

    def __call__(self, u: jnp.ndarray):
        return self.model(u)

    def name(self):
        return self.model.name()

    def plot(self):
        self.model.plot()


class FresnelIr:
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 z: float, wl: float) -> None:

        super().__init__()
        x = jnp.arange(-lx/2, lx/2, lx/nx)
        y = jnp.arange(-ly/2, ly/2, ly/ny)
        yy, xx = jnp.meshgrid(y, x, indexing="ij")
        k = 2 * np.pi / wl
        h = np.exp(1j * k * z) / (1j * wl * z)
        self.h = h * jnp.exp(1j * k / 2 / z * (xx**2 + yy**2))

    def __call__(self, u: jnp.ndarray):
        x = jnp.fft.fft2(jnp.fft.ifftshift(u), norm="ortho")
        x = x * jnp.fft.fft2(jnp.fft.ifftshift(self.h))
        x = jnp.fft.ifftshift(jnp.fft.ifft2(x, norm="ortho")) 
        return x

    def name(self):
        return "Fresnel IR propagator"

    def plot(self):
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(jnp.abs(self.h))

        plt.subplot(2, 2, 3)
        plt.imshow(jnp.real(self.h))

        plt.subplot(2, 2, 4)
        plt.imshow(jnp.imag(self.h))


class FresnelTf:
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 z: float, wl: float) -> None:

        super().__init__()
        fx = jnp.fft.fftfreq(nx, d=lx/nx)
        fy = jnp.fft.fftfreq(ny, d=ly/ny)
        fx = jnp.fft.fftshift(fx)
        fy = jnp.fft.fftshift(fy)
        fyy, fxx = jnp.meshgrid(fy, fx, indexing='ij')
        k = 2 * np.pi / wl
        self.H = jnp.exp(-1j * jnp.pi * wl * z * (fxx**2 + fyy**2))
        self.H = self.H * jnp.asarray(1j * k * z)
        self.H = jnp.fft.ifftshift(self.H)

    def __call__(self, u: jnp.ndarray):
        x = jnp.fft.fft2(jnp.fft.ifftshift(u), norm="ortho")
        x = x * self.H
        x = jnp.fft.ifftshift(jnp.fft.ifft2(x, norm="ortho")) 
        return x

    def name(self):
        return "Fresnel Tf propagator"

    def plot(self):
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(jnp.abs(jnp.fft.ifftshift(self.H)))

        plt.subplot(2, 2, 3)
        plt.imshow(jnp.real(jnp.fft.ifftshift(self.H)))

        plt.subplot(2, 2, 4)
        plt.imshow(jnp.imag(jnp.fft.ifftshift(self.H)))

