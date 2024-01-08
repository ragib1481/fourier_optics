# TODO:Implement fraunhoffer propagation kernel.
# TODO: Implement adaptive sampling criterion for user convenience

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

@jax.jit
def qphase_exp(k, a, x, y):
    return jnp.exp(1j * k/2 * a * (x**2 + y**2))

# TODO: Implement this class for fresnel transfer function approach 
class FresnelTf:
    def __init__(self) -> None:
        pass

    def __call__(self):
        pass

# TODO: Implement this class for fresnel 2 step propagator 
class Fresnel2:
    def __init__(self) -> None:
        pass

    def __call__(self, u: jnp.ndarray):
        pass

class Fresnel:
    r"""
        Class implementing the fresnel propagator using the fresnel
        transform (fresnel-integral) method. The fresnel-integral
        is implmented as:
            $$ U_2(x_2,y_2) = \frac{e^{jkz}}{j\lambda z} 
                            e^{\frac{jk}{2z}(x_2^2+y_2^2)}
                            \mathcal{F}[U_1(x_1,y_1)e^{\frac{jk}{2z}(x_1^2+y_1^2)}]
            $$

        For this method the sampling in the observation plane is estimated from the 
        given parameters. We assume the lengths of the source plane and the 
        observation planes are the same.

        For a more flexible approach that allows for arbitrary 
        sampling in the observation plane Fresnel2 should be used. 
    """

    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 z: float, wl: float) -> None:
        dx1 = lx/nx
        dy1 = ly/ny
        dx2 = wl * z / (nx * dx1)
        dy2= wl * z / (ny * dy1)
        k = 2 * np.pi / wl

        x1 = np.arange(-nx//2, nx//2) * dx1
        y1 = np.arange(-ny//2, ny//2) * dy1
        x2 = np.arange(-nx//2, nx//2) * dx2
        y2 = np.arange(-ny//2, ny//2) * dy2

        yy, xx = np.meshgrid(y1, x1, indexing='ij')
        self.q1 = qphase_exp(k, 1/z, xx, yy)

        yy, xx = np.meshgrid(y2, x2, indexing='ij')
        self.q2 = qphase_exp(k, 1/z, xx, yy)

        self.z_phase = np.exp(1j*k*z) / (1j*wl*z)

    def __call__(self, u1: jnp.ndarray):
        x = self.q1 * u1
        x = jnp.fft.ifftshift(x)
        x = jnp.fft.fft2(x, norm="ortho")
        x = jnp.fft.fftshift(x)
        x = self.z_phase * self.q2 * x
        return x

# class Fresnel:
#     def __init__(self, 
#                  lx: float, ly: float, 
#                  nx: int, ny: int,
#                  z: float, wl: float) -> None:
#
#         super().__init__()
#         dx = lx/nx 
#
#         if dx >= (wl * z / lx):
#             self.model = FresnelTf(lx, ly, nx, ny, z, wl)
#         else:
#             self.model = FresnelIr(lx, ly, nx, ny, z, wl)
#
#     def __call__(self, u: jnp.ndarray):
#         return self.model(u)
#
#     def name(self):
#         return self.model.name()
#
#     def plot(self):
#         self.model.plot()
#
#
# class FresnelIr:
#     def __init__(self, 
#                  lx: float, ly: float, 
#                  nx: int, ny: int,
#                  z: float, wl: float) -> None:
#
#         super().__init__()
#         x = jnp.arange(-lx/2, lx/2, lx/nx)
#         y = jnp.arange(-ly/2, ly/2, ly/ny)
#         yy, xx = jnp.meshgrid(y, x, indexing="ij")
#         k = 2 * np.pi / wl
#         h = np.exp(1j * k * z) / (1j * wl * z)
#         self.h = h * jnp.exp(1j * k / 2 / z * (xx**2 + yy**2))
#
#     def __call__(self, u: jnp.ndarray):
#         x = jnp.fft.fft2(jnp.fft.ifftshift(u), norm="ortho")
#         x = x * jnp.fft.fft2(jnp.fft.ifftshift(self.h))
#         x = jnp.fft.ifftshift(jnp.fft.ifft2(x, norm="ortho")) 
#         return x
#
#     def name(self):
#         return "Fresnel IR propagator"
#
#     def plot(self):
#         plt.figure()
#         plt.subplot(2, 2, 1)
#         plt.imshow(jnp.abs(self.h))
#
#         plt.subplot(2, 2, 3)
#         plt.imshow(jnp.real(self.h))
#
#         plt.subplot(2, 2, 4)
#         plt.imshow(jnp.imag(self.h))
#
#
# class FresnelTf:
#     def __init__(self, 
#                  lx: float, ly: float, 
#                  nx: int, ny: int,
#                  z: float, wl: float) -> None:
#
#         super().__init__()
#         fx = jnp.fft.fftfreq(nx, d=lx/nx)
#         fy = jnp.fft.fftfreq(ny, d=ly/ny)
#         fx = jnp.fft.fftshift(fx)
#         fy = jnp.fft.fftshift(fy)
#         fyy, fxx = jnp.meshgrid(fy, fx, indexing='ij')
#         k = 2 * np.pi / wl
#         self.H = jnp.exp(-1j * jnp.pi * wl * z * (fxx**2 + fyy**2))
#         self.H = self.H * jnp.asarray(1j * k * z)
#         self.H = jnp.fft.ifftshift(self.H)
#
#     def __call__(self, u: jnp.ndarray):
#         x = jnp.fft.fft2(jnp.fft.ifftshift(u), norm="ortho")
#         x = x * self.H
#         x = jnp.fft.ifftshift(jnp.fft.ifft2(x, norm="ortho")) 
#         return x
#
#     def name(self):
#         return "Fresnel Tf propagator"
#
#     def plot(self):
#         plt.figure()
#         plt.subplot(2, 2, 1)
#         plt.imshow(jnp.abs(jnp.fft.ifftshift(self.H)))
#
#         plt.subplot(2, 2, 3)
#         plt.imshow(jnp.real(jnp.fft.ifftshift(self.H)))
#
#         plt.subplot(2, 2, 4)
#         plt.imshow(jnp.imag(jnp.fft.ifftshift(self.H)))

