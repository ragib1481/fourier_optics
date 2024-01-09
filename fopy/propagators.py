# TODO: Implement fraunhoffer propagation kernel.
# TODO: Implement adaptive sampling criterion for user convenience
# TODO: Implement general propagation for a given ABCD matrix

import jax
import jax.numpy as jnp
from matplotlib.cbook import index_of
import numpy as np
import matplotlib.pyplot as plt

@jax.jit
def qphase_exp(k, a, x, y):
    return jnp.exp(1j * k/2 * a * (x**2 + y**2))

class FresnelTf:
    r"""
        Perform propagation using the transfer function approach.
    """
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 z: float, wl: float,
                 mult: float = 1) -> None:
        k = 2 * np.pi / wl
        dx1 = lx/nx
        dy1 = ly/ny
        x = np.arange(-nx//2, nx//2) * dx1
        y = np.arange(-ny//2, ny//2) * dy1
        yy, xx = np.meshgrid(y, x, indexing='ij')
        self.q1 = qphase_exp(k, (1-mult)/z, xx, yy) / mult
        
        dfx = 1/nx/dx1
        dfy = 1/ny/dy1
        fx = np.arange(-nx//2, nx//2) * dfx
        fy = np.arange(-ny//2, ny//2) * dfy
        fyy, fxx = np.meshgrid(fy, fx, indexing='ij')
        self.q2 = qphase_exp(k, -np.pi**2*4*z/mult/k**2, fxx, fyy)

        dx2 = mult * dx1
        dy2 = mult * dy1
        x = np.arange(-nx//2, nx//2) * dx2
        y = np.arange(-ny//2, ny//2) * dy2
        yy, xx = np.meshgrid(y, x, indexing='ij')
        self.q3 = qphase_exp(k, (mult-1)/mult/z, xx, yy)

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        x = u * self.q1
        x = jnp.fft.ifftshift(x)
        x = jnp.fft.fft2(x, norm="ortho")
        x = jnp.fft.fftshift(x)
        x = self.q2 * x
        x = jnp.fft.ifftshift(x)
        x = jnp.fft.ifft2(x, norm="ortho")
        x = jnp.fft.fftshift(x)
        x = self.q3 * x
        return x

class Fresnel2:
    r"""
        Class to implement 2-step fresnel propagator. It propagates the input field
        to an intermediate field and then the field from the intermediate field to
        the observation plane. It has the added advantage of arbitrary sampling in 
        the observation plane.
    """
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 z: float, wl: float,
                 mult: float = 1) -> None:
        # calculate intermediate plane parameters
        z1 = z / (1+mult)
        z2 = z - z1

        # calculate sampling parameters
        dx1 = lx/nx
        dy1 = ly/ny

        dx2 = mult * dx1
        dy2 = mult * dy1

        dxi = wl * np.abs(z1) / (dx1 * nx)
        dyi = wl * np.abs(z1) / (dy1 * ny)

        k = 2 * np.pi / wl

        # calculate grid in the plane 1 
        x = np.arange(-nx//2, nx//2) * dx1
        y = np.arange(-ny//2, ny//2) * dy1
        yy, xx = np.meshgrid(y, x, indexing='ij')
        self.q1 = qphase_exp(k, 1/z1, xx, yy)

        # calculate grid in the intermediate plane 
        x = np.arange(-nx//2, nx//2) * dxi
        y = np.arange(-ny//2, ny//2) * dyi
        yy, xx = np.meshgrid(y, x, indexing='ij')
        self.qi1 = qphase_exp(k, 1/z1, xx, yy)

        self.qi2 = qphase_exp(k, 1/z2, xx, yy)

        # calculate grid in the plane 2
        x = np.arange(-nx//2, nx//2) * dx2
        y = np.arange(-ny//2, ny//2) * dy2
        yy, xx = np.meshgrid(y, x, indexing='ij')
        self.q2 = qphase_exp(k, 1/z2, xx, yy)

        self.z_phase1 = jnp.exp(1j*k*z1) / (1j * wl * z1)
        self.z_phase2 = jnp.exp(1j*k*z2) / (1j * wl * z2)

    def __call__(self, u: jnp.ndarray) -> jnp.ndarray:
        # plane 1 to plane i
        x = self.q1 * u
        x = jnp.fft.ifftshift(x)
        x = jnp.fft.fft2(x, norm="ortho")
        x = jnp.fft.fftshift(x)
        x = self.qi1 * x

        # plane i to plane 2
        x = self.qi2 * x
        x = jnp.fft.ifftshift(x)
        x = jnp.fft.fft2(x, norm="ortho")
        x = jnp.fft.fftshift(x)
        x = self.z_phase1 * self.z_phase2 * self.q2 * x
        return x

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

    def __call__(self, u1: jnp.ndarray) -> jnp.ndarray:
        x = self.q1 * u1
        x = jnp.fft.ifftshift(x)
        x = jnp.fft.fft2(x, norm="ortho")
        x = jnp.fft.fftshift(x)
        x = self.z_phase * self.q2 * x
        return x

