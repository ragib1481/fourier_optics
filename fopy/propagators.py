# TODO: Implement fraunhoffer propagation kernel.
# TODO: Implement adaptive sampling criterion for user convenience
# TODO: Implement general propagation for a given ABCD matrix

from typing import Type
import jax
import jax.numpy as jnp
from matplotlib.cbook import index_of
import numpy as np
import matplotlib.pyplot as plt
from . import fields
from .fields import Field

class Fresnel:
    def __init__(self, z) -> None:
        self.z = z
        self.propIR = FresnelIR(z)
        self.propTF = FresnelTF(z)

    def __call__(self, u1: fields.Field) -> fields.Field:
        '''
            check which method to use based on the simulation criteria. 
            This is based on the discussion from the book:
            Voelz, D.G., 2011. Computational fourier optics: a MATLAB tutorial. 
        '''
        wl = u1.get_wl()
        k = 2 * jnp.pi / wl
        dx, dy = u1.get_sampling()
        lx, ly = u1.get_dim()

        # check the validity of the method using sampling parameters. here 2*lx is 
        # used since for the simulation the support is zero padded to double the length.
        if dx >= wl * self.z / (2*lx):
            self.prop = FresnelTF(self.z)
        else:
            self.prop = FresnelIR(self.z)
        return self.prop(u1)

class FresnelTF:
    def __init__(self, z: float) -> None:
        self.z = z

    def __call__(self, u1: fields.Field) -> fields.Field:
        # get information about the field
        wl = u1.get_wl()
        k = 2 * jnp.pi / wl
        dx, dy = u1.get_sampling()
        lx, ly = u1.get_dim()
        nx, ny = u1.n_samples()

        # padd the field to provide support for accurate simulation
        u1_padded = jnp.pad(u1.get_array(), ((ny//2, ny//2), (nx//2, nx//2)))
        nx, ny = u1_padded.shape

        # compute the transfer function for fresnel propagation
        fx = np.fft.fftfreq(u1_padded.shape[1], dx)
        fy = np.fft.fftfreq(u1_padded.shape[0], dy)
        fyy, fxx = np.meshgrid(fy, fx, indexing='ij')
        H = jnp.exp(1j*k*self.z) * \
            jnp.exp(-1j * jnp.pi * wl * self.z * (fxx**2 + fyy**2))

        # perform propagation
        u2 = jnp.fft.ifftshift(u1_padded)
        u2 = jnp.fft.fft2(u2)
        u2 = jnp.fft.fftshift(jnp.fft.ifft2(u2 * H))

        # get the middle slice of the field to keep the array dimension manageable
        u2 = u2[ny//4:3*(ny//4), nx//4:3*(nx//4)]

        u2_field = Field(wavelength=wl, 
                         nx=nx//2, ny=ny//2, 
                         lx=lx, ly=ly,
                         array=u2)

        # check the validity of the method using sampling parameters. here 2*lx is 
        # used since for the simulation the support was zero padded to double the length.
        if np.abs(dx >= wl * self.z / (2*lx)):
            u2_field.set_valid(True)
        else:
            u2_field.set_valid(False)
        return u2_field

class FresnelIR:
    r"""
        Class implementing the fresnel propagator using the fresnel
        impulse response method. It is implmented as:
            $$ U_2(x_2,y_2) = \mathcal{F}^{-1}[\mathcal{F}[U_1(x,y)]
                                               \mathcal{F}[h(x,y)]]
            $$
            and, 
            $$ h(x,y) = \frac{e^{jkz}}{j\lambda z}e^{\frac{jk}{2z}(x^2+y^2) $$
        
        Simulation Steps:
            1. Take the input field.
            2. Zeros pad the input field so that length of the support is, L = 2*D.
                Here, D is the input field length.
            3. From the value of L, $\lambda$ and z determine the sampling parameter.
    """

    def __init__(self, z: float,) -> None:
        self.z = z

    def __call__(self, u1: fields.Field) -> fields.Field:
        # get information about the field
        wl = u1.get_wl()
        k = 2 * jnp.pi / wl
        dx, dy = u1.get_sampling()
        lx, ly = u1.get_dim()
        nx, ny = u1.n_samples()

        # padd the field to provide support for accurate simulation
        u1_padded = jnp.pad(u1.get_array(), ((ny//2, ny//2), (nx//2, nx//2)))
        nx, ny = u1_padded.shape

        # compute the impulse response for fresnel propagation
        x = np.arange(-nx//2, nx//2) * dx
        y = np.arange(-ny//2, ny//2) * dy
        yy, xx = np.meshgrid(y, x, indexing='ij')
        h = jnp.exp(1j * k * self.z) / (1j * wl * self.z)
        h = h * jnp.exp( (1j*k/2/self.z) * (xx**2 + yy**2) )

        # perform propagation
        u2 = jnp.fft.fft2(jnp.fft.ifftshift(u1_padded)) * \
            jnp.fft.fft2(jnp.fft.ifftshift(h))

        u2 = jnp.fft.fftshift(jnp.fft.ifft2(u2))

        # get the middle slice of the field to keep the array dimension manageable
        u2 = u2[ny//4:3*(ny//4), nx//4:3*(nx//4)]

        u2_field = Field(wavelength=wl, 
                         nx=nx//2, ny=ny//2, 
                         lx=lx, ly=ly,
                         array=u2)

        # check the validity of the method using sampling parameters. here 2*lx is 
        # used since for the simulation the support was zero padded to double the length.
        if dx < np.abs(wl * self.z / (2*lx)):
            u2_field.set_valid(True)
        else:
            u2_field.set_valid(False)
        return u2_field
