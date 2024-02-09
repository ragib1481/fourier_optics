import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

class Field:
    def __init__(self, wavelength, nx, ny, lx, ly, array=None) -> None:
        self.wl = wavelength
        if array is None:
            self.array = jnp.zeros((ny, nx))
        else:
            self.array = array
        self.lx = lx
        self.ly = ly
        self.nx = nx
        self.ny = ny
        self.valid_flag = True

    def is_valid(self):
        return self.valid_flag

    def set_valid(self, flag):
        self.valid_flag = flag

    def get_dim(self):
        return self.lx, self.ly

    def get_wl(self):
        return self.wl

    def n_samples(self):
        return self.nx, self.ny

    def get_array(self):
        return self.array

    def __mul__(self, u):
        field = Field(wavelength=self.wl, 
                      nx=self.nx, ny=self.ny,
                      lx=self.lx, ly=self.ly,
                      array=u * self.array)
        return field

    def get_sampling(self):
        return self.lx/self.nx, self.ly/self.ny

    def plot(self, name=''):
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow(np.abs(self.array), 
                   extent=(-self.lx/2, self.lx/2, self.ly/2, -self.ly/2))
        plt.title("Wavefront Amplitude")

        plt.subplot(2,2,2)
        plt.imshow(np.angle(self.array),
                   extent=(-self.lx/2, self.lx/2, self.ly/2, -self.ly/2))
        plt.title("Wavefront Phase")

        plt.subplot(2,2,3)
        plt.plot(np.arange(-self.nx//2, self.nx//2)*self.lx/self.nx, 
                 np.abs(self.array[self.ny//2,:]))
        plt.title("X-Slice")

        plt.subplot(2,2,4)
        plt.plot(np.arange(-self.ny//2, self.ny//2)*self.ly/self.ny, 
                 np.abs(self.array[:, self.nx//2]))
        plt.title("Y-Slice")

        plt.suptitle(name)


class PlaneWave(Field):
    def __init__(self, phase, wavelength, nx, ny, lx, ly) -> None:
        super().__init__(wavelength, nx, ny, lx, ly)
        self.array = jnp.ones((nx, ny)) * jnp.exp(1j*phase)

