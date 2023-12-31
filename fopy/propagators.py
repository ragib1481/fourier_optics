# TODO:Implement fresnel propagation kernel.
# TODO:Implement fraunhoffer propagation kernel.

import torch
import numpy as np
import matplotlib.pyplot as plt

class Fresnel(torch.nn.Module):
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

    def forward(self, u: torch.Tensor):
        return self.model(u)

    def name(self):
        return self.model.name()

    def plot(self):
        self.model.plot()


class FresnelIr(torch.nn.Module):
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 z: float, wl: float) -> None:

        super().__init__()
        x = torch.linspace(-lx/2, lx/2, nx, requires_grad=False)
        y = torch.linspace(-ly/2, ly/2, ny, requires_grad=False)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        k = 2 * np.pi / wl
        h = np.exp(1j * k * z) / (1j * wl * z)
        self.h = h * torch.exp(1j * k / 2 / z * (xx**2 + yy**2))

    def forward(self, u: torch.Tensor):
        x = torch.fft.fft2(torch.fft.ifftshift(u), norm="ortho")
        x = x * torch.fft.fft2(torch.fft.ifftshift(self.h))
        x = torch.fft.ifftshift(torch.fft.ifft2(x, norm="ortho")) 
        return x

    def name(self):
        return "Fresnel IR propagator"

    def plot(self):
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(torch.abs(self.h).numpy())

        plt.subplot(2, 2, 3)
        plt.imshow(torch.real(self.h).numpy())

        plt.subplot(2, 2, 4)
        plt.imshow(torch.imag(self.h).numpy())


class FresnelTf(torch.nn.Module):
    def __init__(self, 
                 lx: float, ly: float, 
                 nx: int, ny: int,
                 z: float, wl: float) -> None:

        super().__init__()
        fx = torch.fft.fftfreq(nx, d=lx/nx, requires_grad=False)
        fy = torch.fft.fftfreq(ny, d=ly/ny, requires_grad=False)
        fx = torch.fft.fftshift(fx)
        fy = torch.fft.fftshift(fy)
        fyy, fxx = torch.meshgrid(fy, fx, indexing='ij')
        k = 2 * np.pi / wl
        self.H = torch.exp(-1j * np.pi * wl * z * (fxx**2 + fyy**2))
        self.H = self.H * torch.tensor(1j * k * z, requires_grad=False)
        self.H = torch.fft.ifftshift(self.H)

    def forward(self, u: torch.Tensor):
        x = torch.fft.fft2(torch.fft.ifftshift(u), norm="ortho")
        x = x * self.H
        x = torch.fft.ifftshift(torch.fft.ifft2(x, norm="ortho")) 
        return x

    def name(self):
        return "Fresnel Tf propagator"

    def plot(self):
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.imshow(torch.abs(torch.fft.ifftshift(self.H)).numpy())

        plt.subplot(2, 2, 3)
        plt.imshow(torch.real(torch.fft.ifftshift(self.H)).numpy())

        plt.subplot(2, 2, 4)
        plt.imshow(torch.imag(torch.fft.ifftshift(self.H)).numpy())

