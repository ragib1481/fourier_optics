from fopy.propagators import Fresnel, FresnelTf
from fopy.components import RectangularAperture, CircAperture, ThinLens
from fopy.units import mm, nm, um, m
import matplotlib.pyplot as plt
import torch
import numpy as np

nx = 2048
ny = 2048
# lx = 500*um
# ly = 500*um
lx = 500*mm
ly = 500*mm
f = 10*mm
wl = 630*nm

for z in np.arange(2*mm, 22*mm, 2*mm):
    x = torch.ones(ny, nx)
    # aperture = RectangularAperture(lx, ly, 20*um, 20*um, nx, ny)
    aperture = CircAperture(lx, ly, 10*mm, nx, ny) 
    prop = Fresnel(lx, ly, nx, ny, z, wl)
    lens = ThinLens(lx, ly, nx, ny, f, wl) 

    x = aperture(x)
    x = lens(x)
    x = prop(x)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(torch.abs(x).numpy()[ny//2, :])
    plt.subplot(1,2,2)
    plt.imshow(torch.abs(x).numpy())
    plt.suptitle(f"Distance z: {z}, {prop.name()}")
plt.show()
