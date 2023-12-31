from fopy.propagators import Fresnel
from fopy.components import RectangularAperture, CircAperture, ThinLens
from fopy.units import mm, nm, um, m
import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

nx = 250
ny = 250
# lx = 500*um
# ly = 500*um
lx = 0.5*m
ly = 0.5*m
f = 10*mm
wl = 500*nm
z = 2000*m

for z in [1000*m, 2000*m, 4000*m, 20000*m]:
    x = jnp.ones((ny, nx))
    aperture = RectangularAperture(lx, ly, 0.051*m, 0.051*m, nx, ny)
    # aperture = CircAperture(lx, ly, 0.051*m, nx, ny) 
    prop = Fresnel(lx, ly, nx, ny, z, wl)
    lens = ThinLens(lx, ly, nx, ny, f, wl) 

    x = aperture(x)
    # x = lens(x)
    x = prop(x)

    plt.figure()
    plt.subplot(1,2,1)
    plt.plot(jnp.abs(x)[ny//2, :])
    plt.subplot(1,2,2)
    plt.imshow(jnp.abs(x))
    plt.suptitle(f"Distance z: {z}, {prop.name()}")
plt.show()
