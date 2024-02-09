from fopy.fields import Field, PlaneWave
from fopy.propagators import FresnelIR, FresnelTF, Fresnel
from fopy.components import CircAperture, RectangularAperture, ThinLens 
from fopy.units import mm, nm, um, m
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import numpy as np

nx = 1024 
ny = 1024 
lx = 30*mm
ly = 30*mm
wl = 630*nm
z = 100*mm

# define input field
x = PlaneWave(phase=0, wavelength=wl, nx=nx, ny=ny, lx=lx, ly=ly)

# define an aperture
# aperture = RectangularAperture(0.25*mm, 0.25*mm)
aperture = CircAperture(10*mm)

# define a propagator
prop = Fresnel(z)

# field through the aperture 
x = aperture(x)
x.plot()

# propagate the field
x = prop(x)
x.plot(f"Field After Propagation, valid: {x.is_valid()}")

plt.show()

