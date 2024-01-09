from fopy.propagators import Fresnel, Fresnel2, FresnelTf
from fopy.components import RectangularAperture, CircAperture, ThinLens
from fopy.units import mm, nm, um
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import jax.numpy as jnp
import numpy as np

nx = 2048 
ny = 2048 
# lx = 500*um
# ly = 500*um
lx = 25*mm
ly = 25*mm
f = 40*mm
wl = 630*nm
z = 20*mm

zs = np.arange(1*nm, 80*mm, 5*mm)
zs[1:] -= 1*nm
# zs = np.arange(50*mm, 150*mm, 10*mm)
img_array = np.zeros((zs.size, nx, ny))

# aperture = RectangularAperture(lx, ly, 0.051*mm, 0.051*mm, nx, ny)
aperture = CircAperture(lx, ly, 1*mm, nx, ny) 
lens = ThinLens(lx, ly, nx, ny, f, wl) 

for i, z in enumerate(zs):
    x = jnp.ones((ny, nx))
    prop = Fresnel2(lx, ly, nx, ny, z, wl)
    
    x = aperture(x)
    x = lens(x)
    x = prop(x)

    img_array[i, :, :] = jnp.abs(x)

fig, ax = plt.subplots()
im = ax.imshow(img_array[0, :, :])
plt.axis("Off")

def update(i):
    im.set_data(img_array[i, :, :])
    im.set_clim(vmin=img_array[i, :, :].min(), vmax=img_array[i, :, :].max())
    ax.relim()
    ax.set_title(f"z: {zs[i]}mm")
    fig.canvas.draw()
    fig.canvas.flush_events()
    return im,

animation_fig = animation.FuncAnimation(fig, update, frames=img_array.shape[0],
                                        interval=500, blit=True, repeat_delay=500)

# animation_fig.save("./results/fresnel_propagation_with_lens.gif", fps=5, dpi=300)
# animation_fig.save("./results/fresnelXform_propagation_with_lens.gif", fps=5, dpi=300)

plt.show()

