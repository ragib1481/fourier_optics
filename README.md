# Introduction:
This package is built on top of [JAX](https://github.com/google/jax) numerical package 
as a way to simulate optical propagation with Fourier Optics. Majority of the code is 
based on the books:
1. Voelz, D.G., 2011. Computational fourier optics: a MATLAB tutorial.
2. Goodman, J.W., 2005. Introduction to Fourier optics. Roberts and Company publishers.

If you are planning to use this for your projects be-careful of the sampling parameters
and aliasing artifacts.

There is a small example in the main.py file to show scalar propagation. I hope to
build an extensive simulation package for fourier optics overtime. For now a reliable 
package for scalar light propagation simulation is [POPPY] (https://github.com/spacetelescope/poppy).

On the other hand this is my personal project and the sole idea is to have a simulation
package with auto-differentiation capability.

I chose jax to keep support for autograd and optimizers from jaxopt package. 
    
