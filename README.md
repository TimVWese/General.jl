[![Build Status](https://github.com/General.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TimVWese/General.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://timvwese.github.io/General.jl/dev/)

# General.jl

This badly named package adds some functions to ease my research.
They are mainly extensions to [Graphs.jl](https://github.com/JuliaGraphs/Graphs.jl) (more network models, IO) and [NetworkDynamics.jl](https://github.com/PIK-ICoNe/NetworkDynamics.jl) (some simple vertex and edge dynamics, multiplex supporty).

## Contents
This package provides among others

* Network models
  * Scale-free configuration
  * Stochastic spatial
* Network IO
* Basic `StaticEdge`s to use in `network_dynamics`
* Functionality to work with multiplex networks in `NetworkDynamics`

See the [docs](https://timvwese.github.io/General.jl/dev/) for full functionality.

## Installation

Due to heavy tailorization towards my own needs, the package is not included in any registry an can thus be installed by 

```julia
pkg> add https://github.com/TimVWese/General.jl
```
However it may be better to localize the relevant parts of the source code you need and copy them (without conditions or liability, see the (un)license), since the complete package might be bloated.
