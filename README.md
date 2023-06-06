# Hybrid vehicle single track model
This repository contains `Julia` and `Python` code to reproduce numerical results of the manuscript:
*Vehicle single track modeling using physics informed neural differential equations*

## How to run it?
* Install [Julia](https://julialang.org/downloads/) (`v1.6.7` was used herein) and VScode with [Julia for VScode extension](https://www.julia-vscode.org/).
* Configure the local environment with `instantiate` in Julia's built in package manager [Pgk](https://docs.julialang.org/en/v1/stdlib/Pkg/). You can switch from `julia>` to `(your path) pkg>` by pressing `]` and backspace key.
* Open the files in `/doc` folder in VScode and run them with either `Shift + Enter` line by line in Julia's REPL or the entire file with `Ctrl + F5`.

As alternative, you can run the files without VScode from terminal with 
```
julia
]
activate .
<- (delete key)
julia> include("path/to/script.jl")
```

If this works, you can run the script files:
```
julia> include("doc/ode.jl")
julia> include("doc/node.jl")
julia> include("doc/hybrid_node.jl")
```
to reproduce the result figures, which are all stored in `results/` folder as `.html` files.
You can whatch these figures with any browser.

## How to generate the reference model .mat files?
The reference model is implemented with Python in the `reference_model` folder. You will find there a readme.
However, the file `reference_sinlge_track_drift.mat` contains the exported time series from the reference model
and is already stored in the root folder of this repository.

## Purpose
This software is a research prototype, solely developed for and 
published as part of the publication. It will neither be maintained 
nor monitored in any way.

## License
This software is licensed under AGPL-3.0 license.
