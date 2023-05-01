# SupplyChains

Simple implementation of a mathematical model of a supply chain, with some cost optimization heuristics.

## Development setup
Using [ASDF](https://asdf-vm.com/), just run
```sh
asdf install
```
to get the exact julia version used to develop this package.

The run a Julia session at the root of the repo with
```julia-repl
julia> ]
(@v1.8) pkg> activate .
  Activating project at `.../SupplyChains.jl`

(SupplyChains) pkg> instantiate
```

to install all the dependencies.
