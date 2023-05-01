# Supply Chains

```@contents
```

Logistics in a system are the mechanisms for satisfying demands, such are raw materials or time, under certain restrictions, usually a budget, while optimizing a cost.

In an industrial environment, the losgistics are the supply chain. It usually has two stages: a production chain and a distribution chain.

In this toy example, the production chain has some fixed raw materials suppliers and some assembly plants. The distribution chain has some distribution center (supplied by the assembly plants) and selling points.

Each installation has a fixed operation cost, and transportation between installations has a cost per unit. Suppliers, assembly plants and distribution centers have a maximum capacity, and selling points have a demand.

## Supply Chain Manipulation

```@autodocs
Modules = [SupplyChains]
```

## Binary Particle Swarm Optimization

```@autodocs
Modules = [SupplyChains.ParticleSwarm]
```

## Linear Programming Optimization

```@autodocs
Modules = [SupplyChains.LP]
```
