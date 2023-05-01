"""
Generic functions to manipulate supply chains. Optimization methods are in
separate modules.
"""
module SupplyChains

using HiGHS
using JuMP
using LinearAlgebra

export Capacity, Flow, Cost, SupplyChain, size, length

function check_eq(mess, e0, es...)
    for (i, e) in enumerate(es)
        if e0 != e
            error(mess * " at element $i: $e0 ≠ $e")
        end
    end
end

"Unitary cost for the installation in the supply chain"
struct Capacity
    suppliers
    plants
    distributors
    clients
end

"Stub equals because default struct equality doens't work for arrays"
function Base.:(==)(c1::Capacity, c2::Capacity)
    (
        isequal(c1.suppliers, c2.suppliers) &&
            isequal(c1.plants, c2.plants) &&
            isequal(c1.distributors, c2.distributors) &&
            isequal(c1.clients, c2.clients)
    )
end

"""
Returns size of each set of installations in the supply chain
"""
function Base.size(cap::Capacity)
    length.((cap.suppliers, cap.plants, cap.distributors, cap.clients))
end

"""
Generic struct to represent weight on the underling supply chain graph. It has
two uses (for now):
* unitary cost
* product load
"""
struct Flow
    supls_plants
    plants_dists
    dists_clients

    function Flow(cost_sp, cost_pd, cost_dc)
        _, np = size(cost_sp)
        _np, nd = size(cost_pd)
        check_eq("Incongruent matrices size", np, _np)

        _nd, _ = size(cost_dc)
        check_eq("Incongruent matrices size", nd, _nd)

        new(cost_sp, cost_pd, cost_dc)
    end
end

function Base.:(==)(f1::Flow, f2::Flow)
    (
        isequal(f1.supls_plants, f2.supls_plants)
        && isequal(f1.plants_dists, f2.plants_dists)
        && isequal(f1.dists_clients, f2.dists_clients)
    )
end

"""
Given a flow, put to zero all the inactive plants and dist centers weights.
"""
function active_flow(e, active_plants, active_dists)
    plants_matrix = Diagonal(active_plants)
    dists_matrix = Diagonal(active_dists)
    supl_plants_cost = e.supls_plants * plants_matrix
    plants_dist_cost = plants_matrix * e.plants_dists * dists_matrix
    dist_clients_cost = dists_matrix * e.dists_clients
    Flow(supl_plants_cost, plants_dist_cost, dist_clients_cost)
end

function Base.vec(e::Flow)
    sp = vec(e.supls_plants)
    pd = vec(e.plants_dists)
    dc = vec(e.dists_clients)
    [sp; pd; dc]
end

function Base.size(e::Flow)
    (size(e.supls_plants)..., size(e.dists_clients)...)
end

"Unitary cost and fixed cost in a single, convenient struct"
struct Cost
    plants
    distributors
    unitary

    function Cost(fixed_p, fixed_d, cost_sp, cost_pd, cost_dc)
        unitary_cost = Flow(cost_sp, cost_pd, cost_dc)
        d = size(unitary_cost)
        check_eq(
            "Incongruent fixed-unitary cost size",
            (length(fixed_p), length(fixed_d)),
            (d[2], d[3])
        )

        new(fixed_p, fixed_d, unitary_cost)
    end
end

function Base.:(==)(c1::Cost, c2::Cost)
    (
        isequal(c1.plants, c2.plants)
        && isequal(c1.distributors, c2.distributors)
        && isequal(c1.unitary, c2.unitary)
    )
end

Base.size(cost::Cost) = size(cost.unitary)

"""
Given a cost struct, the vector of active plants and distributors and the
active product load, calculate the price of the given setup.
"""
function (c::Cost)(ap, ad, load)
    dims = size(c)
    check_eq("Incongruent number of plants", length(ap), dims[2])
    check_eq("Incongruent number of distributors", length(ad), dims[3])

    au_cost = vec(active_flow(c.unitary, ap, ad))
    al_cost = transpose(au_cost) * vec(load)
    f_vec = [c.plants; c.distributors]
    f_cost = transpose(f_vec) * [ap; ad]

    al_cost + f_cost
end

"""
Finally, putting together everything.
* capacities vector (for assembly plants and distributors)
* cost (a Cost struct)
* max_* max allowed open installations
"""
struct SupplyChain
    capacities
    costs
    max_plants
    max_distributors

    function SupplyChain(cap, cost, mp, md)
        check_eq("Incongruent cost-capacity matrices", size(cap), size(cost))
        new(cap, cost, mp, md)
    end
end

function Base.:(==)(s1::SupplyChain, s2::SupplyChain)
    (
        isequal(s1.capacities, s2.capacities) && isequal(s1.costs, s2.costs)
        && isequal(s1.max_plants, s2.max_plants)
        && isequal(s1.max_distributors, s2.max_distributors)
    )
end

Base.size(chain::SupplyChain) = size(chain.capacities)

function split_pos(s::SupplyChain, pos_vec)
    l = length(pos_vec)
    dims = size(s)
    check_eq("Incongruent active nodes vector", l, dims[2] + dims[3])
    (pos_vec[1:dims[2]], pos_vec[dims[2]+1:l])
end

"""
Given a supply chain and a boolean vector of opened locations, determine if the
current configuration satisfier the demand without exiding the budgets.
"""
function is_feasible(chain, pos_vec)
    active_plants, active_dists = split_pos(chain, pos_vec)

    if sum(active_plants) > chain.max_plants
        false
    elseif sum(active_dists) > chain.max_distributors
        false
    else
        plants_potential = transpose(active_plants) * chain.capacities.plants
        dists_potential = transpose(active_dists) * chain.capacities.distributors

        demand = sum(chain.capacities.clients)

        demand <= plants_potential && demand <= dists_potential
    end
end

include("linear_programming.jl")

"""
Auxiliary structure to hold the initializations params and store the best
results as the optimization is taking place.
"""
mutable struct ChainParams
    chain
    loads
    best_load
    use_lp

    function ChainParams(chain, particles, use_lp)
        loads = [optimal_load(chain, p) for p in eachcol(particles)]
        s = size(particles)
        best_pos = argmin(
            [penalized_cost(chain, particles[:, i], loads[i]) for i in 1:s[2]]
        )
        new(chain, loads, loads[best_pos], use_lp)
    end
end

"""
Objective cost function. Unfeasible solutions are given a huge cost. Feasible
solutions are ranked by their regular cost.
"""
function penalized_cost(chain, pos, load)
    if is_feasible(chain, pos)
        ap, ad = split_pos(chain, pos)
        chain.costs(ap, ad, load)
    else
        10^5
    end
end

include("particle_swarm.jl")

using .ParticleSwarm

"""
Custome optimization function to apply linear optimization at the end of each
swarm optimization round.
"""
function ParticleSwarm.particles_cost(p::BoolParticles, c::ChainParams)
    if c.use_lp
        c.loads = mapslices(pos -> optimal_load(c.chain, pos), p.pos, dims=1)
    end

    new_costs = map(
        i -> penalized_cost(c.chain, p.pos[:, i], c.loads[i]),
        axes(p.pos, 2)
    )

    best_pos = argmin(new_costs)
    c.best_load = c.loads[best_pos]

    new_costs
end

"""
Create a swarm optimizer structure for the given supply chain instance.
"""
function swarm_optimizer(chain; num_particles=16, use_lp=true)
    dims = size(chain)
    vec_l = dims[2] + dims[3]
    particles = zeros(Bool, vec_l, num_particles)
    i = 1
    while i <= num_particles
        p = rand(Bool, vec_l)
        if is_feasible(chain, p)
            particles[:, i] = p
            i = i + 1
        end
    end

    optim = ChainParams(chain, particles, use_lp)
    Swarm(BoolParticles(map(x -> x, particles)), optim)
end

"""
Run a given number of optimization rounds with the given swarm optimizer.
"""
function optimize(swarm; steps = 100)
    for i in 1:steps
        ParticleSwarm.step!(swarm)
        @info "At step $i, best cost: $(swarm.best[2])"
    end
    swarm.best
end

end # module
