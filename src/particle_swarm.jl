"""
Particle swarm is a nature inspired heuristic for optimization. It mimics the
patterns of flocks and fish schools to find global optimums.

At each step, each particle gets closer to its historical best and the global
best.
"""
module ParticleSwarm

export Particles, AccParticles, BoolParticles, Swarm, move!, step!, pso

function check_eq(mess, e0, es...)
    for (i, e) in enumerate(es)
        if e0 != e
            error(mess * "at element $i: $e0 ≠ $e")
        end
    end
end

abstract type Particles{T <: Real} end

eltype(::Type{<:Particles{T}}) where {T} = T

particles_cost(particles::T, obj_fn) where T <: Particles =
    mapslices(obj_fn, particles.pos, dims=1)

move!(p::T, best, α, β) where T <: Particles = move!(
    p,
    best,
    α,
    β,
    rand(0:0.01:1, size(p.pos)) .- 0.5
)

function update_ranks!(p::T, prev_cost, costs, curr_best) where T <: Particles
    (best_pos, best_cost) = curr_best
    for i in axes(p.pos, 1)
        if costs[i] < prev_cost[i]
            p.best_pos[:, i] = p.pos[:, i]
        end
        if costs[i] < best_cost
            best_pos = p.pos[:, i]
            best_cost = costs[i]
        end
    end
    (best_pos, best_cost)
end

"""
Accelerated particles only store their position.
"""
mutable struct AccParticles{T} <: Particles{T}
    pos::Matrix{T}
end

"""
Accelerated movement equation.
"""
function acc_move(p, best, α, β, ϵ)
    b = β .* best
    r = α .* ϵ
    (1 - β) .* p + b + r
end

function move!(p::AccParticles{T}, best, α, β, ϵ) where T <: Real
    for i in axes(p.pos, 2)
        p.pos[:, i] = acc_move(p.pos[:, i], best, α, β, ϵ[:, i])
    end
    p
end

function update_ranks!(p::AccParticles{T}, _, costs, curr_best) where T <: Real
    (best_pos, best_cost) = curr_best
    for i in axes(p.pos, 2)
        if costs[i] < best_cost
            best_pos = p.pos[:, i]
            best_cost = costs[i]
        end
    end
    (best_pos, best_cost)
end

mutable struct BoolParticles <: Particles{Bool}
    pos
    vel
    best_pos

    function BoolParticles(pos::Matrix{Bool})
        z = zeros(size(pos))
        new(pos, z, pos)
    end
end

function bool_move(p, p_vel, p_best, best, α, β, ϵ)
    vel_p_best = α .* (p - p_best)
    vel_best = β .* (p - best)
    vel = p_vel + ϵ .* (vel_p_best + vel_best)

    probs = sigmoid.(vel)
    (rnd_swap.(probs), vel)
end

sigmoid(x) = 1 / (1 + ℯ^(-x))

rnd_swap(p) = rand() < p

function move!(p::BoolParticles, best, α, β, ϵ)
    for i in axes(p.pos, 2)
        (n_pos, n_vel) = bool_move(
            p.pos[:, i],
            p.vel[:, i],
            p.best_pos[:, i],
            best,
            α,
            β,
            ϵ[:, i]
        )
        p.pos[:, i] = n_pos
        p.vel[:, i] = n_vel
    end
    p.pos
end

"""
A swarm of particles. It also keeps track of the current global best and common
particle values acceleration
"""
mutable struct Swarm
    particles
    best
    obj
    costs
    α
    β

    function Swarm(particles::T, obj_fn; α=0.2, β=0.5, kwargs...) where T <: Particles
        costs = particles_cost(particles, obj_fn)
        best = argmin(costs)
        best_pos = isa(best, Int) ? best : best[2]
        new(particles, (particles.pos[:, best_pos], costs[best_pos]), obj_fn, costs, α, β)
    end
end

function Swarm(dims::Integer, cost; num_particles=10, type=apso, range=0:0.1:1, α=0.2, β=0.5, kwargs...)
    particles =
        if type == apso
        AccParticles(rand(range, dims, num_particles))
    elseif type == bpso
            BoolParticles(rand(Bool, dims, num_particles))
    end
    Swarm(
        particles,
        cost,
        α=α,
        β=β,
        kwargs...
            )
end

@enum ParticleType begin
    apso
    bpso
end

function step!(swarm)
    move!(swarm.particles, swarm.best[1], swarm.α, swarm.β)

    prev_costs = swarm.costs
    swarm.costs = particles_cost(swarm.particles, swarm.obj)

    best = update_ranks!(swarm.particles, prev_costs, swarm.costs, swarm.best)
    if best != swarm.best
        swarm.best = best
    end

    swarm.best
end

function pso(cost, dims; steps=20, kwargs...)
    swarm = Swarm(dims, cost, kwargs...)
    for i in 1:steps
        step!(swarm)
        @info "At step $i, best cost: $(swarm.best[2])"
    end
    swarm.best
end

end # module
