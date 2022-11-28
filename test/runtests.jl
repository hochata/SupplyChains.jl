using Test, SupplyChains
import SupplyChains:active_flow

@testset "Creation" begin
    z = hcat(0)
    empty_cap = Capacity([0], [0], [0], [0])
    empty_cost = Cost([0], [0], z, z, z)
    empty_flow = Flow(z, z, z)
    empty_chain = SupplyChain(empty_cap, empty_cost, 0, 0)
    @test empty_cap.clients == [0]
    @test empty_cost.plants == [0]
    @test empty_cost.unitary.supls_plants == z
    @test empty_chain.max_plants == 0
end

dims = rand(1:10, 4)
o_vec = [ones(dims[1]), ones(dims[2]), ones(dims[3]), ones(dims[4])]
o_mtx = [
    ones(dims[1], dims[2]), ones(dims[2], dims[3]),
    ones(dims[3], dims[4])
]
one_cap = Capacity(o_vec[1], o_vec[2], o_vec[3], o_vec[4])
one_cost = Cost(o_vec[2], o_vec[3], o_mtx[1], o_mtx[2], o_mtx[3])
one_flow = Flow(o_mtx[1], o_mtx[2], o_mtx[3])
one_chain = SupplyChain(one_cap, one_cost, dims[2], dims[3])

z_mtx = [
    zeros(dims[1], dims[2]), zeros(dims[2], dims[3]),
    zeros(dims[3], dims[4])
]
z_vec = [zeros(dims[1]), zeros(dims[2]), zeros(dims[3]), zeros(dims[4])]
zero_cap = Capacity(z_vec[1], z_vec[2], z_vec[3], z_vec[4])
zero_cost = Cost(z_vec[2], z_vec[3], z_mtx[1], z_mtx[2], z_mtx[3])
zero_flow = Flow(z_mtx[1], z_mtx[2], z_mtx[3])
zero_chain = SupplyChain(zero_cap, zero_cost, dims[2], dims[3])

@testset "Flow" begin
    @test active_flow(one_flow, o_vec[2], o_vec[3]) == one_flow

    @test active_flow(one_flow, z_vec[2], z_vec[3]) == zero_flow

    only_dists = Flow(z_mtx[1], z_mtx[2], o_mtx[3])
    @test active_flow(one_flow, z_vec[2], o_vec[3]) == only_dists

    supls_plants = zeros(dims[1], dims[2])
    supls_plants[:, 1] = ones(dims[1])
    first_plant = copy(z_vec[2])
    first_plant[1] = 1
    first_plant_flow = Flow(
        supls_plants,
        zeros(dims[2], dims[3]),
        zeros(dims[3], dims[4])
    )
    @test active_flow(one_flow, first_plant, z_vec[3]) == first_plant_flow
end

@testset "Cost" begin
    rm = [
        rand(1:10, (dims[1], dims[2])), rand(1:10, (dims[2], dims[3])),
        rand(1:10, (dims[3], dims[4]))
    ]
    rand_unit_cost = Cost(z_vec[2], z_vec[3], rm[1], rm[2], rm[3])
    @test rand_unit_cost(o_vec[2], o_vec[3], one_flow) == sum(sum.(rm))
    @test rand_unit_cost(z_vec[2], z_vec[3], one_flow) == 0

    sp_only = Flow(o_mtx[1], z_mtx[2], z_mtx[3])
    @test rand_unit_cost(o_vec[2], o_vec[3], sp_only) == sum(rm[1])

    rf = [rand(1:10, dims[2]), rand(1:10, dims[3])]
    rand_fixed_cost = Cost(rf[1], rf[2], z_mtx[1], z_mtx[2], z_mtx[3])
    @test rand_fixed_cost(o_vec[2], o_vec[3], one_flow) == sum(sum.(rf))
end

@testset "Feasibility" begin
    @test SupplyChains.is_feasible(zero_chain, zeros(dims[2] + dims[3]))
    @test SupplyChains.is_feasible(zero_chain, ones(dims[2] + dims[3]))

    min_stage = min(dims[2], dims[3])
    dist_client_cost = ones(dims[3], min_stage)
    feasible_cost = Cost(o_vec[2], o_vec[3], o_mtx[1], o_mtx[2], dist_client_cost)
    feasible_cap = Capacity(o_vec[1], o_vec[2], o_vec[3], ones(min_stage))
    feasible_chain = SupplyChain(feasible_cap, feasible_cost, dims[2], dims[3])
    @test SupplyChains.is_feasible(feasible_chain, ones(dims[2] + dims[3]))
    @test !SupplyChains.is_feasible(feasible_chain, zeros(dims[2] + dims[3]))
end
