"Convert supply chains into linear programming problems"
module LP

using HiGHS
using JuMP

export optimal_load

"""
Use linear programming to optimize  for the open locations given in
the position vector.

Returns a chain flow with the optimal load.
"""
function optimal_load(chain, pos; optimizer = HiGHS.Optimizer)
    s = size(chain)
    caps = chain.capacities
    m = Model(optimizer)
    set_silent(m)

    @variable(m, 0 <= xsp[1:s[1], 1:s[2]])
    @variable(m, 0 <= xpd[1:s[2], 1:s[3]])
    @variable(m, 0 <= xdc[1:s[3], 1:s[4]])

    @constraint(m, [i = 1:s[1]], sum(xsp[i, j] for j in 1:s[2]) <= caps.suppliers[i])
    @constraint(m, [i = 1:s[2]], sum(xpd[i, j] for j in 1:s[3]) <= caps.plants[i])
    @constraint(m, [i = 1:s[3]], sum(xdc[i, j] for j in 1:s[4]) <= caps.distributors[i])
    @constraint(m, [j = 1:s[4]], sum(xdc[i, j] for i in 1:s[3]) >= caps.clients[j])

    @constraint(
        m,
        [j = 1:s[2]],
        sum(xsp[i, j] for i in 1:s[1]) == sum(xpd[j, k] for k in 1:s[3])
    )
    @constraint(
        m,
        [j = 1:s[3]],
        sum(xpd[i, j] for i in 1:s[2]) == sum(xdc[j, k] for k in 1:s[4])
    )

    ap, ad = split_pos(chain, pos)
    flow_cost_vec = transpose(vec(active_flow(chain.costs.unitary, ap, ad)))
    @objective(
        m,
        Min,
        flow_cost_vec * [vec(xsp); vec(xpd); vec(xdc)]
    )

    optimize!(m)
    opt_sp = [value(xsp[i, j]) for i in 1:s[1], j in 1:s[2]]
    opt_pd = [value(xpd[i, j]) for i in 1:s[2], j in 1:s[3]]
    opt_dc = [value(xdc[i, j]) for i in 1:s[3], j in 1:s[4]]

    Flow(opt_sp, opt_pd, opt_dc)
end

end # module
