using General
using Test, Graphs
using NetworkDynamics
using DifferentialEquations

@testset "Basic functionality" begin
    es = [
        [1, 2],
        [4, 3],
        [1, 3, 2],
        [2, -2]
    ]
    @test count_states(es, 1, 1) == 2
    @test count_states(es, 2, 2) == 1

    e = [1, 2, 3]
    Ie!(e, [0, 0, 0], [1, 1, 1], [3], 12)
    @test e == [0, 0, 0]
end

@testset "Majority vertex" begin
    m_v = majority_vertex(2, 2)
    vn = [0, 0]
    m_v.f(vn, [32, 111], [[1, 2], [1, 2], [1, 2]], (Θ=0.5,), 33)
    @test vn == [32, 2]
    m_v.f(vn, [32, 111], [[1, 4], [1, 2], [1, 1]], (Θ=0.5,), 1)
    @test vn == [32, 1]
    m_v.f(vn, [32, 111], [[1, 1], [1, 2], [1, 1]], (Θ=0.1,), "haa")
    @test vn == [32, 2]
    m_v.f(vn, [32, 111], [[1, 1], [1, 2], [1, 1]], (Θ=0.5,), 4)
    @test vn == [32, 1]
    m_v.f(vn, [32, 111], [[1, 1], [1, 2]], (Θ=0.5,), 0)
    @test vn == [32, 2]
    m_v = majority_vertex(1, 2; S1=4, S2=8)
    m_v.f(vn, [32, 111], [[18, 2], [8, 2], [4, 2]], (Θ=0.5,), 0)
    @test vn == [4, 111]
    m_v.f(vn, [32, 111], [[4, 4], [8, 4], [8, 4]], (Θ=0.5,), 0)
    @test vn == [8, 111]
end

@testset "Configuration" begin
    g = configuration_model([3, 3, 3, 3]; allow_collisions=false)
    @test nv(g) == 4
    @test ne(g) == 6
    degs = zeros(Int64, 8000)
    degs[1:2] .= 1
    g = configuration_model(degs)
    @test nv(g) == 8000
    @test ne(g) == 1
end

@testset "Scale free" begin
    g1 = SF_configuration_model(4, 2.3; min_d=3, allow_collisions=false)
    @test nv(g1) == 4
    @test ne(g1) == 6
    g2 = SF_configuration_model(6, 50)
    @test nv(g2) == 6
    @test ne(g2) == 3
end

@testset "Spatial network" begin
    g = spatial_network([0 0; 1 1])
    @test nv(g) == 2
    @test ne(g) == 0
    g = spatial_network([0 0; 1 1]; f=x->1-x)
    @test nv(g) == 2
    @test ne(g) == 1
    g = spatial_network([0 0; 0 1; 1 0; 1 1]; f=x->1*(x>0.9))
    @test nv(g) == 4
    @test ne(g) == 4
    @test !has_edge(g, 1, 4)
    @test !has_edge(g, 2, 3)
    g, ps = spatial_network(3, 2; f = x->1.)
    @test nv(g) == 3
    @test ne(g) == 0
    @test size(ps) == (3, 2)
    @test all(0. .<= ps[:] .<= 1.)
    g, _ = spatial_network(4, 16; f = x->0.)
    @test nv(g) == 4
    @test ne(g) == 3*2*1
end

@testset "Multiplex" begin
    @test_throws AssertionError combine_graphs(Graph(2), Graph(3))
    # test the output Static edges and combined graph in simple case
    g1 = grid((2,2))
    g2 = Graph(4)
    add_edge!(g2, 1, 2)
    add_edge!(g2, 1, 4)
    g3 = Graph(4)
    ses, cg = combine_graphs(g1, g2, g3)
    @test ne(cg) == 5
    @test length(ses) == 5
    for se in ses
        @test se.dim == 6 # Should be 3, but undirectedness doubles dimension
    end

    # test the dynamics in the simple case
    simple_f = (vn, v, es, p, t) -> begin
        for i in eachindex(v)
            vn[i] = 0
            for e in es
                vn[i] += e[i]
            end
        end
    end

    v = ODEVertex(f=simple_f, dim=3)
    nd = network_dynamics(v, ses, cg)
    u0 = ones(Int64, 12)
    prob = DiscreteProblem(nd, u0, (0, 1))
    sol = solve(prob, FunctionMap())
    @test sol[2] == [2, 2, 0, 2, 1, 0, 2, 0, 0, 2, 1, 0]

    # Test the dynamics with degrees, but one-dimesnionals
    complex_f(ds) = begin
        f = (vn, v, es, p, t) -> begin
            for i in eachindex(v)
                vn[i] = ds[i]
                for e in es
                    vn[i] += e[i]
                end
            end
        end
        return ODEVertex(f=f, dim=3)
    end

    vs, ses, cg = combine_graphs(complex_f, g1, g2, g3)
    nd = network_dynamics(vs, ses, cg)
    u0 = ones(Int64, 12)
    prob = DiscreteProblem(nd, u0, (0, 1))
    sol = solve(prob, FunctionMap())
    @test sol[2] == [4, 4, 0, 4, 2, 0, 4, 0, 0, 4, 2, 0]

    # three 2-dimesnional layers
    two_dim_f(ds) = begin
        f = (vn, v, es, p, t) -> begin
            vn[1:2:6] .= ds
            vn[2:2:6] .= 0
            for e in es
               vn[2:2:6] .+= e[2:2:6]
            end
        end
        return ODEVertex(f=f, dim=6)
    end

    vs, ses, cg = combine_graphs(two_dim_f, g1, g2, g3; dims=2)
    nd = network_dynamics(vs, ses, cg)
    u0 = ones(Int64, 24)
    prob = DiscreteProblem(nd, u0, (0, 1))
    sol = solve(prob, FunctionMap())
    @test sol[2] == [
        2, 2, 2, 2, 0, 0,
        2, 2, 1, 1, 0, 0,
        2, 2, 0, 0, 0, 0,
        2, 2, 1, 1, 0, 0
    ]

    # het 2-dimensional layers
    het_dim_f(ds) = begin
        f = (vn, v, es, p, t) -> begin
            vn[1] = ds[1]
            vn[4] = ds[2]
            vn[2] = 0
            vn[3] = ds[1]
            vn[5] = 0 
            idxs = [false, true, true, false,true]
            for e in es
               vn[idxs] += e[idxs]
            end
        end
        return ODEVertex(f=f, dim=5)
    end

    vs, ses, cg = combine_graphs(het_dim_f, g1, g2; dims=[3,2])
    nd = network_dynamics(vs, ses, cg)
    u0 = ones(Int64, 20)
    prob = DiscreteProblem(nd, u0, (0, 1))
    sol = solve(prob, FunctionMap())
    @test sol[2] == [
        2, 2, 4, 2, 2,
        2, 2, 4, 1, 1,
        2, 2, 4, 0, 0,
        2, 2, 4, 1, 1,
    ]

    # test alternative invalid
    vs, ses, cg = combine_graphs(het_dim_f, g1, g2; dims=[3,2], invalid=-1)
    nd = network_dynamics(vs, ses, cg)
    u0 = ones(Int64, 20)
    prob = DiscreteProblem(nd, u0, (0, 1))
    sol = solve(prob, FunctionMap())
    @test sol[2] == [
        2, 1, 3, 2, 1,
        2, 2, 4, 1, 0,
        2, 2, 4, 0, -2,
        2, 1, 3, 1, -1,
    ]
end

@testset "Read/write" begin
    go = erdos_renyi(20, 0.15)
    write_to_mtx("g1", go)
    write_to_mtx("g2.mtx", go)
    g1 = read_from_mtx("g1.mtx")
    g2 = read_from_mtx("g2.mtx")
    @test g1 == go
    @test g2 == go
    rm("g1.mtx")
    rm("g2.mtx")
end

@testset "Connectedness" begin
    g = Graph()
    while length(connected_components(g)) < 2
        g = erdos_renyi(100, 0.005)
    end
    nvg = nv(g)
    make_connected!(g)
    @test nvg == nv(g)
    @test length(connected_components(g)) == 1
    gc = copy(g)
    make_connected!(g)
    @test gc == g
end

