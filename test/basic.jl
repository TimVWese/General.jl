@testitem "Basic functionality" begin
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

@testitem "Multiplex" begin
    using Graphs, NetworkDynamics, DifferentialEquations

    @test_throws AssertionError combine_graphs(Graph(2), Graph(3))
    # test the output Static edges and combined graph in simple case
    g1 = grid((2, 2))
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
    @test sol[:, 2] == [2, 2, 0, 2, 1, 0, 2, 0, 0, 2, 1, 0]

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
    @test sol[:, 2] == [4, 4, 0, 4, 2, 0, 4, 0, 0, 4, 2, 0]

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
    @test sol[:, 2] == [
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
            idxs = [false, true, true, false, true]
            for e in es
                vn[idxs] += e[idxs]
            end
        end
        return ODEVertex(f=f, dim=5)
    end

    vs, ses, cg = combine_graphs(het_dim_f, g1, g2; dims=[3, 2])
    nd = network_dynamics(vs, ses, cg)
    u0 = ones(Int64, 20)
    prob = DiscreteProblem(nd, u0, (0, 1))
    sol = solve(prob, FunctionMap())
    @test sol[:, 2] == [
        2, 2, 4, 2, 2,
        2, 2, 4, 1, 1,
        2, 2, 4, 0, 0,
        2, 2, 4, 1, 1,
    ]

    # test alternative invalid
    vs, ses, cg = combine_graphs(het_dim_f, g1, g2; dims=[3, 2], invalid=-1)
    nd = network_dynamics(vs, ses, cg)
    u0 = ones(Int64, 20)
    prob = DiscreteProblem(nd, u0, (0, 1))
    sol = solve(prob, FunctionMap())
    @test sol[:, 2] == [
        2, 1, 3, 2, 1,
        2, 2, 4, 1, 0,
        2, 2, 4, 0, -2,
        2, 1, 3, 1, -1,
    ]
end

@testitem "Read/write mtx" begin
    using Graphs

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

@testitem "Read tsv" begin
    using Graphs

    # Test undirected graph
    graph = read_from_tsv(joinpath(@__DIR__, "test.tsv"))
    @test nv(graph) == 4
    @test ne(graph) == 3
    @test isequal(typeof(graph), SimpleGraph{Int64})

    # Test directed graph
    graph = read_from_tsv(joinpath(@__DIR__, "test.tsv"); directed=true)
    @test nv(graph) == 4
    @test ne(graph) == 4
    @test isequal(typeof(graph), SimpleDiGraph{Int64})

    # Test reading only first two lines
    graph = read_from_tsv(joinpath(@__DIR__, "test.tsv"), N=2)
    @test nv(graph) == 2
    @test ne(graph) == 1
    @test isequal(typeof(graph), SimpleGraph{Int64})

    # Test reading non-existent file
    @test_throws SystemError read_from_tsv(joinpath(@__DIR__, "nonexistent.tsv"))
end

@testitem "Connectedness" begin
    using Graphs

    er = erdos_renyi(100, 0.005)
    while length(connected_components(er)) < 2
        er = erdos_renyi(100, 0.005)
    end
    nvg = nv(er)
    make_connected!(er)
    @test nvg == nv(er)
    @test length(connected_components(er)) == 1
    gc = copy(er)
    make_connected!(er)
    @test gc == er
end
