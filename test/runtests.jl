using General
using Random
using Test, Graphs
using NetworkDynamics
using DifferentialEquations
using Simplicial: SimplicialComplex

Random.seed!(1234)

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
    g = spatial_network([0 0; 1 1]; f=x -> 1 - x)
    @test nv(g) == 2
    @test ne(g) == 1
    g = spatial_network([0 0; 0 1; 1 0; 1 1]; f=x -> 1 * (x > 0.9))
    @test nv(g) == 4
    @test ne(g) == 4
    @test !has_edge(g, 1, 4)
    @test !has_edge(g, 2, 3)
    g, ps = spatial_network(3, 2; f=x -> 1.0)
    @test nv(g) == 3
    @test ne(g) == 0
    @test size(ps) == (3, 2)
    @test all(0.0 .<= ps[:] .<= 1.0)
    g, _ = spatial_network(4, 16; f=x -> 0.0)
    @test nv(g) == 4
    @test ne(g) == 3 * 2 * 1
end

@testset "Permuted circle graph" begin
    g = permuted_circle(4, 0)
    @test nv(g) == 4
    @test ne(g) == 4
    @test has_edge(g, 1, 2)
    @test has_edge(g, 2, 3)
    @test has_edge(g, 3, 4)
    @test has_edge(g, 4, 1)
    @test diameter(g) == 2

    g = permuted_circle(8, 100)
    @test nv(g) == 8
    @test ne(g) == 8
    for d in degree(g)
        @test d == 2
    end
    @test diameter(g) == 4
end

@testset "Multiplex" begin
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
    @test sol[2] == [
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
    @test sol[2] == [
        2, 1, 3, 2, 1,
        2, 2, 4, 1, 0,
        2, 2, 4, 0, -2,
        2, 1, 3, 1, -1,
    ]
end

@testset "conditional degree" begin
    is_valid = v -> all(x -> (isapprox(1.0, x) || isapprox(0.0, x)), v)

    g1 = erdos_renyi(20, 0.2)
    P1 = conditional_degree_distribution(g1)
    @test size(P1) == (maximum(degree(g1)), maximum(degree(g1)))
    @test is_valid(sum(P1, dims=2))
    @test all(0 .<= P1 .<= 1)

    g3 = grid((10, 10), periodic=true)
    P3 = conditional_degree_distribution(g3)
    @test size(P3) == (4, 4)
    @test is_valid(sum(P3, dims=2))
    @test P3[4, 4] ≈ 1.0
    @test sum(P3, dims=[1, 2])[1] ≈ 1.0
end

@testset "Read/write mtx" begin
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

@testset "Read tsv" begin
    # Test undirected graph
    g = read_from_tsv("test.tsv")
    @test nv(g) == 4
    @test ne(g) == 3
    @test isequal(typeof(g), SimpleGraph{Int64})

    # Test directed graph
    g = read_from_tsv("test.tsv", directed=true)
    @test nv(g) == 4
    @test ne(g) == 4
    @test isequal(typeof(g), SimpleDiGraph{Int64})

    # Test reading only first two lines
    g = read_from_tsv("test.tsv", N=2)
    @test nv(g) == 2
    @test ne(g) == 1
    @test isequal(typeof(g), SimpleGraph{Int64})

    # Test reading non-existent file
    @test_throws SystemError read_from_tsv("nonexistent.tsv")
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

@testset "Vietoris-Rips tests" begin
    # Test 2D points
    points = [0 1 2 3; 0 0 0 0]
    sc = vietoris_rips(points, 1.5)
    @test length(sc.vertices) == 4
    @test length(sc.facets) == 3
    @test all(length.(sc.facets) .== 2)
    @test isequal(typeof(sc), SimplicialComplex{Int64})

    # Test 3D points
    points = [0 1 2 3; 0 0 0 0; 0 0 0 0]
    sc = vietoris_rips(points, 1.5)
    @test length(sc.vertices) == 4
    @test length(sc.facets) == 3
    @test all(length.(sc.facets) .== 2)
    @test isequal(typeof(sc), SimplicialComplex{Int64})

    # Test negative epsilon
    points = [0 1 2 3; 0 0 0 0]
    sc = vietoris_rips(points, -1.5)
    @test length(sc.vertices) == 4
    @test length(sc.facets) == 1
    @test all(length.(sc.facets) .== 4)
    @test isequal(typeof(sc), SimplicialComplex{Int64})
end

@testset "Min k Vietoris Rips complex" begin
    # Test 2D points
    sc = vietoris_rips_mink(10; min_k=2)
    @test length(sc.vertices) == 10
    @test length(sc.facets) > 0
    @test all(sc.dimensions .>= 2)
    @test any(sc.dimensions .== 2)
    @test isequal(typeof(sc), SimplicialComplex{Int64})

    # Test 3D points
    sc = vietoris_rips_mink(10; d=3)
    @test length(sc.vertices) == 10
    @test length(sc.facets) > 0
    @test all(sc.dimensions .>= 2) broken=true
    @test any(sc.dimensions .== 2)
    @test isequal(typeof(sc), SimplicialComplex{Int64})

    # Test higher-dimensional points
    sc = vietoris_rips_mink(10; d=2, min_k=5)
    @test length(sc.vertices) == 10
    @test length(sc.facets) > 0
    @test all(sc.dimensions .>= 5)
    @test any(sc.dimensions .== 5)
    @test isequal(typeof(sc), SimplicialComplex{Int64})
end

@testset "Random Simplicial Complex" begin
    # General test
    sc = random_simplicial_complex(10, [0.5, 0.2, 0.1])
    @test length(sc.vertices) == 10
    @test length(sc.facets) > 0
    @test all(sc.dimensions .>= 0)
    @test all(sc.dimensions .<= 3)
    @test isequal(typeof(sc), SimplicialComplex{Int64})

    # Test 1D complex
    sc = random_simplicial_complex(5, [1.0])
    @test length(sc.vertices) == 5
    @test length(sc.facets) == 10
    @test all(sc.dimensions .== 1)
    @test isequal(typeof(sc), SimplicialComplex{Int64})

    # Test 3D complex
    sc = random_simplicial_complex(5, [1.0, 1.0, 1.0])
    @test length(sc.vertices) == 5
    @test all(sc.dimensions .== 3)
    @test isequal(typeof(sc), SimplicialComplex{Int64})

    # Test 5D complex
    sc = random_simplicial_complex(6, [1.0, 1.0, 1.0, 1.0, 1.0])
    @test length(sc.vertices) == 6
    @test length(sc.facets) == 1
    @test all(sc.dimensions .== 5)
    @test isequal(typeof(sc), SimplicialComplex{Int64})

    # Test connected complex
    @test_throws AssertionError sc = random_simplicial_complex(10, [0.0, 0.2, 0.1])
end


