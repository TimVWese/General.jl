@testitem "degree_distribution" begin
    using Graphs

    # Test empty graph
    g = Graph(0)
    degree_list, degree_counts = degree_distribution(g)
    @test length(degree_list) == 0
    @test length(degree_counts) == 0

    # Test graph with one node
    g = Graph(1)
    degree_list, degree_counts = degree_distribution(g)
    @test length(degree_list) == 1
    @test length(degree_counts) == 1
    @test degree_list[1] == 0
    @test degree_counts[1] == 1

    # Test graph with two nodes and one edge
    g = Graph(2)
    add_edge!(g, 1, 2)
    degree_list, degree_counts = degree_distribution(g)
    @test length(degree_list) == 1
    @test length(degree_counts) == 1
    @test degree_list[1] == 1
    @test degree_counts[1] == 2

    # Test graph with three nodes and two edges
    g = Graph(3)
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    degree_list, degree_counts = degree_distribution(g)
    @test length(degree_list) == 2
    @test length(degree_counts) == 2
    @test degree_list[1] == 1
    @test degree_counts[1] == 2
    @test degree_list[2] == 2
    @test degree_counts[2] == 1

    # Test 6-dimensional grid
    g = grid((3, 3, 3), periodic=true)
    degree_list, degree_counts = degree_distribution(g)
    @test length(degree_list) == 1
    @test length(degree_counts) == 1
    @test degree_list[1] == 6
    @test degree_counts[1] == 27
end

@testitem "conditional degree" begin
    using Graphs, Random
    Random.seed!(1234)

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