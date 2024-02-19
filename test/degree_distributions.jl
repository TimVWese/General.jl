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

@testitem "combined degree distribution" begin
    # For existing degree distributions
    dds = [
        ([1, 2, 3], [4, 5, 6]),
        ([2, 3, 4], [7, 8, 9]),
        ([3, 4, 5], [10, 11, 12])
    ]

    # Call the function with the test data
    all_degrees, aggregated_counts = combined_degree_distribution(dds)

    # Check the results
    @test all_degrees == [1, 2, 3, 4, 5]
    @test aggregated_counts == [4, 12, 24, 20, 12]

    # Test from graphs
    g1 = PathGraph(4) # degree distribution: [2, 2, 0]
    g2 = StarGraph(4) # degree distribution: [3, 0, 1]
    g3 = CycleGraph(4) # degree distribution: [0, 4, 0]
    g4 = CompleteGraph(4) # degree distribution: [0, 0, 4]

    # Call the function with the test data
    result = combined_degree_distribution(g1, g2, g3, g4)

    # Check the results
    @test result == ([1, 2, 3], [5, 6, 5])
end

@testitem "joint distribution" begin
    using Graphs

    # Create two simple graphs
    g1 = PathGraph(5) # Graph with 5 vertices and 4 edges
    g2 = StarGraph(5) # Graph with 5 vertices and 4 edges
    g3 = CompleteGraph(5) # Graph with 5 vertices and 10 edges

    # Calculate the joint distribution
    result = joint_distribution(g1, g2, g3)

    # Expected output
    # For g1: degree distribution is [2, 3]
    # For g2: degree distribution is [4, 0, 0, 1]
    # For g3: degree distribution is [0, 0, 0, 5]
    @test size(result) == (2, 4, 4)
    @test vec(sum(result, dims=[2,3])) == [2, 3]
    @test vec(sum(result, dims=[1,3])) == [4,0,0,1]
    @test vec(sum(result, dims=[1,2])) == [0,0,0,5]

    @test result[1, 4, 4] == 1
    @test result[2, 1, 4] == 3
    @test result[1, 1, 4] == 1
end
