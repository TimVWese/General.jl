@testitem "Vietoris-Rips tests" begin
    using Simplicial

    # Test 2D points
    points = [0 1 2 3; 0 0 0 0]
    simplicial_complex = vietoris_rips(points, 1.5)
    @test length(simplicial_complex.vertices) == 4
    @test length(simplicial_complex.facets) == 3
    @test all(length.(simplicial_complex.facets) .== 2)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})

    # Test 3D points
    points = [0 1 2 3; 0 0 0 0; 0 0 0 0]
    simplicial_complex = vietoris_rips(points, 1.5)
    @test length(simplicial_complex.vertices) == 4
    @test length(simplicial_complex.facets) == 3
    @test all(length.(simplicial_complex.facets) .== 2)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})

    # Test negative epsilon
    points = [0 1 2 3; 0 0 0 0]
    simplicial_complex = vietoris_rips(points, -1.5)
    @test length(simplicial_complex.vertices) == 4
    @test length(simplicial_complex.facets) == 1
    @test all(length.(simplicial_complex.facets) .== 4)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})
end

@testitem "Min k Vietoris Rips complex" begin
    using Simplicial, Random
    Random.seed!(1234)

    # Test 2D points
    simplicial_complex = vietoris_rips_mink(10; min_k=2)
    @test length(simplicial_complex.vertices) == 10
    @test length(simplicial_complex.facets) > 0
    @test all(simplicial_complex.dimensions .>= 2)
    @test any(simplicial_complex.dimensions .== 2)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})

    # Test 3D points
    simplicial_complex = vietoris_rips_mink(10; d=3)
    @test length(simplicial_complex.vertices) == 10
    @test length(simplicial_complex.facets) > 0
    @test all(simplicial_complex.dimensions .>= 2) broken = true
    @test any(simplicial_complex.dimensions .== 2)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})

    # Test higher-dimensional points
    simplicial_complex = vietoris_rips_mink(10; d=2, min_k=5)
    @test length(simplicial_complex.vertices) == 10
    @test length(simplicial_complex.facets) > 0
    @test all(simplicial_complex.dimensions .>= 5)
    @test any(simplicial_complex.dimensions .== 5)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})
end

@testitem "Random Simplicial Complex" begin
    using Simplicial, Random
    Random.seed!(1234)

    # General test
    simplicial_complex = random_simplicial_complex(10, [0.5, 0.2, 0.1])
    @test length(simplicial_complex.vertices) == 10
    @test length(simplicial_complex.facets) > 0
    @test all(simplicial_complex.dimensions .>= 0)
    @test all(simplicial_complex.dimensions .<= 3)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})

    # Test 1D complex
    simplicial_complex = random_simplicial_complex(5, [1.0])
    @test length(simplicial_complex.vertices) == 5
    @test length(simplicial_complex.facets) == 10
    @test all(simplicial_complex.dimensions .== 1)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})

    # Test 3D complex
    simplicial_complex = random_simplicial_complex(5, [1.0, 1.0, 1.0])
    @test length(simplicial_complex.vertices) == 5
    @test all(simplicial_complex.dimensions .== 3)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})

    # Test 5D complex
    simplicial_complex = random_simplicial_complex(6, [1.0, 1.0, 1.0, 1.0, 1.0])
    @test length(simplicial_complex.vertices) == 6
    @test length(simplicial_complex.facets) == 1
    @test all(simplicial_complex.dimensions .== 5)
    @test isequal(typeof(simplicial_complex), SimplicialComplex{Int64})

    # Test connected complex
    @test_throws AssertionError simplicial_complex_error = random_simplicial_complex(10, [0.0, 0.2, 0.1])
end