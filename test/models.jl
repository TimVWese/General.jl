@testitem "Majority vertex" begin
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

@testitem "Configuration" begin
    using Graphs, Random
    Random.seed!(1234)

    g = configuration_model([3, 3, 3, 3]; allow_collisions=false)
    @test nv(g) == 4
    @test ne(g) == 6
    degs = zeros(Int64, 8000)
    degs[1:2] .= 1
    g = configuration_model(degs)
    @test nv(g) == 8000
    @test ne(g) == 1
end

@testitem "Scale free" begin
    using Graphs, Random
    Random.seed!(1234)

    g1 = SF_configuration_model(4, 2.3; min_d=3, allow_collisions=false)
    @test nv(g1) == 4
    @test ne(g1) == 6
    g2 = SF_configuration_model(6, 50)
    @test nv(g2) == 6
    @test ne(g2) == 3
end

@testitem "Spatial network" begin
    using Graphs

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

@testitem "Permuted circle graph" begin
    using Graphs

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
