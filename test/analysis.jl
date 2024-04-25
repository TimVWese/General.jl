@testitem "bowtie" begin
    using Graphs

    g = SimpleDiGraph(17)
    # gscc: 1-4
    add_edge!(g, 1, 2)
    add_edge!(g, 2, 3)
    add_edge!(g, 3, 1)
    add_edge!(g, 3, 4)
    add_edge!(g, 4, 3)

    # goutc: 5
    add_edge!(g, 4, 5)

    # ginc: 8-11
    add_edge!(g, 8, 9)
    add_edge!(g, 8, 10)
    add_edge!(g, 9, 1)
    add_edge!(g, 10, 1)
    add_edge!(g, 11, 3)

    # tubes: 12-13
    add_edge!(g, 11, 12)
    add_edge!(g, 12, 13)
    add_edge!(g, 13, 5)

    # intendrils: 6-7
    add_edge!(g, 6, 5)
    add_edge!(g, 7, 6)

    # outtendrils: 14-15
    add_edge!(g, 9, 14)
    add_edge!(g, 11, 15)

    # Not connected: 16,17
    add_edge!(g, 16, 17)

    gscc, ginc, goutc = bowtie_decomposition(g)
    @test gscc == Set([1, 2, 3, 4])
    @test ginc == Set([8, 9, 10, 11])
    @test goutc == Set([5])

    gscc, ginc, goutc, tubes, outtendrils, intendrils = bowtie_decomposition(g, full=true)
    @test gscc == Set([1, 2, 3, 4])
    @test ginc == Set([8, 9, 10, 11])
    @test goutc == Set([5])
    @test tubes == Set([12, 13])
    @test intendrils == Set([6, 7])
    @test outtendrils == Set([14, 15])
end