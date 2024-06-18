module General

export count_states
export Ie!
export IStaticEdge, SStaticEdge
export combine_graphs
export majority_vertex, majority_termination_cb
export configuration_model
export SF_configuration_model
export spatial_network
export permuted_circle
export make_connected!
export degree_distribution
export conditional_degree_distribution
export combined_degree_distribution
export joint_distribution
export bowtie_decomposition
export read_from_mtx, write_to_mtx
export read_from_tsv
export vietoris_rips, vietoris_rips_mink
export random_simplicial_complex

using DifferentialEquations: DiscreteCallback, terminate!
using Distributions: Categorical
using Graphs
using MatrixMarket
using Simplicial: SimplicialComplex
using NetworkDynamics: StaticEdge, ODEVertex
using LinearAlgebra: norm, Symmetric
using Combinatorics: combinations
using Random: shuffle!
using Logging

"""
    count_states(es, i, s)

Compute the number of edges in `es` which have state `s` at index `i`.

# Examples
```jldoctest
julia> count_states([[1,3], [3,3],[1,1]], 2, 3)
2
```
"""
function count_states(es, i, s)
    return count(e -> e[i] == s, es)
end

"""
Internal dynamics for an identity `StaticEdge`.
Copies the contents of v_s to e.

See also [`IStaticEdge`](@ref)
"""
function Ie!(e, v_s, v_d, p, t)
    e .= v_s
end

"""
Constructor for an identity `StaticEdge`with dimension `dim`.

See also [`Ie!`](@ref), [`SStaticEdge`](@ref)
"""
function IStaticEdge(dim)
    return StaticEdge(f=Ie!, dim=dim, coupling=:undirected)
end

"""
    SStaticEdge(selection; invalid=0)

Constructor for a `StaticEdge` with selection dynamics.
The edge has dimension equal to `length(selection)`.
After evaluation, `e[i]==v_s[i]` if `selection[i]==true` and else `e[i]==invalid`.

See also [`combine_graphs`](@ref), [`IStaticEdge`](@ref)
"""
function SStaticEdge(selection; invalid=0)
    dim = length(selection)
    f = (e, v_s, v_d, p, t) -> begin
        for i in eachindex(v_s)
            e[i] = selection[i] ? v_s[i] : invalid
        end
    end
    return StaticEdge(f=f, dim=dim, coupling=:undirected)
end

struct CombinedGraphIterator
    gs::Tuple
    nv::Int
end

function Base.iterate(cgi::CombinedGraphIterator, state=[1, 0])
    if state[1] > cgi.nv
        return nothing
    end
    while (state[1] <= cgi.nv)
        state[2] += 1
        if state[2] > cgi.nv
            state[1] += 1
            state[2] = state[1] + 1
        end
        for g in cgi.gs
            if has_edge(g, state[1], state[2])
                return (Graphs.SimpleGraphs.SimpleEdge(state...), state)
            end
        end
    end
    return nothing
end

"""
    static_edges, combined_graph = combine_graphs(gs::AbstractGraph...[; dims=[1], invalid=0])

Generate the `combined_graph` which edges are given by the union of the edges of `gs`.
The goal of this function is offer an interface that allows to use multiplex networks in
`network_dynamics`. Each argument represents a single layer.
The array `static_edges` consists of `SStaticEdge`s.
The named keyword `dims` gives the dimension the `StaticEdge` in each layer ought to have.
If `length(dims)==1`, all dimensions are equal to `dim[1]`.
The selection of `static_edges[i]` is given by 
```
[[edge ∈ edges(gs[i]) for j in 1:dims[i]] for i in eachindex(gs)]
```
The output of the `StaticEdge` dynamics corresponding to non-existing edges is set to `invalid`.

See also [`SStaticEdge`](@ref)
"""
function combine_graphs(gs::AbstractGraph...; dims=[1], invalid=0)
    # Create a graph containing the edges of all input graphs
    n_verts = nv(gs[1])
    for g in gs
        @assert n_verts == nv(g) "Graphs have different number of vertices"
    end
    edge_iter = CombinedGraphIterator(gs, n_verts)
    combined_graph = Graphs.SimpleGraphs.SimpleGraphFromIterator(edge_iter)

    # Create an array of the selection edges depending on the existence
    # of edges in the different layers
    dims = length(dims) == 1 ? dims[1] * ones(Int64, length(gs)) : dims
    @assert length(gs) == length(dims) "Number of dimensions does not equal number of graphs"
    static_edges = Array{StaticEdge}(undef, ne(combined_graph))
    for (i, edge) in enumerate(edges(combined_graph))
        static_edges[i] = SStaticEdge(vcat(
                [[edge ∈ edges(gs[i]) for j in 1:dims[i]] for i in eachindex(gs)]...
            ); invalid=invalid)
    end

    return static_edges, combined_graph
end

"""
    ode_vertices, static_edges, combined_graph =
        combine_graphs(v_f::Function, gs::AbstractGraph...[; dims=[1]])

Generate the `combined_graph` which edges are given by the union of the edges of `gs`.
The goal of this function is offer an interface that allows to use multiplex networks in
`network_dynamics`. Each argument represents a single layer.
This method ought to bes used when the vertex dynamics depend on the degree in each layer.

# Arguments
- `v_f::Function`: a constructor `ds -> ODEVertex`, where ds[i] is the degree of the corresponding
vertex in `gs[i]`.
- gs::AbstractGraph...: graphs to combine

See also [`SStaticEdge`](@ref)
"""
function combine_graphs(v_f::Function, gs::AbstractGraph...; dims=[1], invalid=0)
    static_edges, combined_graph = combine_graphs(gs...; dims=dims, invalid=invalid)
    ode_vertices = [v_f([degree(g, v) for g in gs]) for v in vertices(combined_graph)]
    return ode_vertices, static_edges, combined_graph
end


"""
    majority_vertex(i::Integer, dim::Integer; U=1, A=2)

Construct a `ODEVertex` with binary Watts/threshold dynamics.

# Arguments
- i::Integer: index of the relevant state
- dim::Integer: dimension of the total state
- S1=1: value of state 1
- S2=2: value of state two
"""
function majority_vertex(i::Integer, dim::Integer; S1=1, S2=2)
    f = (vₙ, v, es, p, t) -> begin
        vₙ .= v
        vₙ[i] = count_states(es, i, S2) / length(es) >= p.Θ ? S2 : S1
    end
    return ODEVertex(f=f, dim=dim)
end

function majority_termination(u, t, integrator)
    t < 3 ? false : u == integrator.sol(t - 2)
end

majority_termination_cb = DiscreteCallback(majority_termination, terminate!)

# Generate a stub list from a degree distribution
function to_stublist(degree_seq)
    stublist = Int64[]
    for (i, d) in enumerate(degree_seq)
        append!(stublist, fill(i, d))
    end
    return stublist
end

"""
    configuration_model(degree_seq; allow_collisions=true, max_tries=1000)

Contruct a graph with degree distribution `degree_seq` using the configuration model.
The implementation is based on that of the NetworkX python package.

# Arguments
- degree_seq::AbstractVector{<:Integer}: degree sequence of the graph
- allow_collisions::Bool: if `true`, ignore multiple edges between the same pair of nodes and self-loops
- max_tries::Integer: maximum number of attempts to generate a graph

# Examples
```jldoctest
julia> configuration_model([3, 3, 3, 3]; allow_collisions=false)
{4, 6} undirected simple Int64 graph
julia> configuration_model([0, 0])
{2, 0} undirected simple Int64 graph
```
"""
function configuration_model(degree_seq; allow_collisions=true, max_tries=1000)
    @assert isgraphical(degree_seq) "Degree sequence is not graphical"
    N = length(degree_seq)
    stublist = to_stublist(degree_seq)

    for i in 1:max_tries
        g = Graph(N)
        shuffle!(stublist)
        failed = false
        i = 1
        while (i <= length(stublist) && !failed)
            if !allow_collisions && (stublist[i] == stublist[i+1] || has_edge(g, stublist[i], stublist[i+1]))
                failed = true
            else
                add_edge!(g, stublist[i], stublist[i+1])
            end
            i += 2
        end
        if !failed
            return g
        end
    end
    @assert false "Could not generate a graph with the given degree sequence"
end

"""
    SF_configuration_model(N, γ; min_d=1, allow_collisions=true)

Contruct a graph (of size `N`) with power-law degree distribution (``P(k)~k^{-γ}``) using
the configuration model, with minimal degree `min_d`.

# Examples
```jldoctest
julia> SF_configuration_model(4, 2.3; min_d=3, allow_collisions=false)
{4, 6} undirected simple Int64 graph
julia> SF_configuration_model(4, 40)
{4, 2} undirected simple Int64 graph
```

See also [`configuration_model`](@ref)
"""
function SF_configuration_model(N, γ; min_d=1, allow_collisions=true)
    v = min_d:(N-1)
    v = v .^ (-1.0 * γ)
    v ./= sum(v)

    dist = Categorical(v)
    samples = zeros(Int64, N)
    samples[1] = 1
    while !isgraphical(samples)
        samples = rand(dist, N) .+ (min_d - 1)
    end
    return configuration_model(samples; allow_collisions=allow_collisions)
end

"""
    spatial_network(ps::Matrix; f=x -> x)

Generate a spatial network with `size(ps, 1)` nodes embedded in ``[0,1]`` `^size(ps,2)`.
Edge ``(i,j)`` is present with probability `1-f(norm(ps[i,:] - ps[j,:])/norm(ones(size(ps,2))))`

See also [`Graphs.SimpleGraphs.euclidean_graph`](https://juliagraphs.org/Graphs.jl/dev/core_functions/simplegraphs_generators/)
"""
function spatial_network(ps::Matrix; f=x -> x)
    N = size(ps, 1)
    g = Graph(N)
    maxd = norm(ones(size(ps, 2)))
    for (i1, r1) in enumerate(eachrow(ps))
        for i2 in i1+1:N
            r2 = ps[i2, :]
            d = norm(r1 - r2) / maxd
            if rand() > f(d)
                add_edge!(g, i1, i2)
            end
        end
    end
    return g
end

"""
    g, ps = spatial_network(N[, d=2; f=x -> x])

Generate a spatial network `g` with `N` nodes embedded in ``[0,1]^d``.
`ps` contain the generated points on which the edges are based.

See also [`Graphs.SimpleGraphs.euclidean_graph`](https://juliagraphs.org/Graphs.jl/dev/core_functions/simplegraphs_generators/)
"""
function spatial_network(N, d=2; f=x -> x)
    ps = rand(N, d)
    return spatial_network(ps; f=f), ps
end

"""
    g = permuted_circle(n, nb_permutations)

Generate a graph with `n` nodes on a circle (1D periodic grid), i.e. each node is connected to two other nodes.
However, the nodes are permuted randomly `nb_permutations` times, i.e the ordering of the vertices does
not correspond to the ordering on the circle anymore.

# Examples
```jldoctest
julia> g = permuted_circle(4, 1)
{4, 4} undirected simple Int64 graph
```
"""
function permuted_circle(n, nb_permutations)
    adj_matrix = adjacency_matrix(grid((n,); periodic=true))
    for _ in 1:nb_permutations
        node1 = rand(1:n)
        node2 = rand(1:n)
        adj_matrix[:, [node1, node2]] .= adj_matrix[:, [node2, node1]]
        adj_matrix[[node1, node2], :] .= adj_matrix[[node2, node1], :]
    end
    return Graph(Symmetric(adj_matrix))
end

"""
    make_connected!(g::AbstractGraph)

Ensure `g` only consists of one component by connecting the largest component
to each of the other components with one random edge.

See also [`Graphs.connected_components`](https://juliagraphs.org/Graphs.jl/dev/algorithms/connectivity/)
"""
function make_connected!(g::AbstractGraph)
    comps = connected_components(g)
    gcc = comps[argmax(length.(comps))]
    for c in comps
        if c != gcc
            add_edge!(g, rand(c), rand(gcc))
        end
    end
end


######################
# Degree distribution
######################

"""
    degree_distribution(g::Graph; degree_list=nothing)

Calculate the degree distribution of a given graph `g` as a tuple
`(degree_list, degree_counts)`, such that `degree_counts[i]` is the number of
nodes with degree `degree_list[i]`. `degree_list` can be given as an argument,
in that case only the degrees in `degree_list` are considered.

# Examples
```jldoctest
julia> get_degree_distribution(grid((3, 3, 3), periodic=true))
([6,], [27,])
```

"""
function degree_distribution(g::Graph; degree_list=nothing)
    degrees = degree(g)
    if isempty(degrees)
        return [], []
    end
    if isnothing(degree_list)
        degree_list = collect(minimum(degrees):maximum(degrees))
    end
    degree_counts = [count(x -> x == d, degrees) for d in degree_list]
    return degree_list, degree_counts
end

"""

    conditional_degree_distribution(g::AbstractGraph)

Calculate the conditional degree distribution of a given graph `g`. The 
conditional degree distribution is a matrix `P` where each element `P[i,j]` 
represents the probability that a node with degree `i` is connected to a node 
with degree `j`.
"""
function conditional_degree_distribution(g::AbstractGraph)
    d = degree(g)

    P = zeros(Float64, (maximum(d), maximum(d)))
    for i in 1:nv(g)
        di = d[i]
        for j in neighbors(g, i)
            P[di, d[j]] += 1
        end
    end
    for d in unique(d)
        P[d, :] /= sum(P[d, :])
    end
    return P
end

"""
    combined_degree_distribution(dds::Vector{Tuple{Vector{Int}, Vector{Int}}})

Combine the degree distributions of multiple graphs into one. The input is a
vector of tuples, where each tuple contains a degree list and a count list.
"""
function combined_degree_distribution(dds::Vector{Tuple{Vector{Int},Vector{Int}}})
    degrees = [dd[1] for dd in dds]
    counts = [dd[2] for dd in dds]

    min_degree = minimum([minimum(d) for d in degrees])
    max_degree = maximum([maximum(d) for d in degrees])
    all_degrees = collect(min_degree:max_degree)
    aggregated_counts = zeros(Int, length(all_degrees))

    for i in eachindex(dds)
        for j in eachindex(all_degrees)
            if all_degrees[j] in degrees[i]
                aggregated_counts[j] += counts[i][findfirst(x -> x == all_degrees[j], degrees[i])]
            end
        end
    end
    return all_degrees, aggregated_counts
end

"""
    combined_degree_distribution(gs::AbstractGraph...)

Combine the degree distributions of multiple graphs into one.
"""
function combined_degree_distribution(gs::AbstractGraph...)
    return combined_degree_distribution([degree_distribution(g) for g in gs])
end

"""
    joint_distribution(graphs::AbstractGraph...; count_condition = i -> true)

Calculate the joint degree distribution P ∈ N^{nv(gs[1])^{length(gs)}} for a two layer
multiplex network, with each `g` ∈ `gs` being a layer. The joint degree distribution
is a matrix `P` where each element `P[i1, i2, ...]` represents the number of nodes
with degree `i1` in the first layer, `i2` in the second layer, and so on. Only vertices
for which `count_condition(i)` is `true` are considered.
"""
function joint_distribution(graphs::AbstractGraph...; count_condition=i -> true)
    degrees = [degree(g) for g in graphs]
    max_degrees = [maximum(k) for k in degrees]
    n = nv(graphs[1])
    @assert all(nv(g) == n for g in graphs)
    P = zeros(Int, max_degrees...)
    for i in 1:n
        if count_condition(i)
            idx = [k[i] for k in degrees]
            P[idx...] += 1
        end
    end
    return P
end

#################
# Graph analysis
#################
function expand_neighbors(vertices::Set, g, neighborfn; to_ignore=Set())
    expansion = Set()
    new_neighbors = Set([
        n for v in vertices for n in neighborfn(g, v)
        if !(n in to_ignore)
    ])
    while !isempty(new_neighbors)
        union!(expansion, new_neighbors)
        new_neighbors = Set([
            n for v in new_neighbors for n in neighborfn(g, v)
            if !(n in expansion) && !(n in to_ignore)
        ])
    end
    return expansion
end

"""
    bowtie_decomposition(g; full=false)

Perform a bowtie decomposition on the directed graph `g`.

# Returns
- gscc::Set: The largest strongly connected component.
- ginc::Set: All vertices upstream from the largest strongly connected component.
- goutc::Set: All vertices downstream from the largest strongly connected component.
Additionally, if `full=true`, the following sets are also returned:
- tubes::Set: Connections from `ginc` to `goutc` that are not part of `gscc`.
- outtendrils::Set: Vertices downstream from `ginc` that are not part of `gscc` or `tubes`.
- intendrils::Set: Vertices upstream from `goutc` that are not part of `gscc` or `tubes`.
"""
function bowtie_decomposition(g; full=false)
    @assert is_directed(g)
    # Find the largest strongly connected component
    sccs = strongly_connected_components(g)
    gscc = Set(sccs[argmax(length.(sccs))])
    assigned = deepcopy(gscc)
    # Find all vertices upstream from it
    ginc = expand_neighbors(gscc, g, inneighbors; to_ignore=assigned)
    # Find all vertices downstream from it
    goutc = expand_neighbors(gscc, g, outneighbors; to_ignore=assigned)

    if !full
        return gscc, ginc, goutc
    else
        union!(assigned, ginc, goutc)
        outtendrils = expand_neighbors(ginc, g, outneighbors; to_ignore=assigned)
        intendrils = expand_neighbors(goutc, g, inneighbors; to_ignore=assigned)
        tubes = intersect(outtendrils, intendrils)
        outtendrils = setdiff(outtendrils, tubes)
        intendrils = setdiff(intendrils, tubes)
        return gscc, ginc, goutc, tubes, outtendrils, intendrils
    end
end

############
# Graph IO
############

"""
    append_mtx(filename)

Add the extension `.mtx` to `filename` if it does not already have it.
"""
function append_mtx(filename)
    if length(filename) < 4 || filename[end-3:end] != ".mtx"
        filename = filename * ".mtx"
    end
    return filename
end

"""
    read_from_mtx(filename)

Construct a simple graph from a adjacency matrix stored in matrix market format.

See also [`write_to_mtx`](@ref)
"""
function read_from_mtx(filename)
    filename = append_mtx(filename)
    return SimpleGraph(mmread(filename))
end

"""
    write_to_mtx(filename)

Store graph `g` in a matrix market format under `filename`.

See also [`read_from_mtx`](@ref)
"""
function write_to_mtx(filename, g::AbstractGraph)
    filename = append_mtx(filename)
    mmwrite(filename, adjacency_matrix(g))
end

"""
    read_from_tsv(filename; N::Int64=typemax(Int64), directed=false)

Construct a simple graph from a tsv file. The file should contain two columns
with the source and destination of each edge.

# Arguments
- `filename`: The path to the file
- `N`: The number of lines to read. If `N` is smaller than the number of lines
  in the file, only the first `N` lines are read.
- `directed`: If `true`, the graph is directed, otherwise it is undirected.

# Examples
```jldoctest
julia> g = read_from_tsv("test.tsv")
{4, 3} undirected simple Int64 graph

julia> g = read_from_tsv("test.tsv"; N=2, directed=true)
{4, 2} directed simple Int64 graph
```
with `test.tsv`:
```
1	4
4   1
2	4
3	4
```
"""
function read_from_tsv(filename; N::Int64=typemax(Int64), directed=false)
    graph = directed ? DiGraph() : Graph()

    for (num, line) in enumerate(eachline(filename))
        line = split(line, '\t')

        try
            src = parse(Int64, line[1])
            dst = parse(Int64, line[2])

            if nv(graph) < max(src, dst)
                add_vertices!(graph, max(src, dst) - nv(graph))
            end
            add_edge!(graph, src, dst)
        catch
            @warn "Could not parse line $num"
            continue
        end

        if num >= N
            return graph
        end
    end
    return graph
end

#######################
# Simplicial complexes
#######################
"""
    find_r(ps::Matrix{Float64}, k::Int, eps::Float64=1e-6)

Find the smallest radius `r` such that each of the points `ps` has distance less
than `r` to at least `k` other points.
"""
function find_r(ps::Matrix{Float64}, k::Int, eps::Float64=1e-6)
    r_min = 0.0
    r_max = maximum(norm, eachcol(ps))
    while r_max - r_min > eps
        r = (r_min + r_max) / 2
        counts = [count(p -> norm(p - q) <= r, eachcol(ps)) - 1 for q in eachcol(ps)]
        if all(counts .>= k)
            r_max = r
        else
            r_min = r
        end
    end
    return r_max
end

"""
    vietoris_rips(points, epsilon)

Construct the Vietoris-Rips complex of a set of points in Euclidean space.

# Arguments
- `points`: A matrix of size `d` x `n` where `d` is the dimension of the
  Euclidean space and `n` is the number of points.
- `epsilon`: The radius of the balls used to construct the complex. If `epsilon`
  is negative, it will be interpreted as `Inf`.
"""
function vietoris_rips(points, epsilon)
    gg, _ = euclidean_graph(points; cutoff=epsilon)
    return SimplicialComplex(maximal_cliques(gg))
end

"""
    vietoris_rips_mink(n; d=2, min_k=2)

Construct the Vietoris-Rips complex of `n` random points in `d`-dimensional
Euclidean space. The radius of the balls is chosen such each point is at least
part of a `min_k` dimensional complex.

Current implementation and possibly idea do not make much sense!
"""
function vietoris_rips_mink(n; d=2, min_k=2)
    ps = rand(d, n)
    r = find_r(ps, min_k)
    return vietoris_rips(ps, r)
end

"""
    random_simplicial_complex(n, p::Vector)

Construct a random simplicial complex on `n` vertices. The probability of
including a simplex of dimension `k` is `p[k]`, if all of its facets are in the complex.
The function will give an AssertionError, if the graph is not connected.
"""
function random_simplicial_complex(n, p::Vector; connected=true)
    g = erdos_renyi(n, p[1])
    max_k = length(p)
    @assert is_connected(g) "Graph is not connected"
    full_complex = SimplicialComplex(maximal_cliques(g))
    all_simplices = [[] for _ in 1:max_k]
    for simplex in full_complex
        k = length(simplex) - 1
        if 2 <= k <= max_k
            push!(all_simplices[k], simplex)
        end
    end
    simplices = [Array{Set{Int64}}(undef, 0) for _ in 1:max_k]
    simplices[1] = [Set([e.src, e.dst]) for e in edges(g)]
    for k in 2:max_k
        for s in all_simplices[k]
            # Check if all facets are already in the complex
            for facet in combinations(collect(s), k)
                if !(Set(facet) in simplices[k-1])
                    @goto next_simplex
                end
            end
            if rand() < p[k]
                push!(simplices[k], s)
            end
            @label next_simplex
        end
    end
    return SimplicialComplex(vcat(simplices...))
end
end # end module general
