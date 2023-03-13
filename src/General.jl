module General

export count_states
export Ie!
export IStaticEdge, SStaticEdge
export combine_graphs
export majority_vertex, majority_termination_cb
export configuration_model
export SF_configuration_model
export spatial_network
export read_from_mtx, write_to_mtx
export make_connected!
export quick_solve!


using DifferentialEquations: DiscreteCallback, terminate!
using Distributions: Categorical
using Graphs
using MatrixMarket
using NetworkDynamics: StaticEdge, ODEVertex
using LinearAlgebra: norm
using Random: shuffle!

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
    c = 0
    for e in es
        if e[i] == s
            c += 1
        end
    end
    return c
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
    dims = length(dims) == 1 ? dims[1]*ones(Int64, length(gs)) : dims
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

end # end module general