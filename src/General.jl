module General

export count_states
export Ie!
export IStaticEdge, SStaticEdge
export combine_graphs
export majority_vertex, majority_termination_cb
export SF_configuration_model
export spatial_network
export read_from_mtx, write_to_mtx
export make_connected!


using DifferentialEquations: DiscreteCallback, terminate!
using Distributions: Categorical
using Graphs
using MatrixMarket
using NetworkDynamics: StaticEdge, ODEVertex
using LinearAlgebra: norm

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

"""
    static_edges, combined_graph = combine_graphs(gs::AbstractGraph...)

Generate the `combined_graph` which edges are given by the union of the edges of `gs`.
The goal of this function is offer an interface that allows to use multiplex networks in
`network_dynamics`. Each argument represents a single layer.
The array `static_edges` consists of `SStaticEdge`s with `selection==[edge ∈ edges(g) for g in gs]`.

See also [`SStaticEdge`](@ref)
"""
function combine_graphs(gs::AbstractGraph...)
    dim = nv(gs[1])
    combined_graph = Graph(dim)
    for g in gs
        @assert dim == nv(g) "Graphs have different number of vertices"
        for edge in edges(g)
            add_edge!(combined_graph, edge)
        end
    end
    static_edges = Array{StaticEdge}(undef, ne(combined_graph))
    for (i, edge) in enumerate(edges(combined_graph))
        static_edges[i] = SStaticEdge([edge ∈ edges(g) for g in gs])
    end

    return static_edges, combined_graph
end

"""
    ode_vertices, static_edges, combined_graph =
        combine_graphs(v_f::Function, gs::AbstractGraph...)

Generate the `combined_graph` which edges are given by the union of the edges of `gs`.
The goal of this function is offer an interface that allows to use multiplex networks in
`network_dynamics`. Each argument represents a single layer.
The array `static_edges` consists of `SStaticEdge`s with `selection==[edge ∈ edges(g) for g in gs]`.
This method ought to bes used when the vertex dynamics depend on the degree in each layer.

# Arguments
- `v_f::Function`: a constructor `ds -> ODEVertex`, where ds[i] is the degree of the corresponding
vertex in `gs[i]`.
- gs::AbstractGraph...: graphs to combine

See also [`SStaticEdge`](@ref)
"""
function combine_graphs(v_f::Function, gs::AbstractGraph...)
    static_edges, combined_graph = combine_graphs(gs...)
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

"""
    SF_configuration_model(N, γ; min_d=1)

Contruct a graph (of size `N`) with power-law degree distribution (``P(k)~k^{-γ}``) using
the configuration model, with minimal degree `min_d`

# Examples
```jldoctest
julia> SF_configuration_model(4, 2.3; min_d=3)
{4, 6} undirected simple Int64 graph
julia> SF_configuration_model(4, 40)
{4, 2} undirected simple Int64 graph
```

See also [`random_configuration_model`](@ref)
"""
function SF_configuration_model(N, γ; min_d=1)
    v = min_d:(N-1)
    v = v .^ (-1.0 * γ)
    v ./= sum(v)

    dist = Categorical(v)
    samples = zeros(Int64, N)
    samples[1] = 1
    while !isgraphical(samples)
        samples = rand(dist, N) .+ (min_d - 1)
    end
    return random_configuration_model(N, samples)
end

"""
    spatial_network(ps::Matrix; f=x -> x)

Generate a spatial network with `size(ps, 1)` nodes embedded in ``[0,1]`` `^size(ps,2)`.
Edge ``(i,j)`` is present with probability `1-f(norm(ps[i,:] - ps[j,:])/norm(ones(size(ps,2))))`

See also [`euclidean_graph`](@ref).
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

See also [`euclidean_graph`](@ref).
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

See also [`connected_components`](@ref)
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