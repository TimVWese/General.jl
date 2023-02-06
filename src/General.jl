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

function count_states(es, i, s)
    c = 0
    for e in es
        if e[i] == s
            c += 1
        end
    end
    return c
end

function Ie!(e, v_s, v_d, p, t)
    e .= v_s;
end

IStaticEdge = dim -> StaticEdge(f = Ie!, dim = dim, coupling=:undirected);

function SStaticEdge(selection; invalid=0)
    dim = length(selection)
    f = (e, v_s, v_d, p, t) -> begin
        for i in eachindex(v_s)
            e[i] = selection[i] ? v_s[i] : invalid
        end    
    end
    return StaticEdge(f = f, dim=dim, coupling=:undirected)
end

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

function combine_graphs(v_f::Function, gs::AbstractGraph...)
    static_edges, combined_graph = combine_graphs(gs...)
    ode_vertices = [v_f([degree(g, v) for g in gs]) for v in vertices(combined_graph)] 
    return ode_vertices, static_edges, combined_graph
end

function majority_vertex(i::Integer, dim::Integer; U=1, A=2)
    f = (vₙ, v, es, p, t) -> begin
        vₙ .= v;
        vₙ[i] = count_states(es, i, A)/length(es) >= p.Θ ? A : U
    end
    return ODEVertex(f = f, dim = dim);
end

function majority_termination(u, t, integrator)
    t < 3 ? false : u == integrator.sol(t-2)
end

majority_termination_cb = DiscreteCallback(majority_termination, terminate!)

function SF_configuration_model(N, γ; min_d=1)
    v = min_d:(N-1)
    v = v.^(-1.0*γ)
    v ./= sum(v)

    dist = Categorical(v)
    samples = zeros(Int64, N)
    samples[1] = 1
    while !isgraphical(samples)
        samples = rand(dist, N) .+ (min_d-1)
    end
    return random_configuration_model(N, samples)
end

function spatial_network(ps::Matrix; f=x->x)
    N = size(ps, 1)
    g = Graph(N)
    maxd = norm(ones(size(ps, 2)))
    for (i1, r1) in enumerate(eachrow(ps))
        for i2 in i1+1:N
            r2 = ps[i2,:]
            d = norm(r1 - r2)/maxd
            if rand() > f(d)
                add_edge!(g, i1, i2)
            end
        end
    end
    return g
end

function spatial_network(N, d=2; f=x->x)
    ps = rand(N, d)
    return spatial_network(ps; f=f), ps
end

function read_from_mtx(filename)
    if length(filename) < 4 || filename[end-3:end] != ".mtx"
        filename = filename*".mtx"
    end
    return SimpleGraph(mmread(filename))
end

function write_to_mtx(filename, g::AbstractGraph)
    if length(filename) < 4 || filename[end-3:end] != ".mtx"
        filename = filename*".mtx"
    end
    mmwrite(filename, adjacency_matrix(g))
end

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