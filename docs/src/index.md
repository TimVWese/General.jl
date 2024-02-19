# General.jl documentation

```@contents
```

## Dynamical objects

```@docs
count_states(es, i, s)
Ie!(e, v_s, v_d, p, t)
IStaticEdge(dim)
SStaticEdge(selection; invalid=0)
majority_vertex(i::Integer, dim::Integer; S1=1, S2=2)
```

## Multiplex networks

```@docs
combine_graphs(gs::AbstractGraph...)
combine_graphs(v_f::Function, gs::AbstractGraph...)
```

## Network models

```@docs
configuration_model(degree_seq; allow_collisions=true, max_tries=1000)
SF_configuration_model(N, γ; min_d=1)
spatial_network(ps::Matrix; f=x -> x)
spatial_network(N, d=2; f=x -> x)
permuted_circle(n, nb_permutations)
make_connected!(g::AbstractGraph)
```

## Degree distributions

```@docs
degree_distribution(g::Graph; degree_list=nothing)
conditional_degree_distribution(g::AbstractGraph)
combined_degree_distribution(dds::Vector{Tuple{Vector{Int}, Vector{Int}}})
combined_degree_distribution(gs::AbstractGraph...)
joint_distribution(graphs::AbstractGraph...; count_condition = i -> true)
```

## Graph IO

```@docs
read_from_mtx(filename)
write_to_mtx(filename, g::AbstractGraph)
read_from_tsv(filename; N::Int64=typemax(Int64), directed=false)
```

## Simplicial complexes

```@docs
vietoris_rips(points, epsilon)
vietoris_rips_mink(n; d=2, min_k=2)
random_simplicial_complex(n, p::Vector; connected=true)
```
