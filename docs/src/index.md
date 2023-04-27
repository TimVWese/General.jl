# General.jl documentation

```@docs
count_states(es, i, s)
```

```@docs
IStaticEdge(dim)
SStaticEdge(selection; invalid=0)
combine_graphs(gs::AbstractGraph...)
combine_graphs(v_f::Function, gs::AbstractGraph...)
majority_vertex(i::Integer, dim::Integer; S1=1, S2=2)
SF_configuration_model(N, γ; min_d=1)
spatial_network(ps::Matrix; f=x -> x)
spatial_network(N, d=2; f=x -> x)
conditional_degree_distribution(g::AbstractGraph)
read_from_mtx(filename)
write_to_mtx(filename, g::AbstractGraph)
make_connected!(g::AbstractGraph)
```
