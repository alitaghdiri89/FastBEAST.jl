using ClusterTrees
using LinearAlgebra

function create_tree(
    points::Vector{SVector{D, F}},
    options::BoxTreeOptions{I}
) where {D, I, F <: Real}

    hs, ct = NminClusterTrees.NminTrees.boundingbox(points)
    tree = NminClusterTrees.NminTrees.NminTree(length(points), center=ct, radius=F(sqrt(D))*hs)

    D == 3 ? nchildren=8 : nchildren=4
    treeoptions = NminClusterTrees.NminTrees.BoxTreeOptions(dim=D)
    destination = (options.nmin, options.maxlevel)
    state = (1, ct, F(sqrt(D))*hs, 1, 1, Vector(1:length(points)), points)

    NminClusterTrees.NminTrees.child!(tree, treeoptions, state, destination)

    return tree
end
