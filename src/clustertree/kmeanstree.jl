using ClusterTrees
using LinearAlgebra

function create_tree(
    points::Vector{SVector{D,F}}, options::KMeansTreeOptions{I}
) where {I,D,F<:Real}
    hs, ct = NminClusterTrees.boundingbox(points)
    pointsM = reshape([point[i] for point in points for i in 1:3], (3, length(points)))

    tree = NminClusterTrees.NminTree(length(points); center=ct, radius=F(sqrt(D))*hs)

    treeoptions = NminClusterTrees.KMeansTreeOptions(;
        nchildren=options.nchildren, settings=options.KMeansSettings
    )
    destination = (options.nmin, options.maxlevel)
    state = (1, ct, F(sqrt(D))*hs, 1, 1, Vector(1:length(points)), pointsM)

    NminClusterTrees.child!(tree, treeoptions, state, destination)

    return tree
end
