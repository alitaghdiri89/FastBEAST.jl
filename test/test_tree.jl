using FastBEAST
using ClusterTrees
using StaticArrays
using Test

#BoxTree
points = [SVector(1.0, 1.0, 1.0), SVector(0.5, 0.5, 0.5)]

tree = create_tree(points, FastBEAST.KMeansTreeOptions())
nears, fars = computeinteractions(ClusterTrees.BlockTrees.BlockTree(tree, tree))
@test length(fars[2]) == 2
##

# KMeansTree
points = [@SVector rand(3) for i in 1:1000]

tree = create_tree(points, FastBEAST.KMeansTreeOptions(; nmin=5))
nears_strong, fars_strong = computeinteractions(
    ClusterTrees.BlockTrees.BlockTree(tree, tree); η=1.0
)
nears_weak, fars_weak = computeinteractions(
    ClusterTrees.BlockTrees.BlockTree(tree, tree); η=2.0
)

@test length(fars_weak[6]) > length(fars_strong[6])

tree = create_tree(points, FastBEAST.BoxTreeOptions(; nmin=5))
nears_strong, fars_strong = computeinteractions(
    ClusterTrees.BlockTrees.BlockTree(tree, tree); η=1.0
)
nears_weak, fars_weak = computeinteractions(
    ClusterTrees.BlockTrees.BlockTree(tree, tree); η=2.0
)
@test length(fars_weak[3]) > length(fars_strong[3])
