using FastBEAST
using LinearAlgebra
using StaticArrays
using Test

#BoxTree
points2D = [SVector(1.0, 1.0), #3
            SVector(2.0, 2.0), #1
            SVector(1.2, 1.7), #2
            SVector(1.7, 1.2), #4
            SVector(1.21, 1.7)] #2

root = create_tree(points2D, BoxTreeOptions())

@test length(root.children) == 4
@test FastBEAST.indices(root.children[1])[1] == 2
@test FastBEAST.indices(root.children[2])[1] == 3
@test FastBEAST.indices(root.children[2])[2] == 5
@test FastBEAST.indices(root.children[3])[1] == 1
@test FastBEAST.indices(root.children[4])[1] == 4

# KMeansTree
points2D = [SVector(0.1, 0.1), #1
            SVector(1.0, 1.0), #2
            SVector(0.2, 0.2), #3
            SVector(1.1, 1.1)] #4

root = create_tree(points2D, KMeansTreeOptions(algorithm=:naive))

@test length(root.children) == 2
@test FastBEAST.indices(root.children[1])[1] == 1
@test FastBEAST.indices(root.children[1])[2] == 3
@test FastBEAST.indices(root.children[2])[1] == 2
@test FastBEAST.indices(root.children[2])[2] == 4
@test root.children[1].radius ≈ norm([0.05 0.05]) atol=1e-15
