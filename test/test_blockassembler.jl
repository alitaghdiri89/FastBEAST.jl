using ClusterTrees
using FastBEAST
using BEAST
using LinearAlgebra
using CompScienceMeshes
using BlockSparseMatrices

Γ = meshicosphere(40, 1.0)
λ = 2
k = 2 * pi / λ
op = Maxwell3D.singlelayer(; wavenumber=k)
space = raviartthomas(Γ)
##
tree = create_tree(space.pos, KMeansTreeOptions(; nmin=100, maxlevel=50))
blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
nears, fars = FastBEAST.computeinteractions(blktree; η=1.0)
##
@time nearinteractions = FastBEAST.assemble(op, space, space, blktree, nears, ComplexF64);

@time snearinteractions = FastBEAST.assemble(op, space, blktree, nears, ComplexF64);
##
x = rand(ComplexF64, size(nearinteractions, 2))
y = zeros(ComplexF64, size(nearinteractions, 1))
y2 = zeros(ComplexF64, size(nearinteractions, 1))

##
println("Threads=1")
mul!(y, nearinteractions, x);
@time mul!(y, nearinteractions, x);

println("Threads=1")
mul!(y2, snearinteractions, x);
@time mul!(y2, snearinteractions, x);
##
norm(y - y2) / norm(y)
