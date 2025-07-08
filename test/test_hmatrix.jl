using FastBEAST
using LinearAlgebra
using Random
using StaticArrays
using ClusterTrees
using Test
Random.seed!(1)

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where {T}
    if isapprox(testpoint, sourcepoint; rtol=eps()*1e-4)
        return T(0.0)
    else
        return T(1.0) / (norm(testpoint - sourcepoint)) +
               T(1.0) / (norm(testpoint - sourcepoint))*im
    end
end

N = 1000

op = FastBEAST.KernelFunction{ComplexF64}(OneoverRkernel)
space = FastBEAST.PointSpace{Float64}([@SVector rand(3) for i in 1:N])

kmat = zeros(ComplexF64, N, N)
for (j, s) in enumerate(space.pos)
    for (i, t) in enumerate(space.pos)
        kmat[i, j] = op.fct(t, s)
    end
end
tree = FastBEAST.create_tree(space.pos, FastBEAST.KMeansTreeOptions(; nmin=50))

@time hmat = HM.assemble(op, space, space, testtree=tree, trialtree=tree, η=2.0);

x = rand(ComplexF64, N)
@test size(hmat, 1) == size(kmat, 1)
@test size(hmat, 2) == size(kmat, 2)
@test size(hmat) == size(kmat)
@test norm(hmat*x - kmat*x)/norm(kmat*x) ≈ 0 atol=1e-4
@test norm(adjoint(hmat)*x - adjoint(kmat)*x)/norm(adjoint(kmat)*x) ≈ 0 atol=1e-4
@test norm(transpose(hmat)*x - transpose(kmat)*x)/norm(transpose(kmat)*x) ≈ 0 atol=1e-4

##

@time hmat = HM.assemble(op, space, tree=tree, η=2.0);

@test size(hmat, 1) == size(kmat, 1)
@test size(hmat, 2) == size(kmat, 2)
@test size(hmat) == size(kmat)
@test norm(hmat*x - kmat*x)/norm(kmat*x) ≈ 0 atol=1e-4
@test norm(adjoint(hmat)*x - adjoint(kmat)*x)/norm(adjoint(kmat)*x) ≈ 0 atol=1e-4
@test norm(transpose(hmat)*x - transpose(kmat)*x)/norm(transpose(kmat)*x) ≈ 0 atol=1e-4
