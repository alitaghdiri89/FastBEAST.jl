include("../ext/HMatrix/HMatrix.jl")
using .ℌ
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Plots
using HMatrices
using Test

Γ = meshsphere(1.0, 0.2) 
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)

rtols = [10.0^i for i in collect(-4:-1:-10)]# 1e-4 to 1e-10
tst_vec = rand(dim)

fullmat = BEAST.assemble(op, spaceX, spaceX)
trueResult = fullmat * tst_vec

for rtol in rtols
    hmat_builtin = ℌ.assemble(op, spaceX, spaceX; rtol=rtol)
    @test size(hmat_builtin, 1) == size(fullmat, 1)
    @test size(hmat_builtin, 2) == size(fullmat, 2)
    @test norm(hmat_builtin * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol

    hmat_ACAFact = ℌ.assemble(op, spaceX, spaceX; comp=ACAFactWrapper(rtol=rtol))
    @test size(hmat_ACAFact, 1) == size(fullmat, 1)
    @test size(hmat_ACAFact, 2) == size(fullmat, 2)
    @test norm(hmat_ACAFact * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol

    hmat_Adaptive = ℌ.assemble(op, spaceX, spaceX; comp=AdaptiveCrossApproximationWrapper(rtol=rtol))
    @test size(hmat_Adaptive, 1) == size(fullmat, 1)
    @test size(hmat_Adaptive, 2) == size(fullmat, 2)
    @test norm(hmat_Adaptive * tst_vec - trueResult) / norm(trueResult) ≈ 0 atol = rtol
end