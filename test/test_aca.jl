using FastBEAST
using LinearAlgebra
using Random
using StaticArrays
using Test
Random.seed!(10)

@views function fct(B, x, y)
    return B[:, :] = A[x, y]
end

N = 100
A = rand(N, N)

U, S, V = svd(A)
S = [i < 15 ? 10.0^(-i) : 0.0 for i in 1:N]
A = U * diagm(S) * V'

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)
##
U, V = aca(lm)
@test U * V ≈ A atol = 1e-14

U, V = aca(lm; tol=10^-4)
@test U * V ≈ A atol = 1e-4

N = 100
spoints = [@SVector rand(3) for i in 1:N]
tpoints = [SVector(rand(), rand(), rand()) + SVector(3.0, 0.0, 0.0) for i in 1:N]

spoints
A = zeros(Float64, N, N)
for i in 1:N
    for j in 1:N
        A[i, j] = 1 / norm(spoints[i] - tpoints[j])
    end
end

lm = LazyMatrix(fct, Vector(1:size(A, 1)), Vector(1:size(A, 2)), Float64)

U, V = aca(lm; maxrank=N)
@test norm(U * V - A) / norm(A) ≈ 1e-14 atol = 1e-14
U, V = aca(lm; tol=10^-4)
@test norm(U * V - A) / norm(A) ≈ 1e-4 atol = 1e-4

U, V = aca(lm; rowpivstrat=FastBEAST.FillDistance(spoints), maxrank=N)
@test norm(U * V - A) / norm(A) ≈ 1e-14 atol = 1e-14
U, V = aca(lm; rowpivstrat=FastBEAST.FillDistance(spoints), tol=10^-4)
@test norm(U * V - A) / norm(A) ≈ 1e-4 atol = 1e-4

U, V = aca(lm; rowpivstrat=FastBEAST.ModifiedFillDistance(spoints), maxrank=N)
@test norm(U * V - A) / norm(A) ≈ 1e-14 atol = 1e-14
U, V = aca(lm; rowpivstrat=FastBEAST.ModifiedFillDistance(spoints), tol=10^-4)
@test norm(U * V - A) / norm(A) ≈ 1e-4 atol = 1e-4

U, V = aca(
    lm;
    rowpivstrat=FastBEAST.EnforcedPivoting(spoints),
    convcrit=FastBEAST.Combined(Float64),
    maxrank=N,
)
@test norm(U * V - A) / norm(A) ≈ 1e-14 atol = 1e-14
U, V = aca(
    lm;
    rowpivstrat=FastBEAST.EnforcedPivoting(spoints),
    convcrit=FastBEAST.Combined(Float64),
    tol=1e-4,
    maxrank=N,
)
@test norm(U * V - A) / norm(A) ≈ 1e-4 atol = 1e-4

U, V = aca(lm; convcrit=FastBEAST.Combined(Float64), maxrank=N)
@test norm(U * V - A) / norm(A) ≈ 1e-14 atol = 1e-14
U, V = aca(lm; convcrit=FastBEAST.Combined(Float64), tol=1e-4, maxrank=N)
@test norm(U * V - A) / norm(A) ≈ 1e-4 atol = 1e-4

U, V = aca(lm; convcrit=FastBEAST.RandomSampling(Float64), maxrank=N)
@test norm(U * V - A) / norm(A) ≈ 1e-14 atol = 1e-14
U, V = aca(lm; convcrit=FastBEAST.RandomSampling(Float64), tol=1e-4, maxrank=N)
@test norm(U * V - A) / norm(A) ≈ 1e-4 atol = 1e-4
