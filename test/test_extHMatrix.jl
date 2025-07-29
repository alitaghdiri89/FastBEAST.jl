include("../ext/HMatrix/HMatrix.jl")
using .ℌ
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Plots
using HMatrices

Γ = meshsphere(1.0, 0.2) 
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)

mat = ℌ.InterfaceAbstractMatrix(op, spaceX, spaceX)
Xclt = ClusterTree(spaceX.pos, HMatrices.GeometricSplitter(50))
Yclt = ClusterTree(spaceX.pos, HMatrices.GeometricSplitter(50))



@time A = BEAST.assemble(op, spaceX, spaceX)
@time hmat = ℌ.assemble(op, spaceX, spaceX)
comp = ℌ.AdaptiveCrossApproximationWrapper(rtol=1e-4)
hmat = ℌ.assemble(op, spaceX, spaceX; comp=comp)
hmat = ℌ.assemble(op, spaceX, spaceX; comp=ℌ.ACAFactWrapper(rtol=1e-4))
##
# Relative error and relative tolerance
rtols = [10.0^i for i in collect(-4:-1:-10)]# 1e-4 to 1e-10
tst_vec = rand(dim)
true_result = A * tst_vec
# using built-in ACA
"""
errors_builtin_aca = zeros(Float64, size(rtols))
for (i, rtol) in enumerate(rtols)
    hmat = ℌ.assemble(op, spaceX, spaceX; rtol=rtol)
    errors_builtin_aca[i] = norm(hmat * tst_vec - true_result) / norm(true_result)
end
plot(rtols, errors_builtin_aca)
scatter!(rtols, errors_builtin_aca)
plot!(xscale=:log10, yscale=:log10, minorgrid=true)
plot!(legend=false, size=(800, 600))
xlabel!("relative tolerance")
ylabel!("relative error")

# using ACAFact

errors_ACAFact = zeros(Float64, size(rtols))
for (i, rtol) in enumerate(rtols)
    hmat = ℌ.assemble(op, spaceX, spaceX; comp=ℌ.ACAFactWrapper(rtol=rtol))
    errors_ACAFact[i] = norm(hmat * tst_vec - true_result) / norm(true_result)
end
plot(rtols, errors_ACAFact)
scatter!(rtols, errors_ACAFact)
plot!(xscale=:log10, yscale=:log10, minorgrid=true)
plot!(legend=false, size=(800, 600))
xlabel!("relative tolerance")
ylabel!("relative error")"""

# using AdaptiveCrossApproximation
errors_AdaptiveCrossApproximation = zeros(Float64, size(rtols))
for (i, rtol) in enumerate(rtols)
    hmat = ℌ.assemble(op, spaceX, spaceX; comp=ℌ.AdaptiveCrossApproximationWrapper(rtol=rtol))
    errors_AdaptiveCrossApproximation[i] = norm(hmat * tst_vec - true_result) / norm(true_result)
end
##
plot(rtols, errors_AdaptiveCrossApproximation)
scatter!(rtols, errors_AdaptiveCrossApproximation)
plot!(xscale=:log10, yscale=:log10, minorgrid=true)
plot!(legend=false, size=(800, 600))
xlabel!("relative tolerance")
ylabel!("relative error")
png("AdaptiveCrossApproximation_relerr")