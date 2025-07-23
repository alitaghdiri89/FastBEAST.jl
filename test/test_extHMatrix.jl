include("../ext/HMatrix/HMatrix.jl")
using .ℌ
using CompScienceMeshes
using BEAST
using LinearAlgebra
using Plots

Γ = meshsphere(1.0, 0.2) 
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)
@time A = BEAST.assemble(op, spaceX, spaceX)
@time hmat = ℌ.assemble(op, spaceX, spaceX)

# Relative error and relative tolerance
rtols = [10.0^i for i in collect(-4:-1:-10)]# 1e-4 to 1e-10
tst_vec = rand(dim)
true_result = A * tst_vec
# using built-in ACA
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
ylabel!("relative error")
