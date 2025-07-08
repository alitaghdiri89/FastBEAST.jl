using BEAST
using CompScienceMeshes
using FastBEAST
using LinearAlgebra
using MKL
using StaticArrays
using Test

λ = 10
k = 2 * π / λ

Γs = meshrectangle(1.0, 1.0, 0.3)
Γt = translate(meshrectangle(1.0, 1.0, 0.3), SVector(3.0, 0.0, 1.0))

meshes = [
    (Γs, translate(Γs, SVector(3.0, 0.0, 1.0))), (Γs, translate(Γs, SVector(1.0, 0.0, 0.5)))
]

for mesh in meshes
    Xs = raviartthomas(mesh[1])
    Xt = raviartthomas(mesh[2])
    Yt = buffachristiansen(mesh[2])
    X1 = lagrangec0d1(mesh[1])
    Y1 = duallagrangec0d1(mesh[2])

    Os = [
        (Helmholtz3D.singlelayer(;), Y1, X1, false)
        (Helmholtz3D.singlelayer(;), Y1, X1, true)
        (Helmholtz3D.doublelayer(;), Y1, X1, false)
        (Helmholtz3D.doublelayer(;), Y1, X1, true)
        (Helmholtz3D.doublelayer_transposed(; wavenumber=k), Y1, X1, true)
        (Helmholtz3D.hypersingular(; wavenumber=k), Y1, X1, true)
        (Helmholtz3D.singlelayer(; alpha=30.0*im), Y1, X1, true)
        (Helmholtz3D.doublelayer(; alpha=30.0*im), Y1, X1, true)
        (Helmholtz3D.doublelayer_transposed(; alpha=30.0), Y1, X1, true)
        (Helmholtz3D.hypersingular(; alpha=30.0), Y1, X1, true)
        (Helmholtz3D.hypersingular(; alpha=0.0, beta=-100.0), Y1, X1, true)
        (Helmholtz3D.singlelayer(; gamma=3.0), Y1, X1, true)
        (Helmholtz3D.doublelayer(; gamma=3.0), Y1, X1, true)
        (Helmholtz3D.doublelayer_transposed(; gamma=3.0), Y1, X1, true)
        (Helmholtz3D.hypersingular(; gamma=3.0), Y1, X1, true)
        (Maxwell3D.singlelayer(; wavenumber=k), Xt, Xs, true)
        (Maxwell3D.doublelayer(; wavenumber=k), Yt, Xs, true)
    ]

    for (O, Y, X, threading) in Os
        Ofl = assemble(O, Y, X) # full
        Oft = fmmassemble(
            O,
            Y,
            X;
            treeoptions=FastBEAST.KMeansTreeOptions(; nmin=50),
            multithreading=threading,
            computetransposeadjoint=true,
        ) # fast

        for matop in [x -> x, x -> transpose(x), x -> adjoint(x)]
            xF32 = rand(ComplexF32, size(matop(Oft), 2))
            xF64 = rand(ComplexF64, size(matop(Oft), 2))
            ytF32 = matop(Oft)*xF32
            ylF32 = matop(Ofl)*xF32
            ytF64 = matop(Oft)*xF64
            ylF64 = matop(Ofl)*xF64
            @test size(matop(Oft))[1] == size(matop(Oft), 1)
            @test size(matop(Oft))[2] == size(matop(Oft), 2)
            @test eltype(ytF32) == promote_type(eltype(xF32), eltype(Oft))
            @test norm(ytF32 - ylF32)/norm(ylF32) ≈ 0 atol=2e-4
            @test eltype(ytF64) == promote_type(eltype(xF64), eltype(Oft))
            @test norm(ytF64 - ylF64)/norm(ylF64) ≈ 0 atol=2e-4
        end
    end
end
