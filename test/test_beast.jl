using BEAST
using CompScienceMeshes
using FastBEAST
using LinearAlgebra
using MKL
using StaticArrays
using Test

λ = 10
k = 2 * π / λ

Γs = meshrectangle(1.0,1.0, 0.3)

meshes = [
    (Γs, translate(Γs, SVector(3.0, 0.0, 1.0))),
    (Γs, translate(Γs, SVector(1.0, 0.0, 0.5)))
]

for mesh in meshes
    X = raviartthomas(mesh[1])
    Y = raviartthomas(mesh[2])
    SL = Maxwell3D.singlelayer(wavenumber=k)
    mat = assemble(SL, Y, X)
    for multithreading in [true, false]
        hmat = hassemble(
            SL, 
            Y,
            X;
            treeoptions=BoxTreeOptions(nmin=20),
            compressor=FastBEAST.ACAOptions(tol=1e-4),
            multithreading= multithreading
        )

        for matop in [x -> x, x -> transpose(x), x -> adjoint(x)]
            x = rand(size(matop(hmat), 2))
            yt = matop(hmat)*x
            yl = matop(mat)*x
            @test norm(yt - yl)/norm(yl) ≈ 0 atol=1e-4
        end
    end
end
