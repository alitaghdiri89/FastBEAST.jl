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
#=
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
            treeoptions=BoxTreeOptions(nmin=100),
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
=#
## FMM

for mesh in meshes
    X1 = lagrangec0d1(mesh[1])
    Y1 = duallagrangec0d1(mesh[2])

    Os = [
        (Helmholtz3D.singlelayer(; wavenumber=k), Y1, X1)
        (Helmholtz3D.doublelayer(; wavenumber=k), Y1, X1)
        (Helmholtz3D.doublelayer_transposed(; wavenumber=k), Y1, X1)
        (Helmholtz3D.hypersingular(; wavenumber=k), Y1, X1)
    ]

    for (O, Y1, X1) in Os
        Ofl = assemble(O, Y1, X1) # full
        for multithreading in [true, false]
            Oft = fmmassemble(
                O,
                Y1,
                X1;
                treeoptions=FastBEAST.KMeansTreeOptions(; nmin=50),
                multithreading=multithreading,
            ) # fast

            for matop in [x -> x]#, x -> transpose(x), x -> adjoint(x)]
                x = rand(size(matop(Oft), 2))
                yt = matop(Oft) * x
                yl = matop(Ofl) * x
                @test eltype(yt) == promote_type(eltype(x), eltype(Oft))
                @test norm(yt - yl) / norm(yl) ≈ 0 atol = 1e-4
            end
        end
    end
end
