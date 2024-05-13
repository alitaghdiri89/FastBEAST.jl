using FastBEAST
using LinearAlgebra
using Random
using StaticArrays
using Test
Random.seed!(1)

function OneoverRkernel(testpoint::SVector{3,T}, sourcepoint::SVector{3,T}) where T
    if isapprox(testpoint, sourcepoint, rtol=eps()*1e-4)
        return T(0.0)
    else
        return T(1.0) / (norm(testpoint - sourcepoint))
    end
end

function assembler(kernel, testpoints, sourcepoints)
    kernelmatrix = zeros(promote_type(eltype(testpoints[1]),eltype(sourcepoints[1])), 
                length(testpoints), length(sourcepoints))

    for j in eachindex(sourcepoints)
        for i in eachindex(testpoints)
            kernelmatrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
    return kernelmatrix
end


function assembler(kernel, matrix, testpoints, sourcepoints)
    for j in eachindex(sourcepoints)
        for i in eachindex(testpoints)
            matrix[i,j] = kernel(testpoints[i], sourcepoints[j])
        end
    end
end

N = 1000

spoints = [@SVector rand(3) for i = 1:N]
tpoints = spoints
for tpoints in [spoints]
    @views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
        OneoverRkernel, matrix, tpoints[tdata], spoints[sdata]
    )
    stree = create_tree(spoints, BoxTreeOptions(nmin=25))
    ttree = create_tree(tpoints, BoxTreeOptions(nmin=25))
    kmat = assembler(OneoverRkernel, tpoints, spoints)
    for multithreading in [true, false]
        hmat = HMatrix(
            OneoverRkernelassembler,
            ttree,
            stree,
            Int64,
            Float64;
            compressor=FastBEAST.ACAOptions(tol=1e-4),
            multithreading=multithreading
        )

        x = rand(N)
        if tpoints != spoints
            @test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
        end

        @test norm(hmat*x - kmat*x)/norm(kmat*x) ≈ 0 atol=1e-4
    end
end
