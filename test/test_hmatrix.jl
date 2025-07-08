using ClusterTrees
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

N = 2000

spoints = [@SVector rand(3) for i = 1:N]
tpoints = [@SVector rand(3) for i = 1:N] 

@views OneoverRkernelassembler(matrix, tdata, sdata) = assembler(
    OneoverRkernel, matrix, tpoints[tdata], spoints[sdata]
)

stree = create_tree(spoints, KMeansTreeOptions(nmin=20))
ttree = create_tree(tpoints, KMeansTreeOptions(nmin=20))

kmat = assembler(OneoverRkernel, tpoints, spoints)

for multithreading in [true, false]
    hmat = HMatrix(
        OneoverRkernelassembler,
        ClusterTrees.BlockTrees.BlockTree(ttree,stree),
        Int64,
        Float64;
        compressor=FastBEAST.ACAOptions(tol=1e-4),
        verbose=true,
        multithreading=multithreading, 
        η=1.5
    )

    x = rand(N)
    if tpoints != spoints
        @test estimate_reldifference(hmat,kmat) ≈ 0 atol=1e-4
    end

    @test norm(hmat*x - kmat*x)/norm(kmat*x) ≈ 0 atol=1e-4

    @test FastBEAST.storage(hmat) ≈ 
        compressionrate(hmat) * size(hmat, 1) * size(hmat, 2) * 8 * 1e-9
end
