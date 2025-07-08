using LinearAlgebra
using LinearMaps
using ProgressMeter

struct HMatrix{I, K} <: LinearMaps.LinearMap{K}
    fullrankblocks::Vector{MatrixBlock{I, K, Matrix{K}}}
    lowrankblocks::Vector{MatrixBlock{I, K, LowRankMatrix{K}}}
    rowdim::I
    columndim::I
    nnz::I
    maxrank::I
    ismultithreaded::Bool
end

function nnzs(hmat::HT) where HT <: HMatrix
    return hmat.nnz
end

function compressionrate(hmat::HT) where HT <: HMatrix
    fullsize = hmat.rowdim*hmat.columndim
    nonzero = 0
    for lrb in hmat.lowrankblocks
        nonzero += length(lrb.M.U) + length(lrb.M.V)
    end
    for frb in hmat.fullrankblocks
        nonzero += length(frb.M)
    end
    return nonzero/fullsize
end

# Returns storage in GB, assuming 64 bit representations.
function storage(hmat::HT) where HT <: HMatrix
    nonzero = 0
    for lrb in hmat.lowrankblocks
        nonzero += length(lrb.M.U) + length(lrb.M.V)
    end
    for frb in hmat.fullrankblocks
        nonzero += length(frb.M)
    end
    return (nonzero*64)/8 * 10^-9
end


function ismultithreaded(hmat::HT) where HT <: HMatrix
    return hmat.ismultithreaded
end

function Base.size(hmat::HMatrix, dim=nothing)
    if dim === nothing
        return (hmat.rowdim, hmat.columndim)
    elseif dim == 1
        return hmat.rowdim
    elseif dim == 2
        return hmat.columndim
    else
        error("dim must be either 1 or 2")
    end
end

#=
function Base.size(hmat::Adjoint{T}, dim=nothing) where T <: HMatrix
    if dim === nothing
        return reverse(size(adjoint(hmat)))
    elseif dim == 1
        return size(adjoint(hmat),2)
    elseif dim == 2
        return size(adjoint(hmat),1)
    else
        error("dim must be either 1 or 2")
    end
end=#

@views function LinearAlgebra.mul!(y::AbstractVecOrMat, A::HMatrix, x::AbstractVector)
    LinearMaps.check_dim_mul(y, A, x)

    fill!(y, zero(eltype(y)))
    
    if !ismultithreaded(A)

        c = zeros(eltype(y), size(A, 1))

        for mb in A.fullrankblocks
            mul!(c[1:size(mb.M,1)], mb.M, x[mb.σ])
            y[mb.τ] .+= c[1:size(mb.M,1)]
        end
        
        for mb in A.lowrankblocks
            mul!(c[1:size(mb.M, 1)], mb.M, x[mb.σ])
            y[mb.τ] .+= c[1:size(mb.M,1)]
        end

    else
        cc = zeros(eltype(y), size(A, 1), Threads.nthreads())
        yy = zeros(eltype(y), size(A, 1), Threads.nthreads())

        Threads.@threads for mb in A.fullrankblocks
            mul!(cc[1:size(mb.M,1), Threads.threadid()], mb.M, x[mb.σ])
            yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M,1), Threads.threadid()]
        end
        
        Threads.@threads for mb in A.lowrankblocks
            mul!(cc[1:size(mb.M, 1), Threads.threadid()], mb.M, x[mb.σ])
            yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M,1), Threads.threadid()]
        end

        y[:] = sum(yy, dims=2)
    end

    return y
end


@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.TransposeMap{<:Any,<:HMatrix},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(transA.lmap)

        c = zeros(eltype(y), size(transA,1))

        for mb in transA.lmap.fullrankblocks
            mul!(c[1:size(mb.M, 2)], transpose(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M, 2)]
        end

        for mb in transA.lmap.lowrankblocks
            mul!(c[1:size(mb.M,2)], transpose(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M,2)]
        end

    else

        cc = zeros(eltype(y), size(transA, 1), Threads.nthreads())
        yy = zeros(eltype(y), size(transA, 1), Threads.nthreads())

        Threads.@threads for mb in transA.lmap.fullrankblocks
            mul!(cc[1:size(mb.M, 2), Threads.threadid()], transpose(mb.M), x[mb.τ])
            yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        end
        
        Threads.@threads for mb in transA.lmap.lowrankblocks
            mul!(cc[1:size(mb.M, 2), Threads.threadid()], transpose(mb.M), x[mb.τ])
            yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        end

        y[:] = sum(yy, dims=2)

    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat,
    transA::LinearMaps.AdjointMap{<:Any,<:HMatrix},
    x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(transA.lmap)

        c = zeros(eltype(y), size(transA,1))

        for mb in transA.lmap.fullrankblocks
            mul!(c[1:size(adjoint(mb.M),1)], adjoint(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M, 2)]
        end

        for mb in transA.lmap.lowrankblocks
            mul!(c[1:size(adjoint(mb.M),1)], adjoint(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M,2)]
        end

    else

        cc = zeros(eltype(y), size(transA, 1), Threads.nthreads())
        yy = zeros(eltype(y), size(transA, 1), Threads.nthreads())

        Threads.@threads for mb in transA.lmap.fullrankblocks
            mul!(cc[1:size(mb.M, 2), Threads.threadid()], adjoint(mb.M), x[mb.τ])
            yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        end
        
        Threads.@threads for mb in transA.lmap.lowrankblocks
            mul!(cc[1:size(mb.M, 2), Threads.threadid()], adjoint(mb.M), x[mb.τ])
            yy[mb.σ, Threads.threadid()] .+= cc[1:size(mb.M, 2), Threads.threadid()]
        end

        y[:] = sum(yy, dims=2)

    end

    return y
end

function HMatrix(
    matrixassembler::Function,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    ::Type{I},
    ::Type{K};
    farmatrixassembler=matrixassembler,
    compressor=ACAOptions(),
    multithreading=false,
    verbose=false,
    η=0.5
) where {I, K, T}
    
    fullcompressableinteractions = SVector{2}[]
    fullinteractions, compressableinteractions = computeinteractions(tree, η=η)
    MBF = MatrixBlock{I, K, Matrix{K}}
    fullrankblocks_perthread = Vector{MBF}[]
    fullrankblocks = MBF[]

    rowdim = length(value(tree.test_cluster, 1))
    coldim = length(value(tree.trial_cluster, 1))

    if !multithreading
        am = allocate_aca_memory(K, rowdim, coldim, maxrank=compressor.maxrank)
    else
        ams = ACAGlobalMemory{I, real(K), K}[]
        for i in 1:Threads.nthreads()
            push!(ams, allocate_aca_memory(K, rowdim, coldim, maxrank=compressor.maxrank))
        end
    end

    verbose && println("Number of full interactions: ", length(fullinteractions))
    verbose && println(
        "Number of compressable interactions: ",
        length(compressableinteractions)
    )

    if verbose
        p = Progress(length(fullinteractions), desc="Computing full interactions: ")
    end

    if !multithreading
        for fullinteraction in fullinteractions
            push!(
                fullrankblocks,
                getfullmatrixview(
                    matrixassembler,
                    value(tree.test_cluster, fullinteraction[1]),
                    value(tree.trial_cluster, fullinteraction[2]),
                    I,
                    K
                )
            )
            verbose && next!(p)
        end
    elseif multithreading
        for i in 1:Threads.nthreads()
            push!(fullrankblocks_perthread, MBF[])
        end

        Threads.@threads for fullinteraction in fullinteractions
            push!(
                fullrankblocks_perthread[Threads.threadid()],
                getfullmatrixview(
                    matrixassembler,
                    value(tree.test_cluster, fullinteraction[1]),
                    value(tree.trial_cluster, fullinteraction[2]),
                    I,
                    K
                )
            )
            verbose && next!(p)
        end

        for i in eachindex(fullrankblocks_perthread)
            append!(fullrankblocks, fullrankblocks_perthread[i])
        end
    end

    MBL = MatrixBlock{I, K, LowRankMatrix{K}}
    lowrankblocks_perthread = Vector{MBL}[]
    lowrankblocks = MBL[]

    if verbose
        p = Progress(length(compressableinteractions), desc="Compressing far interactions: ")
    end

    if !multithreading
        for level in compressableinteractions
            for compressableinteraction in level
                push!(
                    lowrankblocks, 
                    getcompressedmatrix(
                        farmatrixassembler,
                        value(tree.test_cluster, compressableinteraction[1]),
                        value(tree.trial_cluster, compressableinteraction[2]),
                        I,
                        K,
                        am,
                        compressor=compressor
                    )
                )
                #nonzeros += nnzs(lowrankblocks[end])
                verbose && next!(p)
            end
        end
    elseif multithreading
        for i in 1:Threads.nthreads()
            push!(lowrankblocks_perthread, MBL[])
        end
        for level in compressableinteractions
            Threads.@threads for compressableinteraction in level
                push!(
                    lowrankblocks_perthread[Threads.threadid()],
                    getcompressedmatrix(
                        farmatrixassembler,
                        value(tree.test_cluster, compressableinteraction[1]),
                        value(tree.trial_cluster, compressableinteraction[2]),
                        I,
                        K,
                        ams[Threads.threadid()],
                        compressor=compressor
                    )
                )
                #nonzeros_perthread[Threads.threadid()] += 
                #    nnzs(lowrankblocks_perthread[Threads.threadid()][end])
                verbose && next!(p)
            end
        end

        for i in eachindex(lowrankblocks_perthread)
            append!(lowrankblocks, lowrankblocks_perthread[i])
        end
    end

    correctionindices = []
    for (ind, lbk) in enumerate(lowrankblocks)
        maxrowcols = Int(floor(length(lbk.τ)*length(lbk.σ)/(length(lbk.τ)+length(lbk.σ))))

        if maxrowcols <= size(lbk.M.U)[2] || size(lbk.M.U)[2] == compressor.maxrank
            push!(fullcompressableinteractions, SVector(lbk.τ, lbk.σ))
            push!(correctionindices, ind)
        end 
    end
    deleteat!(lowrankblocks, correctionindices)

    if verbose
        p = Progress(length(fullcompressableinteractions), desc="Correcting far interactions: ")
    end



    if !multithreading
        for fullcompressableinteraction in fullcompressableinteractions
            push!(
                fullrankblocks, 
                getfulllowrankmatrixview(
                    matrixassembler,
                    fullcompressableinteraction[1],
                    fullcompressableinteraction[2],
                    I,
                    K
                )
            )
            verbose && next!(p)
        end
    elseif multithreading

        fulllowrankblocks_perthread = Vector{MBF}[]

        for i in 1:Threads.nthreads()
            push!(fulllowrankblocks_perthread, MBL[])
        end

        Threads.@threads for fullcompressableinteraction in fullcompressableinteractions
            push!(
                fulllowrankblocks_perthread[Threads.threadid()],
                getfulllowrankmatrixview(
                    matrixassembler,
                    fullcompressableinteraction[1],
                    fullcompressableinteraction[2],
                    I,
                    K
                )
            )
            verbose && next!(p)
        end

        for i in eachindex(fulllowrankblocks_perthread)
            append!(fullrankblocks, fulllowrankblocks_perthread[i])
        end
    end

    return HMatrix{I, K}(
        fullrankblocks,
        lowrankblocks,
        rowdim,
        coldim,
        0,
        compressor.maxrank,
        multithreading
    )
end


function getfullmatrixview(
    matrixassembler,
    testidcs,
    trialidcs,
    ::Type{I},
    ::Type{K};
) where {I, K}
    matrix = zeros(K, length(testidcs), length(trialidcs))
    matrixassembler(matrix, testidcs, trialidcs)

    return MatrixBlock{I, K, Matrix{K}}(
        matrix,
        testidcs,
        trialidcs
    )
end

function getfulllowrankmatrixview(
    matrixassembler,
    testind,
    sourceind,
    ::Type{I},
    ::Type{K};
) where {I, K}
    matrix = zeros(K, length(testind), length(sourceind))
    matrixassembler(matrix, testind, sourceind)

    return MatrixBlock{I, K, Matrix{K}}(
        matrix,
        testind,
        sourceind
    )
end

function getcompressedmatrix(
    matrixassembler::Function,
    testidcs,
    trialidcs,
    ::Type{I},
    ::Type{K},
    am;
    compressor=ACAOptions()
) where {I, K}

        lm = LazyMatrix(matrixassembler, testidcs, trialidcs, K)

        maxrank = compressor.maxrank
        maxrank == 0 && Int(round(length(lm.τ)*length(lm.σ)/(length(lm.τ)+length(lm.σ))))

        U, V = aca(
            lm,
            am;
            rowpivstrat=compressor.rowpivstrat,
            columnpivstrat=compressor.columnpivstrat,
            convcrit=compressor.convcrit,
            tol=compressor.tol,
            svdrecompress=compressor.svdrecompress,
            maxrank= maxrank
        )

        mbl = MatrixBlock{I, K, LowRankMatrix{K}}(
            LowRankMatrix(U, V),
            testidcs,
            trialidcs
        )

    return mbl
end
