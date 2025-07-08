#=using LinearAlgebra
using LinearMaps=#
#=
struct HMatrix{I,K} <: LinearMaps.LinearMap{K}
    fullrankblocks::Vector{MatrixBlock{I,K,Matrix{K}}}
    lowrankblocks::Vector{MatrixBlock{I,K,LowRankMatrix{K}}}
    rowdim::I
    columndim::I
    nnz::I
    maxrank::I
    ismultithreaded::Bool
end

function nnzs(hmat::HT) where {HT<:HMatrix}
    return hmat.nnz
end

function compressionrate(hmat::HT) where {HT<:HMatrix}
    fullsize = hmat.rowdim * hmat.columndim
    nonzero = 0
    for lrb in hmat.lowrankblocks
        nonzero += length(lrb.M.U) + length(lrb.M.V)
    end
    for frb in hmat.fullrankblocks
        nonzero += length(frb.M)
    end
    return nonzero / fullsize
end

# Returns storage in GB, assuming 64 bit representations.
function storage(hmat::HT) where {HT<:HMatrix}
    nonzero = 0
    for lrb in hmat.lowrankblocks
        nonzero += length(lrb.M.U) + length(lrb.M.V)
    end
    for frb in hmat.fullrankblocks
        nonzero += length(frb.M)
    end
    return (nonzero * 64) / 8 * 10^-9
end

function ismultithreaded(hmat::HT) where {HT<:HMatrix}
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
            mul!(c[1:size(mb.M, 1)], mb.M, x[mb.σ])
            y[mb.τ] .+= c[1:size(mb.M, 1)]
        end

        for mb in A.lowrankblocks
            mul!(c[1:size(mb.M, 1)], mb.M, x[mb.σ])
            y[mb.τ] .+= c[1:size(mb.M, 1)]
        end

    else
        cc = zeros(eltype(y), size(A, 1), Threads.nthreads())
        yy = zeros(eltype(y), size(A, 1), Threads.nthreads())

        Threads.@threads for mb in A.fullrankblocks
            mul!(cc[1:size(mb.M, 1), Threads.threadid()], mb.M, x[mb.σ])
            yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M, 1), Threads.threadid()]
        end

        Threads.@threads for mb in A.lowrankblocks
            mul!(cc[1:size(mb.M, 1), Threads.threadid()], mb.M, x[mb.σ])
            yy[mb.τ, Threads.threadid()] .+= cc[1:size(mb.M, 1), Threads.threadid()]
        end

        y[:] = sum(yy; dims=2)
    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, transA::LinearMaps.TransposeMap{<:Any,<:HMatrix}, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(transA.lmap)
        c = zeros(eltype(y), size(transA, 1))

        for mb in transA.lmap.fullrankblocks
            mul!(c[1:size(mb.M, 2)], transpose(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M, 2)]
        end

        for mb in transA.lmap.lowrankblocks
            mul!(c[1:size(mb.M, 2)], transpose(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M, 2)]
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

        y[:] = sum(yy; dims=2)
    end

    return y
end

@views function LinearAlgebra.mul!(
    y::AbstractVecOrMat, transA::LinearMaps.AdjointMap{<:Any,<:HMatrix}, x::AbstractVector
)
    LinearMaps.check_dim_mul(y, transA, x)

    fill!(y, zero(eltype(y)))

    if !ismultithreaded(transA.lmap)
        c = zeros(eltype(y), size(transA, 1))

        for mb in transA.lmap.fullrankblocks
            mul!(c[1:size(adjoint(mb.M), 1)], adjoint(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M, 2)]
        end

        for mb in transA.lmap.lowrankblocks
            mul!(c[1:size(adjoint(mb.M), 1)], adjoint(mb.M), x[mb.τ])
            y[mb.σ] .+= c[1:size(mb.M, 2)]
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

        y[:] = sum(yy; dims=2)
    end

    return y
end
=#
function getcompressedmatrix(
    matrixassembler::Function,
    testidcs::Vector{I},
    trialidcs::Vector{I},
    ::Type{I},
    ::Type{K},
    am::ACAGlobalMemory{I,F,K};
    compressor=ACAOptions(),
) where {I,F<:Real,K}
    lm = LazyMatrix(matrixassembler, testidcs, trialidcs, K)

    maxrank = compressor.maxrank
    maxrank == 0 && Int(round(length(lm.τ) * length(lm.σ) / (length(lm.τ) + length(lm.σ))))

    U, V = aca(
        lm,
        am;
        rowpivstrat=compressor.rowpivstrat,
        columnpivstrat=compressor.columnpivstrat,
        convcrit=compressor.convcrit,
        tol=compressor.tol,
        svdrecompress=compressor.svdrecompress,
        maxrank=maxrank,
    )

    mbl = MatrixBlock{I,K,LowRankMatrix{K}}(LowRankMatrix(U, V), testidcs, trialidcs)

    return mbl
end
