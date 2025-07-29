using AdaptiveCrossApproximation
using HMatrices

Base.@kwdef struct AdaptiveCrossApproximationWrapper
    atol::Float64 = 0.0
    rank::Int = typemax(Int)
    rtol::Float64 = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))
end

"""function AdaptiveCrossApproximationWrapper(matDim1::Int, matDim2::Int ;atol::Float64=0.0, rank::Int=40, rtol_::Float64=Nothing)
    aca = AdaptiveCrossApproximation.ACA(
        rowpivoting=AdaptiveCrossApproximation.MaximumValue(zeros(Bool, matDim1)),
        columnpivoting=AdaptiveCrossApproximation.MaximumValue(zeros(Bool, matDim2))
        )
        rtol = isnothing(rtol_) ? (atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))) : rtol_
        return AdaptiveCrossApproximationWrapper(aca, atol, rank, rtol)

    
end"""

function (paca::AdaptiveCrossApproximationWrapper)(
    K::AbstractMatrix{T},
    rowtree::ClusterTree,
    coltree::ClusterTree,
    bufs = nothing
    ) where{T} 
    irange = HMatrices.index_range(rowtree)
    jrange = HMatrices.index_range(coltree)
    maxrank = min(length(irange), length(jrange))
    rowbuffer = zeros(T, maxrank, length(jrange))
    colbuffer = zeros(T, length(irange) , maxrank)
    aca = AdaptiveCrossApproximation.ACA(
        rowpivoting=AdaptiveCrossApproximation.MaximumValue(zeros(Bool, length(irange))),
        columnpivoting=AdaptiveCrossApproximation.MaximumValue(zeros(Bool, length(jrange)))
        )
    npivots, U, V = aca(K, rowbuffer, colbuffer, maxrank, paca.rtol; rowidcs=irange, colidcs=jrange)
    return HMatrices.RkMatrix(colbuffer[:, 1:npivots], Matrix(transpose(rowbuffer[1:npivots, :])))
end


#= struct LeafBlockAdaptiveCrossApproximation <:AbstractMatrix{Float64}
    fullMat :: InterfaceAbstractMatrix
    irange :: Vector{Int64}
    jrange :: Vector{Int64}
end

function LeafBlockAdaptiveCrossApproximation(
    K :: HMatrices.PermutedMatrix, irange :: UnitRange{Int64}, jrange :: UnitRange{Int64})
    permuted_irange = K.rowperm[Vector(irange)]
    permuted_jrange = K.colperm[Vector(jrange)]
    return LeafBlockAdaptiveCrossApproximation(K.data, permuted_irange, permuted_jrange)
end

Base.getindex(K::LeafBlockAdaptiveCrossApproximation, i::Int, j::Int) = K.fullMat[K.irange[i], K.jrange[j]]
Base.size(K::LeafBlockAdaptiveCrossApproximation) = length(K.irange), length(K.jrange)

Base.@kwdef struct AdaptiveCrossApproximationWrapper
    atol::Float64 = 0
    rank::Int = typemax(Int)
    rtol::Float64 = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))
end


function (wrapper::AdaptiveCrossApproximationWrapper)(
    K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    irange = HMatrices.index_range(rowtree)
    jrange = HMatrices.index_range(coltree)
    leaf = LeafBlock(K, irange, jrange)
    (U, V) = AdaptiveCrossApproximation.aca(leaf; maxrank=wrapper.rank, tol=wrapper.rtol)
    return HMatrices.RkMatrix(U, V)
end =#

