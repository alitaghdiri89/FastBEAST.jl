using AdaptiveCrossApproximation
using HMatrices

Base.@kwdef struct AdaptiveCrossApproximationWrapper
    atol::Float64 = 0.0
    rank::Int = typemax(Int)
    rtol::Float64 = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))
end


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

function AdaptiveCrossApproximation.nextrc!(
    buf, A::HMatrices.PermutedMatrix{TT, T}, i, j
    ) where {TT <: InterfaceAbstractMatrix, T}
    """
    overloaded to use K.blockassembler instead of Base.getindex
    """
    permuted_irange = A.rowperm[Vector(i)]
    permuted_jrange = A.colperm[Vector(j)]
    A.data.blockassembler(buf, permuted_irange, permuted_jrange)
end
