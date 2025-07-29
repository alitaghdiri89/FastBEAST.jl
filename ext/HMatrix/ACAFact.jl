using HMatrices
using ACAFact

struct LeafBlock <:AbstractMatrix{Float64}
    fullMat :: InterfaceAbstractMatrix
    irange :: Vector{Int64}
    jrange :: Vector{Int64}
end

function LeafBlock(K :: HMatrices.PermutedMatrix, irange :: UnitRange{Int64}, jrange :: UnitRange{Int64})
    permuted_irange = K.rowperm[Vector(irange)]
    permuted_jrange = K.colperm[Vector(jrange)]
    return LeafBlock(K.data, permuted_irange, permuted_jrange)
end

Base.getindex(K::LeafBlock, i::Int, j::Int) = K.fullMat[K.irange[i], K.jrange[j]]
Base.size(K::LeafBlock) = length(K.irange), length(K.jrange)

Base.@kwdef struct ACAFactWrapper
    atol::Float64 = 0
    rank::Int = typemax(Int)
    rtol::Float64 = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))
end


function (wrapper::ACAFactWrapper)(K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    irange = HMatrices.index_range(rowtree)
    jrange = HMatrices.index_range(coltree)
    leaf = LeafBlock(K, irange, jrange)
    @time (U, V) = aca(leaf, wrapper.rtol)
    return HMatrices.RkMatrix(U, V)
end

function ACAFact.row!(buff, K::LeafBlock, i)
    reshaped_buff = reshape(buff, (1, size(K, 2)))
    K.fullMat.blockassembler(reshaped_buff, [K.irange[i]], K.jrange)
end
function ACAFact.col!(buff, K::LeafBlock, j)
    K.fullMat.blockassembler(buff, K.irange, [K.jrange[j]])
end