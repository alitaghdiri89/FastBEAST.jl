using BEAST
using CompScienceMeshes
using HMatrices
#=using LinearAlgebra
using StaticArrays
using Plots
using DataStructures=#

struct InterfaceAbstractMatrix <:AbstractMatrix{Float64}
    operator::BEAST.HH3DSingleLayerFDBIO
    spaceA::BEAST.LagrangeBasis
    spaceB::BEAST.LagrangeBasis
    blockassembler::Function
end

function InterfaceAbstractMatrix(operator::BEAST.HH3DSingleLayerFDBIO, spaceA::BEAST.LagrangeBasis, spaceB::BEAST.LagrangeBasis)
    blkasm = BEAST.blockassembler(operator, spaceA, spaceB)    
    function blkassembler(Z, tdata, sdata)
        fill!(Z, 0.0)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end
    return InterfaceAbstractMatrix(operator, spaceA, spaceB, blkassembler)
end

function Base.getindex(K::InterfaceAbstractMatrix, i::Int, j::Int)
    blkasm = BEAST.blockassembler(K.operator, K.spaceA, K.spaceB)    
    function blkassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end
    blk = zeros(Float64, 1, 1)
    blkassembler(blk, [i], [j])
    return blk[1, 1]
end

Base.size(K::InterfaceAbstractMatrix) = length(K.spaceA.pos), length(K.spaceB.pos)
function HMatrices.getblock!(
    out, K::HMatrices.PermutedMatrix{TT, T}#=InterfaceAbstractMatrix=#, irange_, jrange_
 ) where {TT <: InterfaceAbstractMatrix, T}
    irange = rangeConverter(irange_, K)
    jrange = rangeConverter(jrange_, K)
    permuted_irange = K.rowperm[Vector(irange)]
    permuted_jrange = K.colperm[Vector(jrange)]
    K.data.blockassembler(out, permuted_irange, permuted_jrange)
    return out
end

function rangeConverter(range_, K)
    if range_ isa Colon
        range = axes(K, 1)
    elseif range_ isa Int
        range = [range_]
    else
        range = Vector(range_)
    end
    return range
end


function assemble(op, spaceA, spaceB;  kwargs...)
    abstract_matrix = InterfaceAbstractMatrix(operator, spaceA, spaceB)
    splitter_nmax = haskey(kwargs, :splitter_nmax) ? kwargs[:splitter_nmax] : 50
    Xclt = ClusterTree(spaceA.pos, HMatrices.GeometricSplitter(splitter_nmax))
    Yclt = ClusterTree(spaceB.pos, HMatrices.GeometricSplitter(splitter_nmax))
    rtol = haskey(kwargs, :rtol) ? kwargs[:rtol] : 1e-4
    comp = haskey(kwargs, :comp) ? kwargs[:comp] : PartialACA(; rtol=rtol)
    hmatrix = assemble_hmatrix(abstract_matrix, Xclt, Yclt, comp=comp, Threads=false)
    return hmatrix
end

function get_hmatrix_memory_size(H)
    """
    Here we traverse the HMatrix tree using a DFS algorithm.
    Wherever we encounter data calculate its size and add it to the total size in bytes.
    data_size = length(data) * element_memory_size
    """
    element_count = 0
    stack = Stack{HMatrix}()
    push!(stack, H)
    while !isempty(stack)
        top = pop!(stack)
        if !isnothing(top.data)
            if isa(top.data, HMatrices.RkMatrix)
                element_count += length(top.data.A) + length(top.data.B)
            else
                element_count += length(top.data)
            end
        end
        for child in top.children
            push!(stack, child)
        end
    end
    return element_count * sizeof(eltype(H))
end