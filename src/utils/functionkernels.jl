using BEAST
using StaticArrays
using LinearAlgebra

# Allows to define functions as kernels and PointSpaces for testing

struct KernelFunction{T} <: BEAST.AbstractOperator
    fct::Function

    function KernelFunction{T}(fct) where {T}
        return new{T}(fct)
    end
end

struct PointSpace{F} <: BEAST.AbstractSpace
    pos::Vector{SVector{3,F}}

    function PointSpace{F}(pos::Vector{SVector{3,F}}) where {F}
        return new{F}(pos)
    end
end

function kernelassembler(
    kernelfct::KernelFunction{T},
    matrix,
    testpoints::Union{Array{SVector{3,Float64},0},Vector{SVector{3,F}}},
    trialpoints::Union{Array{SVector{3,Float64},0},Vector{SVector{3,F}}},
) where {F,T}
    for i in eachindex(testpoints)
        for j in eachindex(trialpoints)
            matrix(kernelfct.fct(testpoints[i], trialpoints[j]), i, j)
        end
    end
end

function BEAST.blockassembler(
    operator::KernelFunction{T},
    testspace::PointSpace{F},
    trialspace::PointSpace{F};
    quadstrat=1,
) where {F,T}
    blkassembler(tdata, sdata, matrix) = kernelassembler(
        operator, matrix, testspace.pos[tdata], trialspace.pos[sdata]
    )

    return blkassembler
end

BEAST.scalartype(f::KernelFunction{T}) where {T} = T
BEAST.defaultquadstrat(op::KernelFunction, tfs::PointSpace, bfs::PointSpace) = 1
