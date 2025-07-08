abstract type PivStrat end

struct MaximumValue <: PivStrat
    usedidcs::Vector{Bool}
end

MaximumValue() = MaximumValue(Bool[])

(::MaximumValue)(rcp::Vector{Int}) = MaximumValue(zeros(Bool, length(rcp)))

function (pivstrat::MaximumValue)(rc::AbstractArray)
    nextidx = 1
    maxval = 0.0
    for i in eachindex(pivstrat.usedidcs)
        if (!pivstrat.usedidcs[i]) && abs(rc[i]) >= maxval
            nextidx = i
            maxval = abs(rc[i])
        end
    end

    pivstrat.usedidcs[nextidx] = true
    return nextidx
end

##

abstract type AbstractFillDistance <: PivStrat end

function filldistance!(fd::AbstractFillDistance, nextidx::Int)
    newh = norm.(fd.pos .- Scalar(fd.pos[nextidx]))
    for idx in eachindex(fd.h)
        fd.h[idx] > newh[idx] && (fd.h[idx] = newh[idx])
    end
end

mutable struct ModifiedFillDistance{D,F} <: AbstractFillDistance
    h::Vector{F}
    pos::Vector{SVector{D,F}}
    usedidcs::Vector{Bool}
end

function ModifiedFillDistance(pos::Vector{SVector{D,F}}) where {D,F}
    return ModifiedFillDistance(F[], pos, Bool[])
end

function (pivstrat::ModifiedFillDistance{D,F})(rcp::Vector{Int}) where {D,F}
    return ModifiedFillDistance(
        zeros(F, length(rcp)), pivstrat.pos[rcp], zeros(Bool, length(rcp))
    )
end

function (pivstrat::ModifiedFillDistance{D,F})(::AbstractArray) where {D,F}
    nextidx = argmax(h)
    filldistance!(pivstrat, nextidx)
    pivstrat.usedidcs[nextidx] = true

    return nextidx
end
