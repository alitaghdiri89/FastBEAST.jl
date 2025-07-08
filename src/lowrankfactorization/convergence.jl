#################
#   ConvCrit    #
#################

abstract type ConvCrit end

function normF(
    normUV²::F,
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
) where {F<:Real,K}
    @views normUV² +=
        (norm(rowbuffer[npivot, 1:maxcolumns]) * norm(colbuffer[1:maxrows, npivot]))^2

    for j in 1:(npivot - 1)
        @views normUV² +=
            2 * real.(
                dot(colbuffer[1:maxrows, npivot], colbuffer[1:maxrows, j]) *
                dot(rowbuffer[npivot, 1:maxcolumns], rowbuffer[j, 1:maxcolumns]),
            )
    end

    return normUV²
end

#Default

mutable struct Default{F} <: ConvCrit
    normUV²::F
end

(::Default{F})(kwargs...) where {F} = Default(0.0)

function (convcrit::Default{F})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
    tol::F,
) where {F<:Real,K}
    rnorm = norm(rowbuffer[npivot, 1:maxcolumns])
    cnorm = norm(colbuffer[1:maxrows, npivot])

    @views if isapprox(rnorm, 0.0) || isapprox(cnorm, 0.0)
        if isapprox(rnorm, 0.0) && isapprox(cnorm, 0.0)
            return npivot - 1, false
        else
            return npivot - 1, true
        end
    else
        convcrit.normUV² = normF(
            convcrit.normUV², rowbuffer, colbuffer, npivot, maxrows, maxcolumns
        )
        return npivot, rnorm * cnorm > tol * sqrt(convcrit.normUV²)
    end
end

#RandomSamples

mutable struct RandomSamples{F<:Real,K} <: ConvCrit
    nsamples::Int
    factor::F
    rows::Vector{Int}
    cols::Vector{Int}
    e::Vector{K}
    normUV²::F
end

function RandomSamples(::Type{K}; nsamples=0, factor=1.0) where {K}
    return RandomSamples(nsamples, factor, Int[], Int[], K[], 0.0)
end

function (convcrit::RandomSamples{F,K})(M::LazyMatrix{Int,K}) where {F,K}
    nsamples = convcrit.nsamples
    if convcrit.nsamples == 0
        nsamples = Int(round((size(M, 1) + size(M, 2)) * factor))
    end
    rows = rand(1:size(M, 1), convcrit.nsamples)
    cols = rand(1:size(M, 2), convcrit.nsamples)
    e = zeros(K, nsamples)
    for i in eachindex(rows)
        M.μ(e[i], rows[i], cols[i])
    end

    return RandomSamples(nsamples, convcrit.factor, rows, cols, e, 0.0)
end

function (convcrit::RandomSamples{F,K})(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
    tol::F,
) where {F<:Real,K}
    for i in eachindex(convcrit.e)
        @views convcrit.e[i] -=
            colbuffer[convcrit.rows[i], npivot] * rowbuffer[npivot, convcrit.cols[i]]
    end

    meanrest = sum(abs.(convcrit.e) .^ 2) / convcrit.nsamples

    convcrit.normUV² = normF(
        convcrit.normUV², rowbuffer, colbuffer, npivot, maxrows, maxcolumns
    )

    return sqrt(meanrest * maxrows * maxcolumns) <= tol * sqrt(convcrit.normUV²)
end

# Combined

mutable struct Combined <: ConvCrit
    critA::ConvCrit
    critB::ConvCrit
end

function (convcrit::Combined)(M::LazyMatrix{Int,K}) where {K}
    return Combined(convcrit.critA(M), convcrit.critB(M))
end

function (convcrit::Combined)(
    rowbuffer::AbstractMatrix{K},
    colbuffer::AbstractMatrix{K},
    npivot::Int,
    maxrows::Int,
    maxcolumns::Int,
    tol::F,
) where {F<:Real,K}
    return convcrit.critA(rowbuffer, colbuffer, npivot, maxrows, maxcolumns, tol) &&
           convcrit.critB(rowbuffer, colbuffer, npivot, maxrows, maxcolumns, tol)
end

#################
#   PivConvCit  #
#################

abstract type PivConvCit end
