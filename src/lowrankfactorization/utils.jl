struct LazyMatrix{I,F} <: AbstractMatrix{F}
    μ::Function
    τ::Vector{I}
    σ::Vector{I}
end

Base.size(A::LazyMatrix) = (length(A.τ), length(A.σ))

function Base.getindex(A::T, I, J) where {K,F,T<:LazyMatrix{K,F}}
    Z = zeros(F, length(I), length(J))
    A.μ(Z, view(A.τ, I), view(A.σ, J))
    return Z
end

function LazyMatrix(μ::Function, τ::Vector{I}, σ::Vector{I}, ::Type{F}) where {I,F}
    return LazyMatrix{I,F}(μ, τ, σ)
end

@views function (A::LazyMatrix{K,F})(Z::S, I, J) where {K,F,S<:AbstractMatrix{F}}
    return A.μ(view(Z, I, J), view(A.τ, I), view(A.σ, J))
end
