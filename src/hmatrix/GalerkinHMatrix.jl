
struct GalerkinHMatrix{I,K,NearInteractionType,FarInteractionType} <:
       LinearMaps.LinearMap{K}
    nearinteractions::NearInteractionType
    farinteractions::FarInteractionType
    dim::Tuple{I,I}
    multithreading::Bool

    function GalerkinHMatrix{I,K}(
        nearinteractions, farinteractions, dim, multithreading
    ) where {I,K}
        return new{I,K,typeof(nearinteractions),typeof(farinteractions)}(
            nearinteractions, farinteractions, dim, multithreading
        )
    end
end

function Base.size(A::GalerkinHMatrix, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    end
end

function GalerkinHMatrix(
    operator,
    space;
    tree=FastBEAST.create_tree(space.pos, FastBEAST.BoxTreeOptions()),
    η=1.0,
    nearquadstrat=BEAST.defaultquadstrat(operator, space, space),
    farquadstrat=BEAST.DoubleNumQStrat(2, 3),
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
)
    blktree = ClusterTrees.BlockTrees.BlockTree(tree, tree)
    nears, fars = FastBEAST.computeinteractions(blktree; η=η)

    nearinteractions = FastBEAST.assemble(
        operator,
        space,
        blktree,
        nears,
        scalartype(operator);
        quadstrat=nearquadstrat,
        multithreading=multithreading,
    )

    fars = reduce(vcat, fars)
    sfars = eltype(fars)[]
    for far in fars
        if far[1] < far[2]
            push!(sfars, far)
        end
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    @views farasm = BEAST.blockassembler(operator, space, space, quadstrat=farquadstrat)
    @views function farassembler(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farasm(tdata, sdata, store)
    end
    am = FastBEAST.allocate_aca_memory(
        scalartype(operator),
        tree.num_elements,
        tree.num_elements,
        multithreading;
        maxrank=compressor.maxrank,
    )
    farinteractions = Vector{
        FastBEAST.MatrixBlock{
            Int,scalartype(operator),FastBEAST.LowRankMatrix{scalartype(operator)}
        },
    }(
        undef, length(sfars)
    )

    _foreach(enumerate(sfars)) do (idx, far)
        return farinteractions[idx] = FastBEAST.getcompressedmatrix(
            farassembler,
            FastBEAST.value(tree, far[1]),
            FastBEAST.value(tree, far[2]),
            Int,
            scalartype(operator),
            am[Threads.threadid()];
            compressor=compressor,
        )
    end

    return GalerkinHMatrix{Int,scalartype(operator)}(
        nearinteractions,
        farinteractions,
        (tree.num_elements, tree.num_elements),
        multithreading,
    )
end

function assemble(operator, space; kwargs...)
    return GalerkinHMatrix(operator, space; kwargs...)
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {M<:GalerkinHMatrix}
    fill!(y, zero(eltype(y)))

    mul!(y, A.nearinteractions, x)
    for lrb in A.farinteractions
        y[lrb.τ] += lrb.M * x[lrb.σ]
        y[lrb.σ] += transpose(lrb.M) * x[lrb.τ]
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {T,Z<:GalerkinHMatrix,M<:LinearMaps.TransposeMap{T,Z}}
    fill!(y, zero(eltype(y)))

    mul!(y, A.lmap.nearinteractions, x)
    for lrb in A.lmap.farinteractions
        y[lrb.τ] += lrb.M * x[lrb.σ]
        y[lrb.σ] += transpose(lrb.M) * x[lrb.τ]
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {T,Z<:GalerkinHMatrix,M<:LinearMaps.AdjointMap{T,Z}}
    fill!(y, zero(eltype(y)))

    mul!(y, adjoint(A.lmap.nearinteractions), x)
    for lrb in A.lmap.farinteractions
        @views y[lrb.τ] += transpose(adjoint(lrb.M)) * x[lrb.σ]
        @views y[lrb.σ] += adjoint(lrb.M) * x[lrb.τ]
    end

    return y
end

function storage(hmat::HM.GalerkinHMatrix)
    ref = size(hmat, 1) * size(hmat, 2)
    hstor = 0.0
    for frb in hmat.nearinteractions.diagonals
        hstor += length(frb.rowindices) * length(frb.colindices)
    end
    for frb in hmat.nearinteractions.offdiagonals
        hstor += length(frb.rowindices) * length(frb.colindices)
    end

    for far in hmat.farinteractions
        hstor += length(far.M.U) + length(far.M.V)
    end

    return hstor * 8 * 10^-9, hstor / ref
end
