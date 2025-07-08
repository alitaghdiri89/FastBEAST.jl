struct PetrovGalerkinHMatrix{I,K,NearInteractionType,FarInteractionType} <:
       LinearMaps.LinearMap{K}
    nearinteractions::NearInteractionType
    farinteractions::FarInteractionType
    dim::Tuple{I,I}
    ismultithreaded::Bool

    function PetrovGalerkinHMatrix{I,K}(
        nearinteractions, farinteractions, dim, ismultithreaded
    ) where {I,K}
        return new{I,K,typeof(nearinteractions),typeof(farinteractions)}(
            nearinteractions, farinteractions, dim, ismultithreaded
        )
    end
end

function Base.size(A::PetrovGalerkinHMatrix, dim=nothing)
    if dim === nothing
        return (A.dim[1], A.dim[2])
    elseif dim == 1
        return A.dim[1]
    elseif dim == 2
        return A.dim[2]
    end
end

function PetrovGalerkinHMatrix(
    operator,
    testspace,
    trialspace;
    testtree=FastBEAST.create_tree(testspace.pos, FastBEAST.BoxTreeOptions(nmin=50)),
    trialtree=FastBEAST.create_tree(trialspace.pos, FastBEAST.BoxTreeOptions(nmin=50)),
    η=1.0,
    nearquadstrat=BEAST.defaultquadstrat(operator, testspace, trialspace),
    farquadstrat=BEAST.DoubleNumQStrat(2, 3),
    compressor=FastBEAST.ACAOptions(; tol=1e-4),
    multithreading=true,
)
    blktree = ClusterTrees.BlockTrees.BlockTree(testtree, trialtree)
    nears, fars = FastBEAST.computeinteractions(blktree; η=η)
    nearinteractions = FastBEAST.assemble(
        operator,
        testspace,
        trialspace,
        blktree,
        nears,
        scalartype(operator);
        quadstrat=nearquadstrat,
        multithreading=multithreading,
    )

    fars = reduce(vcat, fars)
    _foreach = multithreading ? ThreadsX.foreach : Base.foreach

    @views farasm = BEAST.blockassembler(
        operator, testspace, trialspace, quadstrat=farquadstrat
    )
    @views function farassembler(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return farasm(tdata, sdata, store)
    end

    am = FastBEAST.allocate_aca_memory(
        scalartype(operator),
        testtree.num_elements,
        trialtree.num_elements,
        multithreading;
        maxrank=compressor.maxrank,
    )
    farinteractions = Vector{
        FastBEAST.MatrixBlock{
            Int,scalartype(operator),FastBEAST.LowRankMatrix{scalartype(operator)}
        },
    }(
        undef, length(fars)
    )

    _foreach(enumerate(fars)) do (idx, far)
        return farinteractions[idx] = FastBEAST.getcompressedmatrix(
            farassembler,
            FastBEAST.value(testtree, far[1]),
            FastBEAST.value(trialtree, far[2]),
            Int,
            scalartype(operator),
            am[Threads.threadid()];
            compressor=compressor,
        )
    end

    return PetrovGalerkinHMatrix{Int,scalartype(operator)}(
        nearinteractions,
        farinteractions,
        (testtree.num_elements, trialtree.num_elements),
        multithreading,
    )
end

function assemble(operator, testspace, trialspace; kwargs...)
    return PetrovGalerkinHMatrix(operator, testspace, trialspace; kwargs...)
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {M<:PetrovGalerkinHMatrix}
    fill!(y, zero(eltype(y)))

    mul!(y, A.nearinteractions, x)
    for lrb in A.farinteractions
        y[lrb.τ] += lrb.M * x[lrb.σ]
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {T,Z<:PetrovGalerkinHMatrix,M<:LinearMaps.AdjointMap{T,Z}}
    fill!(y, zero(eltype(y)))

    mul!(y, adjoint(A.lmap.nearinteractions), x)
    for lrb in A.lmap.farinteractions
        y[lrb.σ] += adjoint(lrb.M) * x[lrb.τ]
    end

    return y
end

function LinearMaps._unsafe_mul!(
    y::AbstractVector, A::M, x::AbstractVector
) where {T,Z<:PetrovGalerkinHMatrix,M<:LinearMaps.TransposeMap{T,Z}}
    fill!(y, zero(eltype(y)))

    mul!(y, transpose(A.lmap.nearinteractions), x)
    for lrb in A.lmap.farinteractions
        y[lrb.σ] += transpose(lrb.M) * x[lrb.τ]
    end

    return y
end

function storage(hmat::PetrovGalerkinHMatrix)
    ref = size(hmat, 1) * size(hmat, 2)
    hstor = 0.0
    for frb in hmat.nearinteractions.blocks
        hstor += length(frb.rowindices) * length(frb.colindices)
    end

    for far in hmat.farinteractions
        hstor += length(far.M.U) + length(far.M.V)
    end

    return hstor * 8 * 10^-9, hstor / ref
end

function lrbmat(hmat::PetrovGalerkinHMatrix{I,K,NT,FT}) where {I,K,NT,FT}
    A = zeros(K, size(hmat, 1), size(hmat, 2))

    for far in hmat.farinteractions
        A[far.τ, far.σ] = far.M.U * far.M.V
    end
    return A
end

using BlockSparseMatrices

function lrbhmat(hmat::PetrovGalerkinHMatrix{I,K,NT,FT}) where {I,K,NT,FT}
    blocks = BlockSparseMatrices.DenseMatrixBlock{K,Matrix{K},Vector{I}}[]
    nears = BlockSparseMatrix(blocks, hmat.dim)
    return PetrovGalerkinHMatrix{Int,K}(
        nears, hmat.farinteractions, hmat.dim, hmat.ismultithreaded
    )
end

function fullmat(hmat::PetrovGalerkinHMatrix{I,K,NT,FT}) where {I,K,NT,FT}
    A = zeros(K, size(hmat, 1), size(hmat, 2))

    for M in hmat.nearinteractions.blocks
        A[M.rowindices, M.colindices] = M.matrix
    end

    for far in hmat.farinteractions
        A[far.τ, far.σ] = far.M.U * far.M.V
    end
    return A
end
