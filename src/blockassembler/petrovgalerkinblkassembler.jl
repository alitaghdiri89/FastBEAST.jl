using ClusterTrees
using BlockSparseMatrices
using ThreadsX

function assemble(
    op::BEAST.AbstractOperator,
    testspace::BEAST.AbstractSpace,
    trialspace::BEAST.AbstractSpace,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    nears::Vector{Tuple{I,I}},
    ::Type{K};
    quadstrat=BEAST.defaultquadstrat(op, testspace, trialspace),
    multithreading=true,
) where {I,K,T}
    @views nearblkassembler = BEAST.blockassembler(
        op, testspace, trialspace; quadstrat=quadstrat
    )
    @views function nearassembler(Z, tdata, sdata)
        @views store(v, m, n) = (Z[m, n] += v)
        return nearblkassembler(tdata, sdata, store)
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    blocks = Vector{BlockSparseMatrices.DenseMatrixBlock{K,Matrix{K},Vector{I}}}(
        undef, length(nears)
    )

    _foreach(enumerate(nears)) do (idx, near)
        testidcs = value(tree.test_cluster, near[1])
        trialidcs = value(tree.trial_cluster, near[2])
        matrix = zeros(K, length(testidcs), length(trialidcs))
        nearassembler(matrix, testidcs, trialidcs)

        return blocks[idx] = BlockSparseMatrices.DenseMatrixBlock{K,Matrix{K},Vector{Int}}(
            matrix, testidcs, trialidcs
        )
    end

    return BlockSparseMatrix(
        blocks, (tree.test_cluster.num_elements, tree.trial_cluster.num_elements)
    )
end
