using ClusterTrees
using ThreadsX

function assemble(
    op::BEAST.AbstractOperator,
    testspace::BEAST.Space,
    trialspace::BEAST.Space,
    tree::ClusterTrees.BlockTrees.BlockTree{T},
    interactions::Vector{Tuple{I, I}},
    ::Type{K};
    quadstrat=BEAST.defaultquadstrat(op, testspace, trialspace),
    multithreading=true
) where {I,K,T}

    @views nearblkassembler = BEAST.blockassembler(
        op, testspace, trialspace, quadstrat=quadstrat
    )
    @views function nearassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        nearblkassembler(tdata,sdata,store)
    end

    snears = Tuple{Int, Vector{Int}}[]
    for near in interactions
        if length(snears)!= 0 && snears[end][1] == near[1] 
            push!(snears[end][2], near[2])
        else
            push!(snears, (near[1], [near[2]]))
        end
    end

    _foreach = multithreading ? ThreadsX.foreach : Base.foreach
    blocks = Vector{BlockSparseMatrices.DenseMatrixBlock{K, Matrix{K}, Vector{I}}}(
        undef, length(snears)
    )
   
    _foreach(enumerate(snears)) do (idx, snear) 
        testidcs = value(tree.test_cluster, snear[1])
        trialidcs = value(tree.trial_cluster, snear[2])
        matrix = zeros(K, length(testidcs), length(trialidcs))
        assembler(matrix, testidcs, trialidcs)

        blocks[idx] =  BlockSparseMatrices.DenseMatrixBlock{K, Matrix{K}, Vector{Int}}(
            matrix,
            testidcs,
            trialidcs
        )
    end

    return BlockSparseMatrix(
        blocks, (tree.test_cluster.num_elements, tree.trial_cluster.num_elements)
    )
end