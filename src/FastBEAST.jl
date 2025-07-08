module FastBEAST
using LinearAlgebra
using BlockSparseMatrices
import BlockSparseMatrices.AbstractBlockMatrix

include("../ext/NminClusterTrees/NminClusterTrees.jl")
export NminTree

include("clustertree/tree.jl")
include("clustertree/kmeanstree.jl")
include("clustertree/boxtree.jl")

include("aca/aca_utils.jl")
include("aca/pivoting.jl")
include("aca/convergence.jl")
include("aca/aca.jl")
include("skeletons.jl")

include("utils/erroranalysis.jl")
include("utils/functionkernels.jl")
include("hmatrix.jl")
include("fmm.jl")
include("beast.jl")
include("fmm/operators/FMMoperator.jl")
include("hmatrix/hmatrix.jl")

include("blockassembler/galerkinblkassembler.jl")
include("blockassembler/petrovgalerkinblkassembler.jl")

include("lowrankfactorization/lowrankfactorizations.jl")

export LRF
export HM
export KMeansTreeOptions
export BoxTreeOptions
export create_tree
export computeinteractions
export value

export aca, allocate_aca_memory
export LazyMatrix

export MatrixBlock, LowRankMatrix

export assemblefullblocks
export adjoint
export estimate_norm
export estimate_reldifference
export nnz
export compressionrate

export fmmassemble
end
