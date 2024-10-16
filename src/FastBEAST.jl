module FastBEAST
using LinearAlgebra

include("../ext/NminClusterTrees/NminClusterTrees.jl")

include("clustertree/tree.jl")
include("clustertree/kmeanstree.jl")
include("clustertree/boxtree.jl")

include("aca/aca_utils.jl")
include("aca/pivoting.jl")
include("aca/convergence.jl")
include("aca/aca.jl")
include("skeletons.jl")
include("hmatrix.jl")
include("utils.jl")
include("fmm.jl")
include("beast.jl")
include("fmm/operators/FMMoperator.jl")

export KMeansTreeOptions
export BoxTreeOptions
export create_tree
export computeinteractions
export value

export aca, allocate_aca_memory
export LazyMatrix

export MatrixBlock, LowRankMatrix

export HMatrix
export buildhmatrix
export adjoint
export estimate_norm
export estimate_reldifference
export nnz
export compressionrate

export hassemble
export fmmassemble
end
