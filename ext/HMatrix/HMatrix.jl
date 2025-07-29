module ℌ

    include("HMatrices.jl")
    include("ACAFact.jl")
    include("AdaptiveCrossApproximation.jl")  
    
    export assemble
    export get_hmatrix_memory_size
    export ACAFactWrapper
    export AdaptiveCrossApproximationWrapper
end