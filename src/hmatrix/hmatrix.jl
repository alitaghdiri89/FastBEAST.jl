module HM
using BEAST
using ClusterTrees
using FLoops
using LinearMaps
using LinearAlgebra
using ThreadsX

using FastBEAST: FastBEAST

include("GalerkinHMatrix.jl")
include("PetrovGalerkinHMatrix.jl")

end
