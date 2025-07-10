using BEAST
using CompScienceMeshes
using HMatrices
using LinearAlgebra
using StaticArrays
using Plots
using DataStructures

struct InterfaceAbstractMatrix <:AbstractMatrix{Float64}
    operator::BEAST.HH3DSingleLayerFDBIO
    spaceA::BEAST.LagrangeBasis
    spaceB::BEAST.LagrangeBasis
    blockassembler::Function
end

function InterfaceAbstractMatrix(operator::BEAST.HH3DSingleLayerFDBIO, spaceA::BEAST.LagrangeBasis, spaceB::BEAST.LagrangeBasis)
    blkasm = BEAST.blockassembler(operator, spaceA, spaceB)    
    function blkassembler(Z, tdata, sdata)
        fill!(Z, 0.0)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end
    return InterfaceAbstractMatrix(operator, spaceA, spaceB, blkassembler)
end

function Base.getindex(K::InterfaceAbstractMatrix, i::Int, j::Int)
    blkasm = BEAST.blockassembler(K.operator, K.spaceA, K.spaceB)    
    function blkassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end
    blk = zeros(Float64, 1, 1)
    blkassembler(blk, [i], [j])
    return blk[1, 1]
end

Base.size(K::InterfaceAbstractMatrix) = length(K.spaceA.pos), length(K.spaceB.pos)
 

# Overloading HMatrices.getblock! to work with InterfaceAbstractMatrix

function HMatrices.getblock!(
    out, K::HMatrices.PermutedMatrix{TT, T}#=InterfaceAbstractMatrix=#, irange_, jrange_
 ) where {TT <: InterfaceAbstractMatrix, T}
     if irange_ isa Colon
        irange = axes(K, 1)
    elseif irange_ isa Int
        irange = [irange_]
    else
        irange = Vector(irange_)
    end

    if jrange_ isa Colon
        jrange = axes(K, 2)
    elseif jrange_ isa Int
        jrange = [jrange_]
    else
        jrange = Vector(jrange_)
    end
    permuted_irange = K.rowperm[Vector(irange)]
    permuted_jrange = K.colperm[Vector(jrange)]
    K.data.blockassembler(out, permuted_irange, permuted_jrange)
    return out
end

function get_hmatrix_memory_size(H)
    """
    Here we traverse the HMatrix tree using a DFS algorithm.
    Wherever we encounter data calculate its size and add it to the total size in bytes.
    data_size = length(data) * element_memory_size
    """
    element_count = 0
    stack = Stack{HMatrix}()
    push!(stack, H)
    while !isempty(stack)
        top = pop!(stack)
        if !isnothing(top.data)
            if isa(top.data, HMatrices.RkMatrix)
                element_count += length(top.data.A) + length(top.data.B)
            else
                element_count += length(top.data)
            end
        end
        for child in top.children
            push!(stack, child)
        end
    end
    return element_count * sizeof(eltype(H))
end 
# The main interface function

function interface_to_hmatrix(operator, spaceA, spaceB; rtol=1e-6, splitter_nmax=50, comp=nothing)
    abstract_matrix = InterfaceAbstractMatrix(operator, spaceA, spaceB)
    Xclt = ClusterTree(spaceA.pos, HMatrices.GeometricSplitter(splitter_nmax))
    Yclt = ClusterTree(spaceB.pos, HMatrices.GeometricSplitter(splitter_nmax))
    comp = isnothing(comp) ? PartialACA(; rtol=rtol) : comp 
    h_matrix = assemble_hmatrix(abstract_matrix, Xclt, Yclt, comp=comp, threads =false)
    return h_matrix 
end

struct LeafBlock <:AbstractMatrix{Float64}
    fullMat :: InterfaceAbstractMatrix
    irange :: Vector{Int64}
    jrange :: Vector{Int64}
end

function LeafBlock(K :: HMatrices.PermutedMatrix, irange :: UnitRange{Int64}, jrange :: UnitRange{Int64})
    permuted_irange = K.rowperm[Vector(irange)]
    permuted_jrange = K.colperm[Vector(jrange)]
    return LeafBlock(K.data, permuted_irange, permuted_jrange)
end

Base.getindex(K::LeafBlock, i::Int, j::Int) = K.fullMat[K.irange[i], K.jrange[j]]
Base.size(K::LeafBlock) = length(K.irange), length(K.jrange)

using ACAFact
Base.@kwdef struct myACA
    atol::Float64 = 0
    rank::Int = typemax(Int)
    rtol::Float64 = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))
end


function (paca::myACA)(K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    irange = HMatrices.index_range(rowtree)
    jrange = HMatrices.index_range(coltree)
    leaf = LeafBlock(K, irange, jrange)
    (U, V) = aca(leaf, paca.rtol)
    return HMatrices.RkMatrix(U, V)
end

function ACAFact.row!(buff, K::LeafBlock, i)
    reshaped_buff = reshape(buff, (1, size(K, 2)))
    K.fullMat.blockassembler(reshaped_buff, [K.irange[i]], K.jrange)
end
function ACAFact.col!(buff, K::LeafBlock, j)
    K.fullMat.blockassembler(buff, K.irange, [K.jrange[j]])
end

##
## Testing
# setting up test example sephirical same space
Γ = meshsphere(1.0, 0.2) 
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)

@time A = assemble(op, spaceX, spaceX)
@time hmat = interface_to_hmatrix(op, spaceX, spaceX)
##
hmat
##
# test example, 2 rectangles one translated
op = Helmholtz3D.singlelayer()
Γa = meshrectangle(1.0, 1.0, 0.2)
Γb = CompScienceMeshes.translate(meshrectangle(1.0, 1.0, 0.02), SVector(3.0, 0.0, 0.0))
spaceA = lagrangecxd0(Γa)
spaceB = lagrangecxd0(Γb)
dim = length(spaceB.pos)
##
@time A = assemble(op, spaceA, spaceB)
hmat = interface_to_hmatrix(op, spaceA, spaceB; rtol=1e-6)
## 
println("comparing matrices themselves")
absolute_diff = norm(Matrix(hmat) - A)
println("absolute: ", absolute_diff)
println("relative: ", absolute_diff / norm(A))
##
println("Matrix vector product test")
vec = rand(dim)
absolute_err = norm(hmat * vec - A * vec)
println("absolute: ", absolute_err)
println("relative: ", absolute_err / norm(A * vec))
##
"""
Below this line are not important
****************************************************************************************************
"""

#=blkasm = BEAST.blockassembler(op, spaceX,spaceX)    
function blkassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end

function element(i, j)
    blk = zeros(ComplexF64, 1, 1)
    blkassembler(blk, [i], [j])
    return blk[1, 1]
end

refA = zeros(Float64, length(spaceX.pos), length(spaceX.pos))
@time for i = 1:length(spaceX.pos)
    for j=1:length(spaceX.pos)
        refA[i, j] = element(i, j)
    end
end

AR = hmat.data.A
BRT = hmat.data.Bt
mulmat = HMatrices.Matrix(hmat.data)#AR * BRT
norm(mulmat-Matrix(hmat))/norm(A)
norm(A-mulmat)/norm(A)
##
for blk in HMatrices.leaves(hmat)#
    @show HMatrices.rowrange(blk)
    @show Matrix(HMatrices.data(blk))
    @show HMatrices.pivot(hmat)
end
##
M = zeros(Float64, size(hmat)...)
for leaf in HMatrices.leaves(hmat)
    print("itering")
    M[HMatrices.rowrange(leaf), HMatrices.colrange(leaf)] = Matrix(HMatrices.data(leaf))
end
P = HMatrices.PermutedMatrix(M, invperm(HMatrices.rowperm(hmat)), invperm(HMatrices.colperm(hmat)))
##
@show HMatrices.rowperm(hmat)
@show invperm(HMatrices.rowperm(hmat))
=#

##

using HMatrices, LinearAlgebra, StaticArrays
const Point2D = SVector{2,Float64}

# points on a circle
m = n = 1000
#X = Y = [Point2D(sin(i*2π/n),cos(i*2π/n)) for i in 0:n-1]
X = meshrectangle(1.0, 1.0, 0.15).vertices
Y = translate(meshrectangle(1.0, 1.0, 0.15), SVector(4.0, 0.0, 0.0)).vertices

nothing

struct LaplaceMatrix2 <: AbstractMatrix{Float64}
  X::Vector{SVector{3, Float64}}
  Y::Vector{SVector{3, Float64}}
end

Base.getindex(K::LaplaceMatrix2,i::Int,j::Int) = -1/2π*log(norm(K.X[i] - K.Y[j]) + 1e-10)
Base.size(K::LaplaceMatrix2) = length(K.X), length(K.Y)

K = LaplaceMatrix2(X,Y)
# create the abstract matrix


Xclt = ClusterTree(X)
Yclt = ClusterTree(Y)
adm = StrongAdmissibilityStd()

comp = PartialACA(;rtol=1e-6)

H = assemble_hmatrix(K,Xclt,Yclt;adm,comp,threads=false,distributed=false)
##
norm(Matrix(H) - Matrix(K)) / norm(Matrix(K))
##
""" ploting error vs tolerance for sphere"""
powers = collect(-4:-1:-10)
rtols = [10.0^i for i in powers]
rel_errors = zeros(Float64, size(rtols))

Γ = meshsphere(1.0, 0.2) 
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)
A = assemble(op, spaceX, spaceX)

for (i, rtol) in enumerate(rtols)
    hmat = interface_to_hmatrix(op, spaceX, spaceX; rtol=rtol)
    vec = rand(dim)
    rel_errors[i] = norm(hmat * vec - A * vec) / norm(A * vec)   

end
##
plot(rtols, rel_errors)
scatter!(rtols, rel_errors)
plot!(xscale=:log10, yscale=:log10, minorgrid=true)
plot!(legend=false, size=(800, 600))
xlabel!("relative tolerance")
ylabel!("relative error")
png("err_vs_tol")
##
""" testing different matrix sizes"""
Γ = meshsphere(1.0, 0.025) 
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)
@time A = assemble(op, spaceX, spaceX)
@time hmat = interface_to_hmatrix(op, spaceX, spaceX)
vec = rand(dim)
rel_error = norm(hmat * vec - A * vec) / norm(A * vec)
println(hmat)  
println("Relative error: ", rel_error)
println("Size of H matrix: ", get_hmatrix_memory_size(hmat))
println("Size of full matrix: ", (length(A) * sizeof(eltype(A))))
##
cellsize = [0.2, 0.1, 0.05, 0.025]
rel_errs = [9.487778695782366e-9, 8.211300523458904e-8, 7.00491993467494e-8, 6.776679896434196e-8]
fulltime = [0.483899, 7.447569, 125.914074, 1927.346340]
htime = [0.425819, 2.705173, 14.549289, 78.892350]
compratio = [1.004475, 2.410772, 6.490204, 19.156679]
matdim =[804, 3214, 12526, 48192]
hsize = [5004528, 33523608, 190538440, 958363176]
fullsize = [5171328, 82638368, 1255205408, 18579750912]
myacatime = [0.428401, 2.977444, 15.888848, 76.397884]
nlogn = [n * log2(n) for n in matdim]
## 
plot(matdim, compratio)
scatter!(matdim, compratio)
plot!(legend=false, size=(800, 600), minorgrid=true)
xlabel!("Matrix dimension")
ylabel!("Size of full matrix ÷ Size of H Matrix")
png("compratio_vs_dim")

##
plot(matdim, fulltime, label="full matrix")
scatter!(matdim, fulltime, label=false)
plot!(matdim, htime, label="H matrix")
scatter!(matdim, htime, label=false)
plot!(matdim, myacatime, label="myACA")
scatter!(matdim, myacatime, label=false)
plot!(legend=true, size=(800, 600), minorgrid=true)
xlabel!("Matrix dimension")
ylabel!("time (S)")
png("time_vs_dim_aca")
##
plot(matdim, rel_errs)
scatter!(matdim, rel_errs)
plot!(legend=false, size=(800, 600), minorgrid=true)
xlabel!("Matrix dimension")
ylabel!("Relative error")
png("relerr_vs_dim")
##
hsize_gb = hsize ./ 2^30
fullsize_gb = fullsize ./ 2^30
memory_nlogn_coeff = hsize_gb[1] / nlogn[1]
memory_nlogn = nlogn .* memory_nlogn_coeff
plot(matdim, fullsize_gb, label="full matrix")
scatter!(matdim, fullsize_gb, label=false)
plot!(matdim, hsize_gb, label="H matrix")
scatter!(matdim, hsize_gb, label=false)
plot!(matdim, memory_nlogn, label="nlogn")
plot!(legend=true,xscale=:log10, yscale=:log10, size=(800, 600), minorgrid=true)
xlabel!("Matrix dimension")
ylabel!("Memory size (GB)")
png("memory_vs_dim")



## Playground

struct LeafBlock <:AbstractMatrix{Float64}
    fullMat :: InterfaceAbstractMatrix
    irange :: Vector{Int64}
    jrange :: Vector{Int64}
end

function LeafBlock(K :: HMatrices.PermutedMatrix, irange :: UnitRange{Int64}, jrange :: UnitRange{Int64})
    permuted_irange = K.rowperm[Vector(irange)]
    permuted_jrange = K.colperm[Vector(jrange)]
    return LeafBlock(K.data, permuted_irange, permuted_jrange)
end

# Base.getindex(K::LeafBlock, i::Int, j::Int) = K.fullMat[K.irange[i], K.jrange[j]]
function Base.getindex(K::LeafBlock, i::Int, j::Int)
    return K.fullMat[K.irange[i], K.jrange[j]]
end
Base.size(K::LeafBlock) = length(K.irange), length(K.jrange)

using ACAFact
Base.@kwdef struct myACA
    atol::Float64 = 0
    rank::Int = typemax(Int)
    rtol::Float64 = atol > 0 || rank < typemax(Int) ? 0 : sqrt(eps(Float64))
end


function (paca::myACA)(K, rowtree::ClusterTree, coltree::ClusterTree, bufs = nothing)
    irange = HMatrices.index_range(rowtree)
    jrange = HMatrices.index_range(coltree)
    leaf = LeafBlock(K, irange, jrange)
    (U, V) = aca(leaf, paca.rtol)
    return HMatrices.RkMatrix(U, V)
end

function ACAFact.row!(buff, K::LeafBlock, i)
    reshaped_buff = reshape(buff, (1, size(K, 2)))
    K.fullMat.blockassembler(reshaped_buff, [K.irange[i]], K.jrange)
end
function ACAFact.col!(buff, K::LeafBlock, j)
    K.fullMat.blockassembler(buff, K.irange, [K.jrange[j]])
end
##
powers = collect(-4:-1:-10)
rtols = [10.0^i for i in powers]
rel_errors = zeros(Float64, size(rtols))

Γ = meshsphere(1.0, 0.025) 
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
dim = length(spaceX.pos)
##
A = assemble(op, spaceX, spaceX)
##
@time hmat = interface_to_hmatrix(op, spaceX, spaceX; comp=myACA(rtol=1e-6))
##
for (i, rtol) in enumerate(rtols)
    hmat = interface_to_hmatrix(op, spaceX, spaceX; comp=myACA(rtol=rtol))
    vec = rand(dim)
    rel_errors[i] = norm(hmat * vec - A * vec) / norm(A * vec) 
end
##
plot(rtols, rel_errors)
scatter!(rtols, rel_errors)
plot!(xscale=:log10, yscale=:log10, minorgrid=true)
plot!(legend=false, size=(800, 600))
xlabel!("relative tolerance")
ylabel!("relative error")
png("err_vs_tol_myaca")

##
