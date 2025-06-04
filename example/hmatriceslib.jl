using HMatrices, LinearAlgebra, StaticArrays
const Point3D = SVector{3,Float64}
# sample some points on a sphere
m = 1000
X = Y = [Point3D(sin(θ)cos(ϕ),sin(θ)*sin(ϕ),cos(θ)) for (θ,ϕ) in zip(π*rand(m),2π*rand(m))]
function G(x,y) 
  d = norm(x-y) + 1e-8
  1/(4π*d)
end
##
G(X[1], Y[1])
##

K = KernelMatrix(G,X,Y)

H = assemble_hmatrix(K;atol=1e-6)
Xclt = Yclt = ClusterTree(X)
##

##
using CompScienceMeshes
using BEAST
Γ = meshsphere(1.0, 0.1) 
Γ.vertices
Γ.faces
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
#spaceY = lagrangecxd0(Γ)
A = assemble(op, spaceX, spaceX)
##
blkasm = BEAST.blockassembler(op, spaceX,spaceX)    
function blkassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end

##
blk = zeros(Float64, 3, 3)
blkassembler(blk, [1, 2, 3], [1, 2, 3])
blk
##
#ClusterTree
ClusterTree(spaceX.pos)

##

