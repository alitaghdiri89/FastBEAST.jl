using BEAST
using CompScienceMeshes
using HMatrices
##
struct InterfaceAbstractMatrix <:AbstractMatrix{Float64}
    operator::BEAST.HH3DSingleLayerFDBIO
    space::BEAST.LagrangeBasis
end

function Base.getindex(K::InterfaceAbstractMatrix, i::Int, j::Int)
    blkasm = BEAST.blockassembler(K.operator, K.space, K.space)    
    function blkassembler(Z, tdata, sdata)
        @views store(v,m,n) = (Z[m,n] += v)
        blkasm(tdata,sdata,store)
    end
    blk = zeros(ComplexF64, 1, 1)
    blkassembler(blk, [i], [j])
    return blk[1, 1]
end
Base.size(K::InterfaceAbstractMatrix) = length(K.space.pos)

## 
myMat = InterfaceAbstractMatrix(op, spaceX)
myMat[1, 1]
##

function interface_to_hmatrix(operator, space)
    abstract_matrix = InterfaceAbstractMatrix(operator, space)
    Xclt = Yclt = ClusterTree(space.pos)
    h_matrix = assemble_hmatrix(abstract_matrix, Xclt, Yclt)
    return h_matrix 
end
##
hmat = interface_to_hmatrix(op, spaceX)
hmat
##
Γ = meshsphere(1.0, 0.1) 
Γ.vertices
Γ.faces
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
#spaceY = lagrangecxd0(Γ)
A = assemble(op, spaceX, spaceX)
"""
Below this line are not important
****************************************************************************************************
"""
##
spaceX.fns
spaceX.geo
##

blkasm = BEAST.blockassembler(op, spaceX,spaceX)    
function blkassembler(Z, tdata, sdata)
    @views store(v,m,n) = (Z[m,n] += v)
    blkasm(tdata,sdata,store)
end


##
blk = zeros()

function element(i, j)
    blk = zeros(ComplexF64, 1, 1)
    blkassembler(blk, [i], [j])
    return blk[1, 1]
end

element(1,1)
A
##
spaceX