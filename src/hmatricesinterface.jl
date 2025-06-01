using BEAST
using CompScienceMeshes


function hmatricesinterface(block)
    function f(row, column)
        index = (row.-1) * block.size[1] + column
        return block[index]
    end
    """Each element needs to be an SVector because that's what assemble_hmatrix expects"""
    X = [SVector(i) for i in 1:block.size[1]]
    Y = [SVector(i) for i in 1:block.size[2]]
    return KernelMatrix(f,X,Y)
end


## 
"""
1st test the small 3 * 3 blk matrix, seems to work fine.
"""
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
kernel_mat = hmatricesinterface(blk)
kernel_mat.X
##
H = assemble_hmatrix(kernel_mat;atol=1e-6)
##
#ClusterTree
ClusterTree(spaceX.pos)

##
"""
2nd test with the big 3214*3214 matrix A 
while conversion to kernel matrix seems to work fine, assemble_hmatrix fails
    Is there any other difference between A and blk apart from their size?
"""

Γ = meshsphere(1.0, 0.1) 
Γ.vertices
Γ.faces
op = Helmholtz3D.singlelayer()
spaceX = lagrangecxd0(Γ)
#spaceY = lagrangecxd0(Γ)
A = assemble(op, spaceX, spaceX)
##
kernel_mat2 = hmatricesinterface(A)
kernel_mat2
##
H2 = assemble_hmatrix(kernel_mat2; atol=1e-6)

