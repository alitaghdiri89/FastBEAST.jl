using Test, TestItems, TestItemRunner
using FastBEAST

@testitem "BlockSparseMatrices" begin
    include("test_aca.jl")
    include("test_beast.jl")
    include("test_blockassembler.jl")
    include("test_fmm.jl")
    include("test_hmatrix.jl")
    include("test_tree.jl")
end

@testitem "Code formatting (JuliaFormatter.jl)" begin
    using JuliaFormatter
    pkgpath = pkgdir(FastBEAST)
    @test JuliaFormatter.format(pkgpath, overwrite=false)
end

@run_package_tests verbose = true
