using VlasovMethods
using Test

@testset "VlasovMethods.jl" begin
    include("spline_tests.jl")
    include("spline_basis_tests.jl")
    # include("projections_tests.jl")
    # include("electric_field_tests.jl")
end
