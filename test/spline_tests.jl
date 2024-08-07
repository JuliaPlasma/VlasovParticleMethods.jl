using BSplineKit
using LinearAlgebra
using QuadratureRules
using VlasovMethods
using Test

using VlasovMethods: mass_matrix, unit_interval_to_knot_interval


@testset "Spline Utilities" begin

    @test unit_interval_to_knot_interval(0.0, 0.0, 2.0) == 0.0
    @test unit_interval_to_knot_interval(0.5, 0.0, 2.0) == 1.0
    @test unit_interval_to_knot_interval(1.0, 0.0, 2.0) == 2.0

end


@testset "Mass Matrix" begin

    sorder = 2
    nknots = 5
    sknots = collect(0.0:(nknots-1))

    # B-spline basis with Dirichlet BCs
    sknots = collect(0.0:(nknots-1))
    sbasis = BSplineBasis(BSplineOrder(sorder), copy(sknots))
    nbasis = length(sbasis)

    # Dirichlet BCs with (exact) Gauss-Legendre quadrature
    smass = mass_matrix(sbasis, GaussLegendreQuadrature(2))
    rmass = galerkin_matrix(sbasis)

    @test all(isapprox.(smass, rmass; atol = 2eps()))

    # Dirichlet BCs with trapezoidal quadrature
    smass = mass_matrix(sbasis, TrapezoidalQuadrature())
    rmass = Diagonal([0.5, repeat([1.0], nbasis-2)..., 0.5])

    @test all(smass .== rmass)

    # B-spline basis with Periodic BCs
    sknots = collect(0.0:(nknots-1))
    sbasis = PeriodicBSplineBasis(BSplineOrder(sorder), copy(sknots))
    nbasis = length(sbasis)

    # Periodic BCs with (exact) Gauss-Legendre quadrature
    smass = mass_matrix(sbasis, GaussLegendreQuadrature(2))
    rmass = galerkin_matrix(sbasis)

    @test all(isapprox.(smass, rmass; atol = 2eps()))

    # Periodic BCs with trapezoidal quadrature
    smass = mass_matrix(sbasis, TrapezoidalQuadrature())
    rmass = Matrix(1.0I, nbasis, nbasis)
    
    @test all(smass .== rmass)

end
