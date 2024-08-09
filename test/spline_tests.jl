using BSplineKit
using LinearAlgebra
using QuadratureRules
using VlasovMethods
using Test

using VlasovMethods: mass_matrix, order, remap_unit_interval, unique_knots


@testset "Spline Utilities" begin

    @test remap_unit_interval(0.0, 0.0, 2.0) == 0.0
    @test remap_unit_interval(0.5, 0.0, 2.0) == 1.0
    @test remap_unit_interval(1.0, 0.0, 2.0) == 2.0

end


@testset "Mass Matrix" begin

    # Knot vector with 5 knots
    nknots = 5
    sknots = collect(0.0:(nknots-1)) ./ 2

    # B-splines of order 2
    sorder = 2

    # B-spline basis with Dirichlet BCs
    sbasis = BSplineBasis(BSplineOrder(sorder), copy(sknots))
    nbasis = length(sbasis)

    # Dirichlet BCs with (exact) Gauss-Legendre quadrature
    smass = mass_matrix(sbasis, GaussLegendreQuadrature(2))
    rmass = galerkin_matrix(sbasis)

    @test all(isapprox.(smass, rmass; atol = 2eps()))

    # Dirichlet BCs with trapezoidal quadrature
    smass = mass_matrix(sbasis, TrapezoidalQuadrature())
    rmass = Diagonal([0.5, repeat([1.0], nbasis-2)..., 0.5] ./ 2)

    @test all(smass .== rmass)

    # B-spline basis with Periodic BCs
    sbasis = PeriodicBSplineBasis(BSplineOrder(sorder), copy(sknots))
    nbasis = length(sbasis)

    # Periodic BCs with (exact) Gauss-Legendre quadrature
    smass = mass_matrix(sbasis, GaussLegendreQuadrature(2))
    rmass = galerkin_matrix(sbasis)

    @test all(isapprox.(smass, rmass; atol = 2eps()))

    # Periodic BCs with trapezoidal quadrature
    smass = mass_matrix(sbasis, TrapezoidalQuadrature())
    rmass = Matrix(1.0I, nbasis, nbasis) ./ 2
    
    @test all(smass .== rmass)


    # B-splines of order 3
    sorder = 3

    # B-spline basis with Dirichlet BCs
    sbasis = BSplineBasis(BSplineOrder(sorder), copy(sknots))

    # Dirichlet BCs with (exact) Gauss-Legendre quadrature
    smass = mass_matrix(sbasis, GaussLegendreQuadrature(3))
    rmass = galerkin_matrix(sbasis)

    @test all(isapprox.(smass, rmass; atol = 2eps()))

    # B-spline basis with Periodic BCs
    sbasis = PeriodicBSplineBasis(BSplineOrder(sorder), copy(sknots))
    
    # Periodic BCs with (exact) Gauss-Legendre quadrature
    smass = mass_matrix(sbasis, GaussLegendreQuadrature(3))
    rmass = galerkin_matrix(sbasis)

    @test all(isapprox.(smass, rmass; atol = 2eps()))

end

@testset "N-dimensional B-Spline" begin
    d = 1
    o = 2
    k = 0:0.1:2
    q = GaussLegendreQuadrature(2)

    @test_nowarn SplineND(d, o, copy(k), q)
    @test_nowarn SplineND(d, o, copy(k), :Natural, q)
    @test_nowarn SplineND(d, o, copy(k), :Dirichlet, q)
    @test_nowarn SplineND(d, o, copy(k), :Periodic, q)

    s = SplineND(d, o, k, q)

    # @test SplineND(d, o, copy(k), :Natural, q) == s
    @test eltype(s) == Float64
    @test ndims(s) == d
    @test order(s) == o
    @test unique_knots(s) == k


    d = 2
    o = 3
    k = -2:0.1:+1
    q = GaussLegendreQuadrature(3)

    @test_nowarn SplineND(d, o, copy(k), q)
    @test_nowarn SplineND(d, o, copy(k), :Natural, q)
    @test_nowarn SplineND(d, o, copy(k), :Dirichlet, q)
    @test_nowarn SplineND(d, o, copy(k), :Periodic, q)

    s = SplineND(d, o, k, q)

    # @test SplineND(d, o, copy(k), :Natural, q) == s
    @test eltype(s) == Float64
    @test ndims(s) == d
    @test order(s) == o
    @test unique_knots(s) == k

end
