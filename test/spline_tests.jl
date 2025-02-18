using BSplineKit
using LinearAlgebra
using QuadratureRules
using Random
using VlasovMethods
using Test

using VlasovMethods: mass_matrix, order, remap_unit_interval, unique_knots
using VlasovMethods: evaluate_basis, evaluate_basis_derivative, indices


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


    # B-splines of order 3 to 6
    for sorder in 3:6

        # B-spline basis with Dirichlet BCs
        sbasis = BSplineBasis(BSplineOrder(sorder), copy(sknots))

        # Dirichlet BCs with (exact) Gauss-Legendre quadrature
        smass = mass_matrix(sbasis, GaussLegendreQuadrature(sorder))
        rmass = galerkin_matrix(sbasis)

        @test all(isapprox.(smass, rmass; atol = 2eps()))


        # B-spline basis with Periodic BCs
        sbasis = PeriodicBSplineBasis(BSplineOrder(sorder), copy(sknots))
        
        # Periodic BCs with (exact) Gauss-Legendre quadrature
        smass = mass_matrix(sbasis, GaussLegendreQuadrature(sorder))
        rmass = galerkin_matrix(sbasis)

        @test all(isapprox.(smass, rmass; atol = 2eps()))

    end

end


@testset "1-dimensional B-Spline" begin

    d = 1
    o = 2
    k = 0:0.1:2
    q = GaussLegendreQuadrature(2)

    @test_nowarn SplineND(d, o, k, q)
    @test_nowarn SplineND(d, o, k, :Natural, q)
    @test_nowarn SplineND(d, o, k, :Dirichlet, q)
    @test_nowarn SplineND(d, o, k, :Periodic, q)


    ### B-spline with natural boundary conditions ###

    s = SplineND(d, o, k, :Natural, q)

    @test eltype(s) == Float64
    @test ndims(s) == d
    @test order(s) == o
    @test size(s) == (length(k)+(o-2),)
    @test unique_knots(s) == k

    rand!(s.coefficients)

    @test s(k[begin]) == s([k[begin]]) == s.coefficients[begin]
    @test s(k[end]) == s([k[end]]) == s.coefficients[end]


    s = SplineND(1, 2, 0:0.5:1, :Natural, GaussLegendreQuadrature(1))

    @test evaluate_basis(s, [0.0], 1) == 1.0
    @test evaluate_basis(s, [0.0], 2) == 0.0
    @test evaluate_basis(s, [0.0], 3) == 0.0

    @test evaluate_basis(s, [0.5], 1) == 0.0
    @test evaluate_basis(s, [0.5], 2) == 1.0
    @test evaluate_basis(s, [0.5], 3) == 0.0

    @test evaluate_basis(s, [1.0], 1) == 0.0
    @test evaluate_basis(s, [1.0], 2) == 0.0
    @test evaluate_basis(s, [1.0], 3) == 1.0


    s = SplineND(1, 3, 0:0.5:1, :Natural, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.0], 1) == 1.0
    @test evaluate_basis(s, [0.0], 2) == 0.0
    @test evaluate_basis(s, [0.0], 3) == 0.0
    @test evaluate_basis(s, [0.0], 4) == 0.0

    @test evaluate_basis(s, [0.5], 1) == 0.0
    @test evaluate_basis(s, [0.5], 2) == 0.5
    @test evaluate_basis(s, [0.5], 3) == 0.5
    @test evaluate_basis(s, [0.5], 4) == 0.0

    @test evaluate_basis(s, [1.0], 1) == 0.0
    @test evaluate_basis(s, [1.0], 2) == 0.0
    @test evaluate_basis(s, [1.0], 3) == 0.0
    @test evaluate_basis(s, [1.0], 4) == 1.0


    ### B-spline with Dirichlet boundary conditions ###

    s = SplineND(d, o, k, :Dirichlet, q)

    @test eltype(s) == Float64
    @test ndims(s) == d
    @test order(s) == o
    @test size(s) == (length(k)-2,)
    @test unique_knots(s) == k

    rand!(s.coefficients)

    @test s(k[begin]) == s([k[begin]]) == 0
    @test s(k[end]) == s([k[end]]) == 0


    s = SplineND(1, 2, 0:0.25:1, :Dirichlet, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.00], 1) == 0.0
    @test evaluate_basis(s, [0.00], 2) == 0.0
    @test evaluate_basis(s, [0.00], 3) == 0.0

    @test evaluate_basis(s, [0.25], 1) == 1.0
    @test evaluate_basis(s, [0.25], 2) == 0.0
    @test evaluate_basis(s, [0.25], 3) == 0.0

    @test evaluate_basis(s, [0.50], 1) == 0.0
    @test evaluate_basis(s, [0.50], 2) == 1.0
    @test evaluate_basis(s, [0.50], 3) == 0.0

    @test evaluate_basis(s, [0.75], 1) == 0.0
    @test evaluate_basis(s, [0.75], 2) == 0.0
    @test evaluate_basis(s, [0.75], 3) == 1.0

    @test evaluate_basis(s, [1.00], 1) == 0.0
    @test evaluate_basis(s, [1.00], 2) == 0.0
    @test evaluate_basis(s, [1.00], 3) == 0.0


    s = SplineND(1, 3, 0:0.25:1, :Dirichlet, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.00], 1) == 0.0
    @test evaluate_basis(s, [0.00], 2) == 0.0
    @test evaluate_basis(s, [0.00], 3) == 0.0
    @test evaluate_basis(s, [0.00], 4) == 0.0

    @test evaluate_basis(s, [0.25], 1) == 0.5
    @test evaluate_basis(s, [0.25], 2) == 0.5
    @test evaluate_basis(s, [0.25], 3) == 0.0
    @test evaluate_basis(s, [0.25], 4) == 0.0

    @test evaluate_basis(s, [0.50], 1) == 0.0
    @test evaluate_basis(s, [0.50], 2) == 0.5
    @test evaluate_basis(s, [0.50], 3) == 0.5
    @test evaluate_basis(s, [0.50], 4) == 0.0

    @test evaluate_basis(s, [0.75], 1) == 0.0
    @test evaluate_basis(s, [0.75], 2) == 0.0
    @test evaluate_basis(s, [0.75], 3) == 0.5
    @test evaluate_basis(s, [0.75], 4) == 0.5

    @test evaluate_basis(s, [1.00], 1) == 0.0
    @test evaluate_basis(s, [1.00], 2) == 0.0
    @test evaluate_basis(s, [1.00], 3) == 0.0
    @test evaluate_basis(s, [1.00], 4) == 0.0

    
    ### B-spline with periodic boundary conditions ###

    s = SplineND(d, o, k, :Periodic, q)

    @test eltype(s) == Float64
    @test ndims(s) == d
    @test order(s) == o
    @test size(s) == (length(k)-1,)
    @test unique_knots(s) == k

    rand!(s.coefficients)

    @test s(k[begin]) == s(k[end])


    s = SplineND(1, 2, 0:0.25:1, :Periodic, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.00], 1) == 1.0
    @test evaluate_basis(s, [0.00], 2) == 0.0
    @test evaluate_basis(s, [0.00], 3) == 0.0
    @test evaluate_basis(s, [0.00], 4) == 0.0

    @test evaluate_basis(s, [0.25], 1) == 0.0
    @test evaluate_basis(s, [0.25], 2) == 1.0
    @test evaluate_basis(s, [0.25], 3) == 0.0
    @test evaluate_basis(s, [0.25], 4) == 0.0

    @test evaluate_basis(s, [0.50], 1) == 0.0
    @test evaluate_basis(s, [0.50], 2) == 0.0
    @test evaluate_basis(s, [0.50], 3) == 1.0
    @test evaluate_basis(s, [0.50], 4) == 0.0

    @test evaluate_basis(s, [0.75], 1) == 0.0
    @test evaluate_basis(s, [0.75], 2) == 0.0
    @test evaluate_basis(s, [0.75], 3) == 0.0
    @test evaluate_basis(s, [0.75], 4) == 1.0

    @test evaluate_basis(s, [1.00], 1) == 0.0
    @test evaluate_basis(s, [1.00], 2) == 0.0
    @test evaluate_basis(s, [1.00], 3) == 0.0
    @test evaluate_basis(s, [1.00], 4) == 0.0


    s = SplineND(1, 3, 0:0.25:1, :Periodic, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.00], 1) == 0.5
    @test evaluate_basis(s, [0.00], 2) == 0.0
    @test evaluate_basis(s, [0.00], 3) == 0.0
    @test evaluate_basis(s, [0.00], 4) == 0.0

    @test evaluate_basis(s, [0.25], 1) == 0.5
    @test evaluate_basis(s, [0.25], 2) == 0.5
    @test evaluate_basis(s, [0.25], 3) == 0.0
    @test evaluate_basis(s, [0.25], 4) == 0.0

    @test evaluate_basis(s, [0.50], 1) == 0.0
    @test evaluate_basis(s, [0.50], 2) == 0.5
    @test evaluate_basis(s, [0.50], 3) == 0.5
    @test evaluate_basis(s, [0.50], 4) == 0.0

    @test evaluate_basis(s, [0.75], 1) == 0.0
    @test evaluate_basis(s, [0.75], 2) == 0.0
    @test evaluate_basis(s, [0.75], 3) == 0.5
    @test evaluate_basis(s, [0.75], 4) == 0.5

    @test evaluate_basis(s, [1.00], 1) == 0.0
    @test evaluate_basis(s, [1.00], 2) == 0.0
    @test evaluate_basis(s, [1.00], 3) == 0.0
    @test evaluate_basis(s, [1.00], 4) == 0.5

    
    ### Check size, knots, and indices

    for o in 2:8
        b = SplineND(d, o, k, :Natural, q)
        @test size(b) == (length(k) + (o-2),)
        @test unique_knots(b) == k
        @test indices(b, 1) == (1,)
        @test indices(b, size(b)[1]) == size(b)
            
        b = SplineND(d, o, k, :Dirichlet, q)
        @test size(b) == (length(k) + (o-4),)
        @test unique_knots(b) == k
        @test indices(b, 1) == (1,)
        @test indices(b, size(b)[1]) == size(b)
        
        b = SplineND(d, o, k, :Periodic, q)
        @test size(b) == (length(k)-1,)
        @test unique_knots(b) == k
        @test indices(b, 1) == (1,)
        @test indices(b, size(b)[1]) == size(b)
    end

end


@testset "2-dimensional B-Spline" begin

    d = 2
    o = 3
    k = -2:0.1:+1
    q = GaussLegendreQuadrature(3)

    @test_nowarn SplineND(d, o, k, q)
    @test_nowarn SplineND(d, o, k, :Natural, q)
    @test_nowarn SplineND(d, o, k, :Dirichlet, q)
    @test_nowarn SplineND(d, o, k, :Periodic, q)

    ### B-spline with natural boundary conditions ###

    s = SplineND(d, o, k, :Natural, q)

    @test eltype(s) == Float64
    @test ndims(s) == d
    @test order(s) == o
    @test size(s) == (length(k)+(o-2), length(k)+(o-2))
    @test unique_knots(s) == k

    rand!(s.coefficients)

    @test s(k[begin], k[begin]) == s.coefficients[begin, begin]
    @test s(k[begin], k[end]) == s.coefficients[begin, end]
    @test s(k[end], k[begin]) == s.coefficients[end, begin]
    @test s(k[end], k[end]) == s.coefficients[end, end]


    s = SplineND(2, 2, 0:0.5:1, :Natural, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.0, 0.0], (1,1)) == 1.0
    @test evaluate_basis(s, [0.0, 0.0], (1,2)) == 0.0
    @test evaluate_basis(s, [0.0, 0.0], (1,3)) == 0.0

    @test evaluate_basis(s, [0.0, 0.5], (1,1)) == 0.0
    @test evaluate_basis(s, [0.0, 0.5], (1,2)) == 1.0
    @test evaluate_basis(s, [0.0, 0.5], (1,3)) == 0.0

    @test evaluate_basis(s, [0.0, 1.0], (1,1)) == 0.0
    @test evaluate_basis(s, [0.0, 1.0], (1,2)) == 0.0
    @test evaluate_basis(s, [0.0, 1.0], (1,3)) == 1.0

    @test evaluate_basis(s, [0.5, 0.0], (2,1)) == 1.0
    @test evaluate_basis(s, [0.5, 0.0], (2,2)) == 0.0
    @test evaluate_basis(s, [0.5, 0.0], (2,3)) == 0.0

    @test evaluate_basis(s, [0.5, 0.5], (2,1)) == 0.0
    @test evaluate_basis(s, [0.5, 0.5], (2,2)) == 1.0
    @test evaluate_basis(s, [0.5, 0.5], (2,3)) == 0.0

    @test evaluate_basis(s, [0.5, 1.0], (2,1)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (2,2)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (2,3)) == 1.0

    @test evaluate_basis(s, [1.0, 0.0], (3,1)) == 1.0
    @test evaluate_basis(s, [1.0, 0.0], (3,2)) == 0.0
    @test evaluate_basis(s, [1.0, 0.0], (3,3)) == 0.0

    @test evaluate_basis(s, [1.0, 0.5], (3,1)) == 0.0
    @test evaluate_basis(s, [1.0, 0.5], (3,2)) == 1.0
    @test evaluate_basis(s, [1.0, 0.5], (3,3)) == 0.0

    @test evaluate_basis(s, [1.0, 1.0], (3,1)) == 0.0
    @test evaluate_basis(s, [1.0, 1.0], (3,2)) == 0.0
    @test evaluate_basis(s, [1.0, 1.0], (3,3)) == 1.0

    @test evaluate_basis(s, [0.0, 0.5], (2,1)) == 0.0
    @test evaluate_basis(s, [0.0, 1.0], (3,1)) == 0.0

    @test evaluate_basis(s, [0.5, 0.0], (1,2)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (3,2)) == 0.0

    @test evaluate_basis(s, [1.0, 0.0], (1,3)) == 0.0
    @test evaluate_basis(s, [1.0, 0.5], (2,3)) == 0.0


    s = SplineND(2, 3, 0:0.5:1, :Natural, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.0, 0.0], (1,1)) == 1.0
    @test evaluate_basis(s, [0.0, 0.0], (1,2)) == 0.0
    @test evaluate_basis(s, [0.0, 0.0], (1,3)) == 0.0
    @test evaluate_basis(s, [0.0, 0.0], (1,4)) == 0.0

    @test evaluate_basis(s, [0.0, 0.5], (1,1)) == 0.0
    @test evaluate_basis(s, [0.0, 0.5], (1,2)) == 0.5
    @test evaluate_basis(s, [0.0, 0.5], (1,3)) == 0.5
    @test evaluate_basis(s, [0.0, 0.5], (1,4)) == 0.0

    @test evaluate_basis(s, [0.0, 1.0], (1,1)) == 0.0
    @test evaluate_basis(s, [0.0, 1.0], (1,2)) == 0.0
    @test evaluate_basis(s, [0.0, 1.0], (1,3)) == 0.0
    @test evaluate_basis(s, [0.0, 1.0], (1,4)) == 1.0
    
    @test evaluate_basis(s, [0.5, 0.0], (2,1)) == 0.5
    @test evaluate_basis(s, [0.5, 0.0], (2,2)) == 0.0
    @test evaluate_basis(s, [0.5, 0.0], (2,3)) == 0.0
    @test evaluate_basis(s, [0.5, 0.0], (2,4)) == 0.0

    @test evaluate_basis(s, [0.5, 0.5], (2,1)) == 0.0
    @test evaluate_basis(s, [0.5, 0.5], (2,2)) == 0.25
    @test evaluate_basis(s, [0.5, 0.5], (2,3)) == 0.25
    @test evaluate_basis(s, [0.5, 0.5], (2,4)) == 0.0

    @test evaluate_basis(s, [0.5, 1.0], (2,1)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (2,2)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (2,3)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (2,4)) == 0.5

    @test evaluate_basis(s, [0.5, 0.0], (3,1)) == 0.5
    @test evaluate_basis(s, [0.5, 0.0], (3,2)) == 0.0
    @test evaluate_basis(s, [0.5, 0.0], (3,3)) == 0.0
    @test evaluate_basis(s, [0.5, 0.0], (3,4)) == 0.0

    @test evaluate_basis(s, [0.5, 0.5], (3,1)) == 0.0
    @test evaluate_basis(s, [0.5, 0.5], (3,2)) == 0.25
    @test evaluate_basis(s, [0.5, 0.5], (3,3)) == 0.25
    @test evaluate_basis(s, [0.5, 0.5], (3,4)) == 0.0

    @test evaluate_basis(s, [0.5, 1.0], (3,1)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (3,2)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (3,3)) == 0.0
    @test evaluate_basis(s, [0.5, 1.0], (3,4)) == 0.5

    @test evaluate_basis(s, [1.0, 0.0], (4,1)) == 1.0
    @test evaluate_basis(s, [1.0, 0.0], (4,2)) == 0.0
    @test evaluate_basis(s, [1.0, 0.0], (4,3)) == 0.0
    @test evaluate_basis(s, [1.0, 0.0], (4,4)) == 0.0

    @test evaluate_basis(s, [1.0, 0.5], (4,1)) == 0.0
    @test evaluate_basis(s, [1.0, 0.5], (4,2)) == 0.5
    @test evaluate_basis(s, [1.0, 0.5], (4,3)) == 0.5
    @test evaluate_basis(s, [1.0, 0.5], (4,4)) == 0.0

    @test evaluate_basis(s, [1.0, 1.0], (4,1)) == 0.0
    @test evaluate_basis(s, [1.0, 1.0], (4,2)) == 0.0
    @test evaluate_basis(s, [1.0, 1.0], (4,3)) == 0.0
    @test evaluate_basis(s, [1.0, 1.0], (4,4)) == 1.0


    ### B-spline with Dirichlet boundary conditions ###

    s = SplineND(d, o, k, :Dirichlet, q)

    @test eltype(s) == Float64
    @test ndims(s) == d
    @test order(s) == o
    @test size(s) == (length(k)-1, length(k)-1)
    @test unique_knots(s) == k

    rand!(s.coefficients)

    @test s(k[begin], k[begin]) == 0
    @test s(k[begin], k[end]) == 0
    @test s(k[end], k[begin]) == 0
    @test s(k[end], k[end]) == 0

    
    s = SplineND(2, 2, 0:0.25:1, :Dirichlet, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.00, 0.00], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 0.00], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 0.00], (1,3)) == 0.0

    @test evaluate_basis(s, [0.00, 0.25], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 0.25], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 0.25], (1,3)) == 0.0

    @test evaluate_basis(s, [0.00, 0.50], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 0.50], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 0.50], (1,3)) == 0.0

    @test evaluate_basis(s, [0.00, 0.75], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 0.75], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 0.75], (1,3)) == 0.0

    @test evaluate_basis(s, [0.00, 1.00], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 1.00], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 1.00], (1,3)) == 0.0

    @test evaluate_basis(s, [0.25, 0.00], (1,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.00], (1,2)) == 0.0
    @test evaluate_basis(s, [0.25, 0.00], (1,3)) == 0.0

    @test evaluate_basis(s, [0.25, 0.25], (1,1)) == 1.0
    @test evaluate_basis(s, [0.25, 0.25], (1,2)) == 0.0
    @test evaluate_basis(s, [0.25, 0.25], (1,3)) == 0.0

    @test evaluate_basis(s, [0.25, 0.50], (1,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.50], (1,2)) == 1.0
    @test evaluate_basis(s, [0.25, 0.50], (1,3)) == 0.0

    @test evaluate_basis(s, [0.25, 0.75], (1,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (1,2)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (1,3)) == 1.0

    @test evaluate_basis(s, [0.25, 1.00], (1,1)) == 0.0
    @test evaluate_basis(s, [0.25, 1.00], (1,2)) == 0.0
    @test evaluate_basis(s, [0.25, 1.00], (1,3)) == 0.0

    @test evaluate_basis(s, [0.50, 0.00], (2,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.00], (2,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.00], (2,3)) == 0.0

    @test evaluate_basis(s, [0.50, 0.25], (2,1)) == 1.0
    @test evaluate_basis(s, [0.50, 0.25], (2,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.25], (2,3)) == 0.0

    @test evaluate_basis(s, [0.50, 0.50], (2,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.50], (2,2)) == 1.0
    @test evaluate_basis(s, [0.50, 0.50], (2,3)) == 0.0

    @test evaluate_basis(s, [0.50, 0.75], (2,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (2,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (2,3)) == 1.0

    @test evaluate_basis(s, [0.50, 1.00], (2,1)) == 0.0
    @test evaluate_basis(s, [0.50, 1.00], (2,2)) == 0.0
    @test evaluate_basis(s, [0.50, 1.00], (2,3)) == 0.0

    @test evaluate_basis(s, [0.75, 0.00], (3,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.00], (3,2)) == 0.0
    @test evaluate_basis(s, [0.75, 0.00], (3,3)) == 0.0

    @test evaluate_basis(s, [0.75, 0.25], (3,1)) == 1.0
    @test evaluate_basis(s, [0.75, 0.25], (3,2)) == 0.0
    @test evaluate_basis(s, [0.75, 0.25], (3,3)) == 0.0

    @test evaluate_basis(s, [0.75, 0.50], (3,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.50], (3,2)) == 1.0
    @test evaluate_basis(s, [0.75, 0.50], (3,3)) == 0.0

    @test evaluate_basis(s, [0.75, 0.75], (3,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (3,2)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (3,3)) == 1.0

    @test evaluate_basis(s, [0.75, 1.00], (3,1)) == 0.0
    @test evaluate_basis(s, [0.75, 1.00], (3,2)) == 0.0
    @test evaluate_basis(s, [0.75, 1.00], (3,3)) == 0.0

    @test evaluate_basis(s, [1.00, 0.00], (3,1)) == 0.0
    @test evaluate_basis(s, [1.00, 0.00], (3,2)) == 0.0
    @test evaluate_basis(s, [1.00, 0.00], (3,3)) == 0.0

    @test evaluate_basis(s, [1.00, 0.25], (3,1)) == 0.0
    @test evaluate_basis(s, [1.00, 0.25], (3,2)) == 0.0
    @test evaluate_basis(s, [1.00, 0.25], (3,3)) == 0.0

    @test evaluate_basis(s, [1.00, 0.50], (3,1)) == 0.0
    @test evaluate_basis(s, [1.00, 0.50], (3,2)) == 0.0
    @test evaluate_basis(s, [1.00, 0.50], (3,3)) == 0.0

    @test evaluate_basis(s, [1.00, 0.75], (3,1)) == 0.0
    @test evaluate_basis(s, [1.00, 0.75], (3,2)) == 0.0
    @test evaluate_basis(s, [1.00, 0.75], (3,3)) == 0.0

    @test evaluate_basis(s, [1.00, 1.00], (3,1)) == 0.0
    @test evaluate_basis(s, [1.00, 1.00], (3,2)) == 0.0
    @test evaluate_basis(s, [1.00, 1.00], (3,3)) == 0.0


    s = SplineND(2, 3, 0:0.25:1, :Dirichlet, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.25, 0.25], (1,1)) == 0.25
    @test evaluate_basis(s, [0.25, 0.25], (1,2)) == 0.25
    @test evaluate_basis(s, [0.25, 0.25], (1,3)) == 0.0
    @test evaluate_basis(s, [0.25, 0.25], (1,4)) == 0.0

    @test evaluate_basis(s, [0.25, 0.50], (1,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.50], (1,2)) == 0.25
    @test evaluate_basis(s, [0.25, 0.50], (1,3)) == 0.25
    @test evaluate_basis(s, [0.25, 0.50], (1,4)) == 0.0

    @test evaluate_basis(s, [0.25, 0.75], (1,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (1,2)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (1,3)) == 0.25
    @test evaluate_basis(s, [0.25, 0.75], (1,4)) == 0.25

    @test evaluate_basis(s, [0.50, 0.25], (2,1)) == 0.25
    @test evaluate_basis(s, [0.50, 0.25], (2,2)) == 0.25
    @test evaluate_basis(s, [0.50, 0.25], (2,3)) == 0.0
    @test evaluate_basis(s, [0.50, 0.25], (2,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.50], (2,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.50], (2,2)) == 0.25
    @test evaluate_basis(s, [0.50, 0.50], (2,3)) == 0.25
    @test evaluate_basis(s, [0.50, 0.50], (2,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.75], (2,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (2,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (2,3)) == 0.25
    @test evaluate_basis(s, [0.50, 0.75], (2,4)) == 0.25

    @test evaluate_basis(s, [0.50, 0.25], (3,1)) == 0.25
    @test evaluate_basis(s, [0.50, 0.25], (3,2)) == 0.25
    @test evaluate_basis(s, [0.50, 0.25], (3,3)) == 0.0
    @test evaluate_basis(s, [0.50, 0.25], (3,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.50], (3,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.50], (3,2)) == 0.25
    @test evaluate_basis(s, [0.50, 0.50], (3,3)) == 0.25
    @test evaluate_basis(s, [0.50, 0.50], (3,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.75], (3,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (3,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (3,3)) == 0.25
    @test evaluate_basis(s, [0.50, 0.75], (3,4)) == 0.25

    @test evaluate_basis(s, [0.75, 0.25], (4,1)) == 0.25
    @test evaluate_basis(s, [0.75, 0.25], (4,2)) == 0.25
    @test evaluate_basis(s, [0.75, 0.25], (4,3)) == 0.0
    @test evaluate_basis(s, [0.75, 0.25], (4,4)) == 0.0

    @test evaluate_basis(s, [0.75, 0.50], (4,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.50], (4,2)) == 0.25
    @test evaluate_basis(s, [0.75, 0.50], (4,3)) == 0.25
    @test evaluate_basis(s, [0.75, 0.50], (4,4)) == 0.0

    @test evaluate_basis(s, [0.75, 0.75], (4,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (4,2)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (4,3)) == 0.25
    @test evaluate_basis(s, [0.75, 0.75], (4,4)) == 0.25


    ### B-spline with periodic boundary conditions ###

    s = SplineND(d, o, k, :Periodic, q)

    @test eltype(s) == Float64
    @test ndims(s) == d
    @test order(s) == o
    @test size(s) == (length(k)-1, length(k)-1)
    @test unique_knots(s) == k

    rand!(s.coefficients)

    @test s(k[begin], k[begin]) ≈ s(k[begin], k[end]) atol = 2eps()
    @test s(k[begin], k[begin]) ≈ s(k[end], k[begin]) atol = 2eps()
    @test s(k[begin], k[begin]) ≈ s(k[end], k[end])   atol = 2eps()


    s = SplineND(2, 2, 0:0.25:1, :Periodic, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.00, 0.00], (1,1)) == 1.0
    @test evaluate_basis(s, [0.00, 0.00], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 0.00], (1,3)) == 0.0
    @test evaluate_basis(s, [0.00, 0.00], (1,4)) == 0.0

    @test evaluate_basis(s, [0.00, 0.25], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 0.25], (1,2)) == 1.0
    @test evaluate_basis(s, [0.00, 0.25], (1,3)) == 0.0
    @test evaluate_basis(s, [0.00, 0.25], (1,4)) == 0.0

    @test evaluate_basis(s, [0.00, 0.50], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 0.50], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 0.50], (1,3)) == 1.0
    @test evaluate_basis(s, [0.00, 0.50], (1,4)) == 0.0

    @test evaluate_basis(s, [0.00, 0.75], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 0.75], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 0.75], (1,3)) == 0.0
    @test evaluate_basis(s, [0.00, 0.75], (1,4)) == 1.0

    @test evaluate_basis(s, [0.00, 1.00], (1,1)) == 0.0
    @test evaluate_basis(s, [0.00, 1.00], (1,2)) == 0.0
    @test evaluate_basis(s, [0.00, 1.00], (1,3)) == 0.0
    @test evaluate_basis(s, [0.00, 1.00], (1,4)) == 0.0

    @test evaluate_basis(s, [0.25, 0.00], (2,1)) == 1.0
    @test evaluate_basis(s, [0.25, 0.00], (2,2)) == 0.0
    @test evaluate_basis(s, [0.25, 0.00], (2,3)) == 0.0
    @test evaluate_basis(s, [0.25, 0.00], (2,4)) == 0.0

    @test evaluate_basis(s, [0.25, 0.25], (2,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.25], (2,2)) == 1.0
    @test evaluate_basis(s, [0.25, 0.25], (2,3)) == 0.0
    @test evaluate_basis(s, [0.25, 0.25], (2,4)) == 0.0

    @test evaluate_basis(s, [0.25, 0.50], (2,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.50], (2,2)) == 0.0
    @test evaluate_basis(s, [0.25, 0.50], (2,3)) == 1.0
    @test evaluate_basis(s, [0.25, 0.50], (2,4)) == 0.0

    @test evaluate_basis(s, [0.25, 0.75], (2,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (2,2)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (2,3)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (2,4)) == 1.0

    @test evaluate_basis(s, [0.25, 1.00], (2,1)) == 0.0
    @test evaluate_basis(s, [0.25, 1.00], (2,2)) == 0.0
    @test evaluate_basis(s, [0.25, 1.00], (2,3)) == 0.0
    @test evaluate_basis(s, [0.25, 1.00], (2,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.00], (3,1)) == 1.0
    @test evaluate_basis(s, [0.50, 0.00], (3,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.00], (3,3)) == 0.0
    @test evaluate_basis(s, [0.50, 0.00], (3,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.25], (3,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.25], (3,2)) == 1.0
    @test evaluate_basis(s, [0.50, 0.25], (3,3)) == 0.0
    @test evaluate_basis(s, [0.50, 0.25], (3,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.50], (3,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.50], (3,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.50], (3,3)) == 1.0
    @test evaluate_basis(s, [0.50, 0.50], (3,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.75], (3,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (3,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (3,3)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (3,4)) == 1.0

    @test evaluate_basis(s, [0.50, 1.00], (3,1)) == 0.0
    @test evaluate_basis(s, [0.50, 1.00], (3,2)) == 0.0
    @test evaluate_basis(s, [0.50, 1.00], (3,3)) == 0.0
    @test evaluate_basis(s, [0.50, 1.00], (3,4)) == 0.0

    @test evaluate_basis(s, [0.75, 0.00], (4,1)) == 1.0
    @test evaluate_basis(s, [0.75, 0.00], (4,2)) == 0.0
    @test evaluate_basis(s, [0.75, 0.00], (4,3)) == 0.0
    @test evaluate_basis(s, [0.75, 0.00], (4,4)) == 0.0

    @test evaluate_basis(s, [0.75, 0.25], (4,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.25], (4,2)) == 1.0
    @test evaluate_basis(s, [0.75, 0.25], (4,3)) == 0.0
    @test evaluate_basis(s, [0.75, 0.25], (4,4)) == 0.0

    @test evaluate_basis(s, [0.75, 0.50], (4,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.50], (4,2)) == 0.0
    @test evaluate_basis(s, [0.75, 0.50], (4,3)) == 1.0
    @test evaluate_basis(s, [0.75, 0.50], (4,4)) == 0.0

    @test evaluate_basis(s, [0.75, 0.75], (4,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (4,2)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (4,3)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (4,4)) == 1.0

    @test evaluate_basis(s, [0.75, 1.00], (4,1)) == 0.0
    @test evaluate_basis(s, [0.75, 1.00], (4,2)) == 0.0
    @test evaluate_basis(s, [0.75, 1.00], (4,3)) == 0.0
    @test evaluate_basis(s, [0.75, 1.00], (4,4)) == 0.0

    @test evaluate_basis(s, [1.00, 0.00], (1,1)) == 0.0
    @test evaluate_basis(s, [1.00, 0.00], (1,2)) == 0.0
    @test evaluate_basis(s, [1.00, 0.00], (1,3)) == 0.0
    @test evaluate_basis(s, [1.00, 0.00], (1,4)) == 0.0

    @test evaluate_basis(s, [1.00, 0.25], (2,1)) == 0.0
    @test evaluate_basis(s, [1.00, 0.25], (2,2)) == 0.0
    @test evaluate_basis(s, [1.00, 0.25], (2,3)) == 0.0
    @test evaluate_basis(s, [1.00, 0.25], (2,4)) == 0.0

    @test evaluate_basis(s, [1.00, 0.50], (3,1)) == 0.0
    @test evaluate_basis(s, [1.00, 0.50], (3,2)) == 0.0
    @test evaluate_basis(s, [1.00, 0.50], (3,3)) == 0.0
    @test evaluate_basis(s, [1.00, 0.50], (3,4)) == 0.0

    @test evaluate_basis(s, [1.00, 0.75], (4,1)) == 0.0
    @test evaluate_basis(s, [1.00, 0.75], (4,2)) == 0.0
    @test evaluate_basis(s, [1.00, 0.75], (4,3)) == 0.0
    @test evaluate_basis(s, [1.00, 0.75], (4,4)) == 0.0

    @test evaluate_basis(s, [1.00, 1.00], (1,1)) == 0.0
    @test evaluate_basis(s, [1.00, 1.00], (2,2)) == 0.0
    @test evaluate_basis(s, [1.00, 1.00], (3,3)) == 0.0
    @test evaluate_basis(s, [1.00, 1.00], (4,4)) == 0.0


    s = SplineND(2, 3, 0:0.25:1, :Periodic, GaussLegendreQuadrature(2))

    @test evaluate_basis(s, [0.25, 0.25], (1,1)) == 0.25
    @test evaluate_basis(s, [0.25, 0.25], (1,2)) == 0.25
    @test evaluate_basis(s, [0.25, 0.25], (1,3)) == 0.0
    @test evaluate_basis(s, [0.25, 0.25], (1,4)) == 0.0

    @test evaluate_basis(s, [0.25, 0.50], (1,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.50], (1,2)) == 0.25
    @test evaluate_basis(s, [0.25, 0.50], (1,3)) == 0.25
    @test evaluate_basis(s, [0.25, 0.50], (1,4)) == 0.0

    @test evaluate_basis(s, [0.25, 0.75], (1,1)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (1,2)) == 0.0
    @test evaluate_basis(s, [0.25, 0.75], (1,3)) == 0.25
    @test evaluate_basis(s, [0.25, 0.75], (1,4)) == 0.25

    @test evaluate_basis(s, [0.50, 0.25], (2,1)) == 0.25
    @test evaluate_basis(s, [0.50, 0.25], (2,2)) == 0.25
    @test evaluate_basis(s, [0.50, 0.25], (2,3)) == 0.0
    @test evaluate_basis(s, [0.50, 0.25], (2,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.50], (2,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.50], (2,2)) == 0.25
    @test evaluate_basis(s, [0.50, 0.50], (2,3)) == 0.25
    @test evaluate_basis(s, [0.50, 0.50], (2,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.75], (2,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (2,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (2,3)) == 0.25
    @test evaluate_basis(s, [0.50, 0.75], (2,4)) == 0.25

    @test evaluate_basis(s, [0.50, 0.25], (3,1)) == 0.25
    @test evaluate_basis(s, [0.50, 0.25], (3,2)) == 0.25
    @test evaluate_basis(s, [0.50, 0.25], (3,3)) == 0.0
    @test evaluate_basis(s, [0.50, 0.25], (3,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.50], (3,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.50], (3,2)) == 0.25
    @test evaluate_basis(s, [0.50, 0.50], (3,3)) == 0.25
    @test evaluate_basis(s, [0.50, 0.50], (3,4)) == 0.0

    @test evaluate_basis(s, [0.50, 0.75], (3,1)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (3,2)) == 0.0
    @test evaluate_basis(s, [0.50, 0.75], (3,3)) == 0.25
    @test evaluate_basis(s, [0.50, 0.75], (3,4)) == 0.25

    @test evaluate_basis(s, [0.75, 0.25], (4,1)) == 0.25
    @test evaluate_basis(s, [0.75, 0.25], (4,2)) == 0.25
    @test evaluate_basis(s, [0.75, 0.25], (4,3)) == 0.0
    @test evaluate_basis(s, [0.75, 0.25], (4,4)) == 0.0

    @test evaluate_basis(s, [0.75, 0.50], (4,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.50], (4,2)) == 0.25
    @test evaluate_basis(s, [0.75, 0.50], (4,3)) == 0.25
    @test evaluate_basis(s, [0.75, 0.50], (4,4)) == 0.0

    @test evaluate_basis(s, [0.75, 0.75], (4,1)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (4,2)) == 0.0
    @test evaluate_basis(s, [0.75, 0.75], (4,3)) == 0.25
    @test evaluate_basis(s, [0.75, 0.75], (4,4)) == 0.25


    ### Check size and knots

    for o in 2:8
        b = SplineND(d, o, k, :Natural, q)
        @test size(b) == (length(k) + (o-2), length(k) + (o-2))
        @test unique_knots(b) == k
        @test indices(b, 1) == (1,1)
        @test indices(b, size(b)[1] * size(b)[2]) == size(b)
        
        b = SplineND(d, o, k, :Dirichlet, q)
        @test size(b) == (length(k) + (o-4), length(k) + (o-4))
        @test unique_knots(b) == k
        @test indices(b, 1) == (1,1)
        @test indices(b, size(b)[1] * size(b)[2]) == size(b)
        
        b = SplineND(d, o, k, :Periodic, q)
        @test size(b) == (length(k)-1, length(k)-1)
        @test unique_knots(b) == k
        @test indices(b, 1) == (1,1)
        @test indices(b, size(b)[1] * size(b)[2]) == size(b)
    end

end
