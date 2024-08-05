struct SplineDistribution{DT, XD, VD, ST, BT, MT, FT} <: DistributionFunction{DT,XD,VD}
    spline::ST
    # spline::Spline{DT, BT, Vector{DT}}
    basis::BT
    coefficients::Vector{DT}
    mass_matrix::MT
    mass_fact::FT

    function SplineDistribution(xdim, vdim, basis::BT, coefficients::AbstractArray{DT}, mass_matrix) where {BT, DT}
        _coefficients = Vector(coefficients)
        # mass_1d = galerkin_matrix(basis)
        if vdim == 1
            spline = Spline(basis, _coefficients)
            # mass_matrix = mass_1d
        elseif vdim == 2
            spline = TwoDSpline(basis, _coefficients)
            # mass_matrix = kron(mass_1d, mass_1d)
        end
        # mass_fact = lu(mass_matrix)
        mass_fact = cholesky(mass_matrix)
        new{DT, xdim, vdim, typeof(spline), typeof(basis), typeof(mass_matrix), typeof(mass_fact)}(spline, basis, _coefficients, mass_matrix, mass_fact)
    end
end

Base.eltype(::SplineDistribution{DT}) where {DT} = DT
Base.length(dist::SplineDistribution) = length(dist.coefficients)

Base.similar(AT, s::SplineDistribution{DT,XD,VD}) where {DT,XD,VD} =
    SplineDistribution(XD, VD, s.basis, zeros(AT, axes(s.coefficients)), s.mass_matrix)

similar_type(AT, ::SplineDistribution{DT, XD, VD, TwoDSpline{DT, BT, BT2}, BT, MT, FT}) where {DT, XD, VD, BT, MT, FT, BT2} = SplineDistribution{AT, XD, VD, TwoDSpline{AT, BT, BT2}, BT, MT, FT}
similar_type(AT, ::SplineDistribution{DT, XD, VD, Spline{DT, BT, Vector{DT}}, BT, MT, FT}) where {DT, XD, VD, BT, MT, FT} = SplineDistribution{AT, XD, VD, Spline{AT, BT, Vector{AT}}, BT, MT, FT}


function SplineDistribution(xdim, vdim, nknots::KT, s_order::OT, domain::Tuple, length_big_cell, bc::Symbol=:Dirichlet, compute_mass_galerkin::Bool=true) where {KT, OT}
    ts = collect(LinRange(domain..., nknots))
    if length_big_cell > 0
        extended_knots = [domain[1] - length_big_cell, ts..., domain[2] + length_big_cell]
    else
        extended_knots = ts
    end

    # b = BSplineBasis(BSplineOrder(s_order), extended_knots)

    # b = BSplineBasis(BSplineOrder(s_order), extended_knots)
    if bc == :Dirichlet
        b = BSplineBasis(BSplineOrder(s_order), extended_knots)
        basis = RecombinedBSplineBasis(Derivative(0), b)
    elseif bc == :Periodic
        basis = PeriodicBSplineBasis(BSplineOrder(s_order), extended_knots)
    else #TODO: add other boundary condition options here
        b = BSplineBasis(BSplineOrder(s_order), extended_knots)
        basis = b
    end

    if compute_mass_galerkin
        mass_1d = galerkin_matrix(basis, Matrix{Float64})
        # mass_1d = galerkin_matrix(basis)
    else
        mass_1d = compute_mass_matrix(basis, extended_knots)
    end

    coefficients = zeros(length(basis)^vdim)

    if vdim == 1
        mass_matrix = mass_1d
    elseif vdim == 2
        mass_matrix = kron(mass_1d, mass_1d)
    end

    return SplineDistribution(xdim, vdim, basis, coefficients, mass_matrix)
end

# TODO: Move to spline subdir. This is not distribution function specific functionality.
# TODO: Make quadrature rule an argument.
function compute_mass_matrix(basis, knots)
    M = zeros(length(basis), length(basis))
    for k in CartesianIndices(M)
        if k[1] ≥ k[2]
            integrand = [basis[k[1]](v) * basis[k[2]](v) for v in knots]
            M[k] = trapz(knots, integrand)
        else
            M[k] = M[k[2], k[1]]
        end
    end  
    
    return M
end
