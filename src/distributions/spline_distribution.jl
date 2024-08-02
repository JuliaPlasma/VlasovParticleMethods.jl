struct SplineDistribution{XD, VD, ST, DT, BT, MT, FT} <: DistributionFunction{XD,VD}
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
        new{xdim, vdim, typeof(spline), DT, typeof(basis), typeof(mass_matrix), typeof(mass_fact)}(spline, basis, _coefficients, mass_matrix, mass_fact)
    end
end

Base.size(dist::SplineDistribution) = length(dist.coefficients)
Base.eltype(::SplineDistribution{XD, VD, ST, DT, BT, MT, FT}) where {XD, VD, ST, DT, BT, MT, FT} = DT

Cache(AT, s::SplineDistribution{XD, VD, ST, DT, BT, MT, FT}) where {XD, VD, ST, DT, BT, MT, FT} = SplineDistribution(XD, VD, s.basis, zeros(AT, axes(s.coefficients)), s.mass_matrix)
CacheType(AT, ::SplineDistribution{XD, VD, TwoDSpline{DT, BT, BT2}, DT, BT, MT, FT}) where {XD, VD, DT, BT, MT, FT, BT2} = SplineDistribution{XD, VD, TwoDSpline{AT, BT, BT2}, AT, BT, MT, FT}
CacheType(AT, ::SplineDistribution{XD, VD, Spline{DT, BT, Vector{DT}}, DT, BT, MT, FT}) where {XD, VD, DT, BT, MT, FT} = SplineDistribution{XD, VD, Spline{AT, BT, Vector{AT}}, AT, BT, MT, FT}


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

    if vdim == 1
        coefficients = zeros(length(basis))
        mass_matrix = mass_1d
    elseif vdim == 2
        coefficients = zeros(length(basis)^2)
        mass_matrix = kron(mass_1d, mass_1d)

    end
    return SplineDistribution(xdim, vdim, basis, coefficients, mass_matrix)
end

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


_cachehash(ST) = hash(Threads.threadid(), hash(ST))

struct SplineDistributionCache{disttype}
    s::disttype
    caches::Dict{UInt64, SplineDistribution}

    function SplineDistributionCache(s::SplineDistribution)
        caches = Dict{UInt64, SplineDistribution}()
        caches[_cachehash(eltype(s))] = s
        new{typeof(s)}(s, caches)
    end
end

@inline function Base.getindex(c::SplineDistributionCache, ST::DataType)
    key = _cachehash(ST)
    if haskey(c.caches, key)
        c.caches[key]
    else
        c.caches[key] = Cache(ST, c.s)
    end::CacheType(ST, c.s)
end
