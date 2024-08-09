
remap_unit_interval(y, x₀, x₁) = x₀ + y * (x₁ - x₀)

unique_knots(basis::AbstractBSplineBasis) = sort([Set(BSplineKit.knots(basis))...])
unique_knots(basis::PeriodicBSplineBasis) = sort([Set(BSplineKit.knots(basis))...])[begin+(BSplineKit.order(basis)-1):end]


function _mass_matrix_integrand(basis::AbstractBSplineBasis, knots, i, j, k)
    remap = y -> remap_unit_interval(y, knots[k], knots[k+1])
    v -> (basis[i] ∘ remap)(v) * (basis[j] ∘ remap)(v)
end

function _mass_matrix_integrand(basis::PeriodicBSplineBasis, knots, i, j, k)
    remap  = y -> remap_unit_interval(y, knots[k], knots[k+1])
    remapm = y -> remap_unit_interval(y, knots[k], knots[k+1]) - (knots[end] - knots[begin])
    remapp = y -> remap_unit_interval(y, knots[k], knots[k+1]) + (knots[end] - knots[begin])
    v -> ((basis[i] ∘ remap)(v) + (basis[i] ∘ remapm)(v) + (basis[i] ∘ remapp)(v)) *
         ((basis[j] ∘ remap)(v) + (basis[j] ∘ remapm)(v) + (basis[j] ∘ remapp)(v))
end

function mass_matrix(basis::AbstractBSplineBasis, quadrature::QuadratureRule)
    M = zeros(length(basis), length(basis))
    knots = unique_knots(basis)

    for i in axes(M,1)
        for j in axes(M,2)
            for k in eachindex(knots[begin:end-1])
                M[i,j] += quadrature(_mass_matrix_integrand(basis, knots, i, j, k)) * (knots[k+1] - knots[k])
            end
        end
    end  
    
    return M
end


_spline_basis(p::BSplineOrder, knots::AbstractVector, ::Val{:Natural}) =
    BSplineBasis(p, knots)

_spline_basis(p::BSplineOrder, knots::AbstractVector, ::Val{:Dirichlet}) =
    RecombinedBSplineBasis(BSplineKit.Derivative(0), BSplineBasis(p, knots))

_spline_basis(p::BSplineOrder, knots::AbstractVector, ::Val{:Periodic}) =
    PeriodicBSplineBasis(p, knots)

function _spline_basis_derivative(basis::AbstractBSplineBasis)
    if BSplineKit.BSplines.has_parent_basis(basis)
        return BSplineKit.BSplines.basis_derivative(parent(basis), Derivative(1))
    else
        return BSplineKit.BSplines.basis_derivative(basis, Derivative(1))
    end
end

function _mass_kron(D, mass_mat)
    if D == 1
        return mass_mat
    else
        kron((mass_mat for _ in 1:D)...)
    end
end


struct SplineND{T, D, BT <: AbstractBSplineBasis, DT, CT1 <: AbstractArray{T}, CT2 <: AbstractVector{T}, MT <: AbstractMatrix{T}, FT, QT <: QuadratureRule{T}}
    basis::BT
    derivative::DT
    coefficients::CT1
    coeff_vector::CT2
    mass_matrix::MT
    mass_factor::FT
    quadrature::QT

    function SplineND{T,D}(basis::AbstractBSplineBasis, quadrature::QuadratureRule; mass_quadrature = quadrature) where {T,D}
        basis_der = _spline_basis_derivative(basis)

        mass_mat = _mass_kron(D, mass_matrix(basis, mass_quadrature))
        mass_fac = cholesky(mass_mat)

        coefficients = zeros(T, (length(basis) for _ in 1:D)...)
        coeff_vector = vec(coefficients)

        new{
            T,
            D,
            typeof(basis),
            typeof(basis_der),
            typeof(coefficients),
            typeof(coeff_vector),
            typeof(mass_mat),
            typeof(mass_fac), 
            typeof(quadrature)
           }(
            basis,
            basis_der,
            coefficients,
            coeff_vector,
            mass_mat,
            mass_fac,
            quadrature
            )
    end
end

SplineND(D::Int, args...; kwargs...) =
    SplineND{Float64,D}(args...; kwargs...)

SplineND(D::Int, p::Int, knots::AbstractVector, bcs::Symbol, args...; kwargs...) =
    SplineND(D, _spline_basis(BSplineOrder(p), knots, Val(bcs)), args...; kwargs...)

SplineND(D::Int, p::Int, knots::AbstractVector, args...; kwargs...) =
    SplineND(D, p, knots, :Natural, args...; kwargs...)


Base.eltype(::SplineND{T,D}) where {T,D} = T
Base.ndims(::SplineND{T,D}) where {T,D} = D

basis(s::SplineND) = s.basis
knots(s::SplineND) = BSplineKit.knots(basis(s))
unique_knots(s::SplineND) = unique_knots(basis(s))

order(::SplineND{T,D,BT}) where {T,D,BT} = BSplineKit.order(BT)

(s::SplineND)(x::AbstractVector) = evaluate(s, x)
(s::SplineND)(x::Number, y::Number) = evaluate(s, @SVector [x, y])
