
remap_unit_interval(y, x₀, x₁) = x₀ + y * (x₁ - x₀)

unique_knot_vector(basis::AbstractBSplineBasis) = sort([Set(BSplineKit.knots(basis))...])
unique_knot_vector(basis::PeriodicBSplineBasis) = sort([Set(BSplineKit.knots(basis))...])[begin+(BSplineKit.order(basis)-1):end]

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
    knots = unique_knot_vector(basis)

    for i in axes(M,1)
        for j in axes(M,2)
            for k in eachindex(knots[begin:end-1])
                M[i,j] += quadrature(_mass_matrix_integrand(basis, knots, i, j, k)) * (knots[k+1] - knots[k])
            end
        end
    end  
    
    return M
end


struct SplineND{T, D, BT, DT, CT <: AbstractVector{T}, MT <: AbstractMatrix{T}, FT, QT <: QuadratureRule{T}}
    basis::BT
    derivative::DT
    coefficients::CT
    mass_matrix::MT
    mass_factor::FT
    quadrature::QT

    function SplineND(D, basis, coefficients, quadrature; mass_quadrature = quadrature)
        if BSplineKit.BSplines.has_parent_basis(basis)
            basis_der = BSplineKit.BSplines.basis_derivative(parent(basis), Derivative(1))
        else
            basis_der = BSplineKit.BSplines.basis_derivative(basis, Derivative(1))
        end

        mass_1d = mass_matrix(basis, mass_quadrature)
        mass_nd = kron((mass_1d for _ in 1:D)...)
        mass_fac = cholesky(mass_nd)

        new{
            eltype(coefficients),
            D,
            typeof(basis),
            typeof(basis_der),
            typeof(coefficients),
            typeof(mass_nd),
            typeof(mass_fac), 
            typeof(quadrature)
           }(
            basis,
            coefficients,
            basis_der,
            mass_nd,
            mass_fac,
            quadrature
            )
    end
end

