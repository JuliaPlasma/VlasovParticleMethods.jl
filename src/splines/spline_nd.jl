
unit_interval_to_knot_interval(y, x₀, x₁) = x₀ + y * (x₁ - x₀)

function mass_matrix(basis::AbstractBSplineBasis, quadrature::QuadratureRule)
    M = zeros(length(basis), length(basis))
    K = sort([Set(BSplineKit.knots(basis))...])

    for i in axes(M,1)
        for j in axes(M,2)
            for k in eachindex(K[begin:end-1])
                remap = y -> unit_interval_to_knot_interval(y, K[k], K[k+1])
                M[i,j] += quadrature(v -> (basis[i] ∘ remap)(v) * (basis[j] ∘ remap)(v)) * (K[k+1] - K[k])
            end
        end
    end  
    
    return M
end

function mass_matrix(basis::PeriodicBSplineBasis, quadrature::QuadratureRule)
    p = BSplineKit.order(basis)
    M = zeros(length(basis), length(basis))
    K = sort([Set(BSplineKit.knots(basis))...])[begin+(p-1):end]
    Δ = K[end] - K[begin]

    for i in axes(M,1)
        for j in axes(M,2)
            for k in eachindex(@view K[begin:end-1])
                remap  = y -> unit_interval_to_knot_interval(y, K[k], K[k+1])
                remapm = y -> unit_interval_to_knot_interval(y, K[k], K[k+1]) - Δ
                integrand = v -> ((basis[i] ∘ remap)(v) + (basis[i] ∘ remapm)(v)) *
                                 ((basis[j] ∘ remap)(v) + (basis[j] ∘ remapm)(v))
                M[i,j] += quadrature(integrand) * (K[k+1] - K[k])
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

