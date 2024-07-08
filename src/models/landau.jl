struct Landau{XD, VD, DT <: DistributionFunction{XD,VD}, ET <: Entropy, T} <: VlasovModel
    dist::DT    # distribution function
    ent::ET     # entropy 
    ν::T        # collision frequency 
    
    function Landau(dist::DistributionFunction{XD,VD}, ent::Entropy; ν::T=1.) where {XD, VD, T}
        new{XD, VD, typeof(dist), typeof(ent), T}(dist, ent, ν)
    end
end

function integrand_J(v::AbstractArray{T}, B, i, j, sdist) where T
    return (B[i, T](v[1]) * B[j, T](v[2]) * (1. + log(abs(sdist.spline(v)))))::T
end

function integrand_J_M(v::AbstractArray{T}, B, i, j, sdist) where T
    return (B[i, T](v[1]) * B[j, T](v[2]) * (1. + log(f_Maxwellian(v))))::T
end

function compute_J(sdist::SplineDistribution{1,2})
    T = eltype(sdist)
    int = zeros(T, size(sdist))
    B = sdist.basis
    d_start = T(BSplineKit.knots(B)[1])
    d_end = T(BSplineKit.knots(B)[end])
    domain = ([d_start, d_start], [d_end, d_end])

    Threads.@threads for k in 1:size(sdist)
        i, j = ij_from_k(k, length(B))
        integrand(v) = integrand_J(v, B, i, j, sdist)
        # integand(v) = B[i, T](v[1]) * B[j, T](v[2]) * (1. + log(abs(sdist.spline(v))))
        int[k], _ = HCubature.hcubature(integrand,[d_start, d_start], [d_end, d_end], atol = 1e-4, rtol=1e-4)
    end

    ldiv!(sdist.mass_fact, int)

    return int
end

function compute_U(v_α, v_β)
    n = length(v_α)
    U = zeros(eltype(v_α),(n,n))

    norm_diff = norm(v_α - v_β)

    if v_α != v_β
        for i in CartesianIndices(U)
            if i[1] == i[2]
                U[i] += 1/norm_diff
            end
            U[i] -= (v_α[i[1]] - v_β[i[1]])*(v_α[i[2]] - v_β[i[2]])./norm_diff^3
        end
    end
    return U
end

function compute_U(v_α_1::T, v_α_2, v_β_1, v_β_2) where{T}
    # n = length(v_α)
    U = zeros(T,(2,2))

    # norm_diff = norm(v_α - v_β)
    norm_diff = sqrt((v_α_1 - v_β_1)^2 + (v_α_2 - v_β_2)^2)

    if norm_diff > eps(T)
        U[1,1] = 1/norm_diff - (v_α_1 - v_β_1)^2/norm_diff^3
        U[2,2] = 1/norm_diff - (v_α_2 - v_β_2)^2/norm_diff^3

        U[1,2] = 1/norm_diff - (v_α_1 - v_β_1)*(v_α_2 - v_β_2)/norm_diff^3
        U[2,1] = U[1,2]
    end
    return U
end

function Landau_rhs!(v̇, v, v_array, L, B, sdist)
    # computes rhs for a single particle, assuming that the projection and other particle velocities are taken from the previous timestep 
    # params.L is the vector L_k, which depends on the projection
    # params.idist.particles.v is the particle velocities
    # params.fdist.basis is the spline basis

    K = size(sdist) # length of tensor product basis (this is the square of the 1d basis length)

    for α in axes(v_array, 2)
        U = compute_U(v, v_array[:,α])
        for k in 1:K # could make this more precise by using find_knot_interval function 
            v̇ .+=  L[k] * U * (eval_bfd(B, k, v) - eval_bfd(B, k, v_array[:, α]))
        end
    end

    return v̇
end

# particle-to-particle version
function Landau_rhs(v, params)
    # computes rhs for a single particle, assuming that the projection and other particle velocities are taken from the previous timestep 
    # params.L is the vector L_k, which depends on the projection
    # params.idist.particles.v is the particle velocities
    # params.fdist.basis is the spline basis
    v̇ = zero(v)
    K = size(params.sdist) # length of tensor product basis (this is the square of the 1d basis length)

    ind1, res1 = evaluate_der_2d(params.B, v)

    for α in axes(params.v_array, 2)
        U = compute_U(v, params.v_array[:,α])
        ind_α, res_α = evaluate_der_2d(params.B, params.v_array[:, α])

        for (i, k) in pairs(ind1)
            if k > 0 && k ≤ K
                v̇ .+=  params.dist.particles.w[1,α] * params.L[k] * U * res1[:, i]
            end
        end

        for (i2, k2) in pairs(ind_α)
            if k2 > 0 && k2 ≤ K
                v̇ .-=  params.dist.particles.w[1,α] * params.L[k2] * U * res_α[:, i2]
            end
        end
    end

    return v̇
end


function compute_K_plus(v_array::AbstractArray{T}, dist, sdist) where {T}
    M = size(sdist)
    K1 = zeros(T, (M, size(v_array,2)))
    K2 = zeros(T, (M, size(v_array,2)))

    Threads.@threads for α in axes(v_array, 2)
        klist, der_array = evaluate_der_2d(sdist.basis, v_array[:,α])
        for (i, k) in pairs(klist)
            if k > 0 && k <= M
                K1[k, α] = dist.particles.w[1,α] * der_array[1,i]
                K2[k, α] = dist.particles.w[1,α] * der_array[2,i]
            end
        end
    end

    if rank(K1) < M  || rank(K2) < M
        println("K1 or K2 not full rank")
        @show size(K1,1) - rank(K1)
        @show size(K2,1) - rank(K2)
    end

    return pinv(K1), pinv(K2)
end

function f_Maxwellian(v)
    return 1/(2π) * exp(- 0.5 * norm(v)^2)
end

# function L_integrand(v::AbstractArray{T}, k, sdist) where T
#     v1 = [v[1], v[2]]
#     v2 = [v[3], v[4]]
    
#     id_list_1 = evaluate_der_2d_indices(sdist.basis, v1)
#     id_list_2 = evaluate_der_2d_indices(sdist.basis, v2)
    
#     if (k[1] in id_list_1 || k[1] in id_list_2) && (k[2] in id_list_1 || k[2] in id_list_2)
#         # U = compute_U(v1, v2)
#         basis_derivative = zeros(T, 2)
#         eval_bfd!(basis_derivative, sdist.basis, k[1], v1, 0, 1)
#         eval_bfd!(basis_derivative, sdist.basis, k[1], v2, 1, -1)

#         integrand = transpose(basis_derivative) * sdist.spline(v1) * compute_U(v1, v2)

#         eval_bfd!(basis_derivative, sdist.basis, k[2], v1, 0, 1)
#         eval_bfd!(basis_derivative, sdist.basis, k[2], v2, 1, -1)

#         return (integrand * sdist.spline(v2) * basis_derivative)::T

#     else
#         return zero(T)
#     end
# end

function L_integrand_vec(v::AbstractVector{T}, params) where T
    v1 = [v[1], v[2]]
    v2 = [v[3], v[4]]
    
    id_list_1 = evaluate_der_2d_indices(params.sdist.basis, v1)
    id_list_2 = evaluate_der_2d_indices(params.sdist.basis, v2)
    
    if (params.k[1] in id_list_1 || params.k[1] in id_list_2) && (params.k[2] in id_list_1 || params.k[2] in id_list_2)
        # U = compute_U(v1, v2)
        basis_derivative = zeros(T, 2)
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[1], v1, 0, 1)
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[1], v2, 1, -1)

        integrand = transpose(basis_derivative) * params.sdist.spline(v1) * compute_U(v1, v2)
        # integrand = transpose(basis_derivative) * params.sdist.spline(v1) * compute_U(v1, v2)

        eval_bfd!(basis_derivative, params.sdist.basis, params.k[2], v1, 0, 1)
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[2], v2, 1, -1)

        return (integrand * params.sdist.spline(v2) * basis_derivative)::T

    else
        return zero(T)
    end
end


function L_integrand_vec_M(v::AbstractVector{T}, params) where T
    v1 = [v[1], v[2]]
    v2 = [v[3], v[4]]
    
    id_list_1 = evaluate_der_2d_indices(params.sdist.basis, v1)
    id_list_2 = evaluate_der_2d_indices(params.sdist.basis, v2)
    
    if (params.k[1] in id_list_1 || params.k[1] in id_list_2) && (params.k[2] in id_list_1 || params.k[2] in id_list_2)
        # U = compute_U(v1, v2)
        basis_derivative = zeros(T, 2)
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[1], v1, 0, 1)
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[1], v2, 1, -1)

        integrand = transpose(basis_derivative) * f_Maxwellian(v1) * compute_U(v1, v2)
        # integrand = transpose(basis_derivative) * params.sdist.spline(v1) * compute_U(v1, v2)

        eval_bfd!(basis_derivative, params.sdist.basis, params.k[2], v1, 0, 1)
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[2], v2, 1, -1)

        return (integrand * f_Maxwellian(v2) * basis_derivative)::T

    else
        return zero(T)
    end
end


function L_integrand(v::AbstractVector{T}, params) where T
    # v1 = [v[1], v[2]]
    # v2 = [v[3], v[4]]
    
    id_list_1 = evaluate_der_2d_indices(params.sdist.basis, v[1], v[2])
    id_list_2 = evaluate_der_2d_indices(params.sdist.basis, v[3], v[4])
    
    if (params.k[1] in id_list_1 || params.k[1] in id_list_2) && (params.k[2] in id_list_1 || params.k[2] in id_list_2)
        # U = compute_U(v1, v2)
        basis_derivative = zeros(T, 2)
        # @show v
        # println("eval basis der 1")
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[1], v[1], v[2], 0, 1)
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[1], v[3], v[4], 1, -1)
        # println("eval integrand 1")
        integrand = transpose(basis_derivative) * params.sdist.spline(v[1], v[2]) * compute_U(v[1], v[2], v[3], v[4])
        # integrand = transpose(basis_derivative) * params.sdist.spline(v1) * compute_U(v1, v2)
        # println("eval basis der 2")
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[2], v[1], v[2], 0, 1)
        eval_bfd!(basis_derivative, params.sdist.basis, params.k[2], v[3], v[4], 1, -1)

        return (integrand * params.sdist.spline(v[3], v[4]) * basis_derivative)::T

    else
        return zero(T)
    end
end

function compute_L_ij_new(sdist)
    T = eltype(sdist)
    m = BSplineKit.order(sdist.basis)
    L = zeros(T, (size(sdist), size(sdist)))
    d_start = T(BSplineKit.knots(sdist.basis)[1])
    d_end = T(BSplineKit.knots(sdist.basis)[end])
    domain = ([d_start, d_start, d_start, d_start], [d_end, d_end, d_end, d_end])

    Threads.@threads for k in CartesianIndices(L)
        i1, j1 = ij_from_k(k[1], length(sdist.basis))
        i2, j2 = ij_from_k(k[2], length(sdist.basis))
        if k[1] ≤ k[2] && (abs(i1 - i2) ≤ m) && (abs(j1 - j2) ≤ m) 
            params = (k = k, sdist = sdist)
            prob = IntegralProblem(L_integrand_vec, domain, params)
            sol = Integrals.solve(prob, HCubatureJL(); abstol = 1e-4, reltol = 1e-4)
            L[k] = sol.u
        end
    end

    Threads.@threads for k in CartesianIndices(L)
        if k[1] > k[2] 
            L[k] = L[k[2], k[1]]
        end
    end

    return L .* 0.5
end

function compute_L_ij(sdist)
    T = eltype(sdist)
    m = BSplineKit.order(sdist.basis)
    L = zeros(T, (size(sdist), size(sdist)))
    d_start = T(BSplineKit.knots(sdist.basis)[1])
    d_end = T(BSplineKit.knots(sdist.basis)[end])
    domain = ([d_start, d_start, d_start, d_start], [d_end, d_end, d_end, d_end])

    Threads.@threads for k in CartesianIndices(L)
        if k[1] ≤ k[2]
            params = (k = k, sdist = sdist)
            prob = IntegralProblem(L_integrand_vec, domain, params)
            sol = Integrals.solve(prob, HCubatureJL(); abstol = 1e-4, reltol = 1e-4)
            L[k] = sol.u
        end
    end

    Threads.@threads for k in CartesianIndices(L)
        if k[1] > k[2] 
            L[k] = L[k[2], k[1]]
        end
    end

    return L .* 0.5
end



# function compute_L_ij(sdist)
#     T = eltype(sdist)
#     L = zeros(T, (size(sdist), size(sdist)))
#     d_start = T(BSplineKit.knots(sdist.basis)[1])
#     d_end = T(BSplineKit.knots(sdist.basis)[end])
    
#     Threads.@threads for k in CartesianIndices(L)
#         if k[1] ≤ k[2]
#             integrand(v) = L_integrand(v, k, sdist)
#             L[k], _ = hcubature(integrand, [d_start, d_start, d_start, d_start], [d_end, d_end, d_end, d_end], atol = 1e-4, rtol = 1e-4)
#         end
#     end

#     Threads.@threads for k in CartesianIndices(L)
#         if k[1] > k[2]
#             L[k] = L[k[2], k[1]]
#         end
#     end

#     return L .* 0.5
# end

# spline-to-spline? version 
function Landau_rhs_2!(v̇, v::AbstractArray{ST}, params) where {ST}
    # v̇ = zero(v)
    # project v onto params.sdist
    sdist = params.ent.cache[ST]

    S = projection(v, params.dist, sdist)

    # compute K matrices 
    # println("compute K")
    K1_plus, K2_plus = compute_K_plus(v, params.dist, sdist)

    # compute L_ij matrix
    # println("computing L")
    Lij = compute_L_ij(sdist)

    # compute J vector
    # println("computing J")
    J = compute_J(sdist)
    # J = compute_J_M(sdist)

    # solve for vector field
    v̇[1,:] .= K1_plus * Lij * J  
    v̇[2,:] .= K2_plus * Lij * J  

    return v̇

end

function Landau_rhs_2_fM!(v̇, v::AbstractArray{ST}, params) where {ST}
    # v̇ = zero(v)
    # project v onto params.sdist
    sdist = params.ent.cache[ST]

    # S = projection(v, params.dist, sdist)
    S = project_Maxwellian(sdist)

    # compute K matrices 
    # println("compute K")
    K1_plus, K2_plus = compute_K_plus(v, params.dist, sdist)

    # compute L_ij matrix
    # println("computing L")
    Lij = compute_L_ij(sdist)

    # compute J vector
    # println("computing J")
    J = compute_J(sdist)
    # J = compute_J_M(sdist)

    # solve for vector field
    v̇[1,:] .= K1_plus * Lij * J  
    v̇[2,:] .= K2_plus * Lij * J  

    return v̇

end