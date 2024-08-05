struct Landau{XD, VD, DT <: DistributionFunction{XD,VD}, ET <: Entropy, T} <: VlasovModel
    dist::DT    # distribution function
    ent::ET     # entropy 
    ν::T        # collision frequency 
    
    function Landau(dist::DistributionFunction{XD,VD}, ent::Entropy; ν::T=1.) where {XD, VD, T}
        new{XD, VD, typeof(dist), typeof(ent), T}(dist, ent, ν)
    end
end

function integrand_J(v::AbstractArray{T}, B, i, j, sdist) where T
    return B[i, T](v[1]) * B[j, T](v[2]) * (one(T) + log(sdist.spline(v)))
end

function integrand_J(v::AbstractArray{T}, params) where T
    # if params.sdist.spline(v) > eps(T)
        # return (params.B[params.i, T](v[1]) * params.B[params.j, T](v[2]) * (1. + log(f_Maxwellian(v))))::T
    return params.B[params.i, T](v[1]) * params.B[params.j, T](v[2]) * (one(T) + log(params.sdist.spline(v)))
    # else
    #     return zero(T)
    # end
end

# function compute_J(sdist::SplineDistribution{1,2})
#     T = eltype(sdist)
#     int = zeros(T, size(sdist))
#     B = sdist.basis
#     d_start = T(BSplineKit.knots(B)[1])
#     d_end = T(BSplineKit.knots(B)[end])
#     domain = ([d_start, d_start], [d_end, d_end])

#     for k in 1:size(sdist)
#         i, j = ij_from_k(k, length(B))
#         integrand(v) = integrand_J(v, B, i, j, sdist)
#         # integand(v) = B[i, T](v[1]) * B[j, T](v[2]) * (1. + log(abs(sdist.spline(v))))
#         int[k], _ = HCubature.hcubature(integrand,[d_start, d_start], [d_end, d_end], atol = 1e-6, rtol=1e-6)
#     end

#     ldiv!(sdist.mass_fact, int)

#     return int
# end

function compute_J_gl(sdist::SplineDistribution{1,2}, n)
    T = eltype(sdist)
    int = zeros(T, size(sdist))

    Threads.@threads for k in 1:size(sdist)
        i, j = ij_from_k(k, length(sdist.basis))
        params = (sdist = sdist, B = sdist.basis, i = i, j = j)
        int[k] = gauss_quad_2d(integrand_J, sdist.basis, n, params)
    end

    ldiv!(sdist.mass_fact, int)

    return int
end

# function compute_U(v_α::AbstractVector{T}, v_β::AbstractVector{T}) where {T}
#     n = length(v_α)
#     U = zeros(T, (n,n))

#     norm_diff = euclidean(v_α, v_β)

#     if v_α != v_β
#         for i in CartesianIndices(U)
#             if i[1] == i[2]
#                 U[i] += 1 / norm_diff
#             end
#             U[i] -= (v_α[i[1]] - v_β[i[1]]) * (v_α[i[2]] - v_β[i[2]]) / norm_diff^3
#         end
#     end

#     return U
# end


function compute_U!(U, v_α, v_β)
    norm_diff = euclidean(v_α, v_β)

    U .= 0

    if v_α != v_β
        for i in axes(U,1)
            for j in axes(U,2)
                if i == j
                    U[i,j] += 1 / norm_diff
                end
                U[i,j] -= (v_α[i] - v_β[i]) * (v_α[j] - v_β[j]) / norm_diff^3
            end
        end
    end

    return U
end

# function compute_U(v_α, v_β)
#     n = length(v_α)
#     U = zeros(eltype(v_α), (n,n))
#     compute_U!(U, v_α, v_β)
# end

function compute_U(v_α::AbstractVector{T}, v_β::AbstractVector{T}) where {T}
    norm_diff = euclidean(v_α, v_β)

    if v_α != v_β
        U11 = - (v_α[1] - v_β[1]) * (v_α[1] - v_β[1]) / norm_diff^3 + inv(norm_diff)
        U12 = - (v_α[1] - v_β[1]) * (v_α[2] - v_β[2]) / norm_diff^3
        U21 = - (v_α[2] - v_β[2]) * (v_α[1] - v_β[1]) / norm_diff^3
        U22 = - (v_α[2] - v_β[2]) * (v_α[2] - v_β[2]) / norm_diff^3 + inv(norm_diff)

        return @SMatrix [ U11  U12 ;
                          U21  U22 ]
    else
        return @SMatrix [ zero(T)  zero(T) ;
                          zero(T)  zero(T) ]
    end
end


# particle-to-particle version
# function Landau_rhs(v, params)
#     # computes rhs for a single particle, assuming that the projection and other particle velocities are taken from the previous timestep 
#     # params.L is the vector L_k, which depends on the projection
#     # params.idist.particles.v is the particle velocities
#     # params.fdist.basis is the spline basis
#     v̇ = zero(v)
#     K = size(params.sdist) # length of tensor product basis (this is the square of the 1d basis length)
#     n = length(v)
#     U = zeros(n,n)

#     ind_s, res_s = evaluate_der_2d(params.B, v)

#     for α in axes(params.v_array, 2)
#         vα = params.v_array[:,α]
#         compute_U!(U, v, vα)
#         ind_α, res_α = evaluate_der_2d(params.B, vα)

#         for (i, k) in pairs(ind_s)
#             if k > 0 && k ≤ K
#                 v̇ .+= params.dist.particles.w[1,α] * params.L[k] * (U * res_s[:, i])
#             end
#         end

#         for (i, k) in pairs(ind_α)
#             if k > 0 && k ≤ K
#                 v̇ .-= params.dist.particles.w[1,α] * params.L[k] * (U * res_α[:, i])
#             end
#         end
#     end

#     return v̇
# end


function compute_K(v_array::AbstractArray{T}, dist, sdist) where {T}
    M = size(sdist)
    K1 = zeros(T, (M, size(v_array,2)))
    K2 = zeros(T, (M, size(v_array,2)))

    Threads.@threads for α in axes(v_array, 2)
        klist, der_array = evaluate_der_2d(sdist.basis, v_array[:,α])
        for (i, k) in pairs(klist)
            if k > 0 && k <= M
                K1[k,α] = dist.particles.w[1,α] * der_array[1,i]
                K2[k,α] = dist.particles.w[1,α] * der_array[2,i]
            end
        end
    end

    return K1, K2
end

function compute_K_plus(v_array::AbstractArray{T}, dist, sdist) where {T}
    K1, K2 = compute_K(v_array, dist, sdist)

    if rank(K1) < size(sdist) || rank(K2) < size(sdist)
        println("K1 or K2 not full rank")
        @show size(K1,1) - rank(K1)
        @show size(K2,1) - rank(K2)
    end

    return pinv(K1), pinv(K2)
end

# function L_integrand_vec(v::AbstractVector{T}, params) where T
#     v1 = [v[1], v[2]]
#     v2 = [v[3], v[4]]
    
#     id_list_1 = evaluate_der_2d_indices(params.sdist.basis, v1)
#     id_list_2 = evaluate_der_2d_indices(params.sdist.basis, v2)
    
#     if (params.k[1] in id_list_1 || params.k[1] in id_list_2) && (params.k[2] in id_list_1 || params.k[2] in id_list_2)
#         # U = compute_U(v1, v2)
#         basis_derivative = zeros(T, 2)
#         eval_bfd!(basis_derivative, params.sdist.basis, params.k[1], v1, 0, 1)
#         eval_bfd!(basis_derivative, params.sdist.basis, params.k[1], v2, 1, -1)

#         integrand = transpose(basis_derivative) * params.sdist.spline(v1) * compute_U(v1, v2)
#         # integrand = transpose(basis_derivative) * params.sdist.spline(v1) * compute_U(v1, v2)

#         eval_bfd!(basis_derivative, params.sdist.basis, params.k[2], v1, 0, 1)
#         eval_bfd!(basis_derivative, params.sdist.basis, params.k[2], v2, 1, -1)

#         return (integrand * params.sdist.spline(v2) * basis_derivative)::T

#     else
#         return zero(T)
#     end
# end


# function compute_L_ij(sdist)
#     T = eltype(sdist)
#     L = zeros(T, (size(sdist), size(sdist)))
#     B = sdist.basis
#     M = length(B)
#     knots = T.(BSplineKit.knots(sdist.basis))
#     # d_start = T(BSplineKit.knots(sdist.basis)[1])
#     # d_end = T(BSplineKit.knots(sdist.basis)[end])
#     # domain = ([d_start, d_start, d_start, d_start], [d_end, d_end, d_end, d_end])
    
#     Threads.@threads for k in CartesianIndices(L)
#         i1, j1 = ij_from_k(k[1], M)
#         i2, j2 = ij_from_k(k[2], M)
#         if k[1] ≤ k[2] && length(BSplines.common_support(B[i1], B[i2])) > 1 && length(BSplines.common_support(B[j1], B[j2])) > 1

#             irange = knots[BSplines.common_support(B[i1], B[i2])]
#             jrange = knots[BSplines.common_support(B[j1], B[j2])]

#             domain = ([irange[1], jrange[1], irange[1], jrange[1]], [irange[end], jrange[end], irange[end], jrange[end]])

#             params = (k = k, sdist = sdist)
#             prob = IntegralProblem(L_integrand_vec, domain, params)
#             sol = Integrals.solve(prob, HCubatureJL(); abstol = 1e-5, reltol = 1e-5)
#             L[k] = sol.u
#         end
#     end

#     Threads.@threads for k in CartesianIndices(L)
#         if k[1] > k[2] 
#             L[k] = L[k[2], k[1]]
#         end
#     end

#     return L .* 0.5
# end


function L_integrand_gh(v1::AbstractVector{T}, v2::AbstractVector{T}, sdist, i, j) where T
    basis_derivative1 = eval_bfd(sdist.basis, i, v1) - eval_bfd(sdist.basis, i, v2)
    basis_derivative2 = eval_bfd(sdist.basis, j, v1) - eval_bfd(sdist.basis, j, v2)

    sdist.spline(v1) * dot(basis_derivative1, compute_U(v1, v2) * basis_derivative2) * sdist.spline(v2)
end

function L_integrand_gh(v1::AbstractVector{T}, v2::AbstractVector{T}, params) where T
    id_list_1 = evaluate_der_2d_indices(params.sdist.basis, v1)
    id_list_2 = evaluate_der_2d_indices(params.sdist.basis, v2)
    
    if (params.k[1] in id_list_1 || params.k[1] in id_list_2) && (params.k[2] in id_list_1 || params.k[2] in id_list_2)
        L_integrand_gh(v1, v2, params.sdist, params.k[1], params.k[2])
    else
        return zero(T)
    end
end

function compute_L_ij_gh(sdist::SplineDistribution{1,2}, n::Int)
    T = eltype(sdist)
    L = zeros(T, (size(sdist), size(sdist)))
    # B = sdist.basis
    # M = length(B)

    Threads.@threads for i in axes(L,1)
        for j in axes(L,2)[i:end]
            # i1, j1 = ij_from_k(i, M)
            # i2, j2 = ij_from_k(j, M)

            # iknots = BSplines.common_support(B[i1], B[i2])
            # jknots = BSplines.common_support(B[j1], B[j2])

            params = (k = (i,j), sdist = sdist)
            L[i,j] = gauss_quad(L_integrand_gh, sdist.basis, n, params) / 2
        end
    end

    for i in axes(L,1)
        for j in axes(L,2)[begin:i-1]
            L[i,j] = L[j,i]
        end
    end

    return L
end


# spline-to-spline? version 
function Landau_rhs_2!(v̇, v::AbstractArray{ST}, params) where {ST}

    # project v onto params.sdist
    # println("sdist")
    sdist = params.ent.cache[ST]
    # println("projection")
    S = projection(v, params.dist, sdist)

    # compute K matrices 
    # println("compute K")
    # K1_plus, K2_plus = compute_K_plus(v, params.dist, sdist)
    K1, K2 = compute_K(v, params.dist, sdist)

    if rank(K1) < size(sdist) || rank(K2) < size(sdist)
        println("K1 or K2 not full rank")
        @show size(K1,1) - rank(K1)
        @show size(K2,1) - rank(K2)
    end

    # compute L_ij matrix
    # println("computing L")
    # Lij = compute_L_ij(sdist)
    # Lij = compute_L_ij_new(sdist)
    Lij = compute_L_ij_gh(sdist, params.n)

    # compute J vector
    # println("computing J")
    # J = compute_J(sdist)
    J = compute_J_gl(sdist, params.n)
    # J = compute_J_gl(sdist, params.sdist2, params.n)
    # J = compute_J_M(sdist)

    # solve for vector field
    # v̇[1,:] .= K1_plus * Lij * J  
    # v̇[2,:] .= K2_plus * Lij * J
    # v̇[1,:] .= -1 .* (K1_plus * (Lij * J))
    # v̇[2,:] .= -1 .* (K2_plus * (Lij * J))

    v̇[1,:] .= K1 \ (Lij * J)
    v̇[2,:] .= K2 \ (Lij * J)

    # v̇[1,:] .*= -1
    # v̇[2,:] .*= -1

    return v̇
end
