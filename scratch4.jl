using Plots, LinearAlgebra, Distributions, QuadGK

function A_ela(l_mat, k=1., l=1.)
    A = zeros(size(l_mat))
    for i in axes(l_mat)[1]
        if i - 1 < 1
            j = axes(l_mat)[1][end]
        else
            j = i - 1
        end
        if i + 1 > axes(l_mat)[1][end]
            h = 1
        else
            h = i + 1
        end
        d_xy_prev = l_mat[j, :] - l_mat[i, :]
        d_p_hat = d_xy_prev/norm(d_xy_prev)
        d_xy_next = l_mat[h, :] - l_mat[i, :]
        d_n_hat = d_xy_next/norm(d_xy_prev)

        A_xy_prev = (norm(d_xy_prev) - l)*d_p_hat
        A_xy_next = (norm(d_xy_next) - l)*d_n_hat

        A[i, :] = k*(A_xy_prev + A_xy_next)
    end
    return A
end

rhat(l_ij::Vector{Float64}) = l_ij/norm(l_ij)

#f_basis(n, dsi) = sign(cos(n*dsi/2) - cos(3*n*dsi/2))*(1 - cos(2*n*dsi))
f_basis(n, dsi) = (-1)^floor((abs(n*dsi/π) - 1 ) / 2) * (1 - cos(2*n*dsi))
f_mag(n_vec::Vector{Int}, dsi::Float64) = sum(f_basis(n, dsi) for n in n_vec)

function f_vec(n_vec::Vector{Int}, dsi::Float64, li::Vector{Float64}, lj::Vector{Float64})
    l_ij = li - lj
    return f_mag(n_vec, dsi) * rhat(l_ij)
end

function a_int(l_mat::Matrix{Float64}, n_vec::Vector{Int}, s_vec, i::Int, j::Int, k::Float64=1.)
    println(i, ", ", j)
    li = l_mat[i, :]
    lj = l_mat[j, :]

    integrand(s) = f_vec(n_vec, s_vec[i] - s, li, lj)
    result, _ = sqrt(2π)*k/2 .* quadgk(integrand, [0., 2π])

    return result#collect(hcat(resultx, resulty)')
end

function A_int(l_mat::Matrix{Float64}, n_vec::Vector{Int}, s_vec, k::Float64=1.)
    A_sum = zeros(Float64, size(l_mat))
    for i in axes(l_mat)[1]
        println(i)
        for j in axes(l_mat)[1]
            if i == j
                continue
            end
            #println(size(a_int(l_mat, n_vec, s_vec, i, j, k)))
            A_sum[i, :] += a_int(l_mat, n_vec, s_vec, i, j, k)
        end
    end

    return A_sum
end

function l_fcn_factory_nearest(l_mat::Matrix{Float64})
    N = size(l_mat, 1)
    return (s::Float64) -> l_mat[Int(round(s*(N-1)/(2π)))+1, :]
end

function stormer_verlet_step!(x, v, a, a_fcn, Δt::Float64)
    x .+= v .* Δt .+ 0.5 .* a .* Δt^2
    a_new = a_fcn(x)
    v .+= 0.5*(a .+ a_new) .* Δt
    return a_new
end

function init_points(n, mx=1, my=1)
    s_vec = 0.:2π/n:2π*(1-1/n)
    return hcat(sin.(mx*s_vec), cos.(my*s_vec)), s_vec
end
init_points(n; mx=1, my=1) = init_points(n, mx, my)

n = 20
x, s_vec = init_points(n, 1, 3)
v = zeros(size(x))
plot_loop(x, xlims=[-2,2], ylims=[-2,2]) = plot(vcat(x[:, 1], x[1, 1]), vcat(x[:, 2], x[1, 2]), xlims=xlims, ylims=ylims)
a_fcn = (x) -> A_ela(x, 10., 0.3) + A_int(x, [1], s_vec, -10.)
anim = @animate for i in 1:100
    println(i)
    stormer_verlet_step!(x, v, a_fcn(x), a_fcn, 0.01)
    p = plot_loop(x, [-2,2], [-2,2])
    display(p)
end
gif(anim, "elastic.mp4", fps = 30)