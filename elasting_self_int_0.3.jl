using Colors, Plots, LinearAlgebra, Distributions, QuadGK

rotator(θ::Real) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
R90 = rotator(pi/2)

#include("elastic_nearest_neighbor_0.1.jl")
include("elastic_n_neighbors_0.1 copy.jl")

f_basis(n, dsi) = sign(n)*(-1)^floor((abs(2*(-n)*dsi/π) + 1) / 2) * (1 - cos(4*n*dsi))
#f_basis(n, dsi) = sin(n*dsi)^2
f_mag(n_vec::Vector{Int}, dsi::Float64) = sum(f_basis(n, dsi) for n in n_vec)

function a_int_mat(l_mat::Matrix{Float64}, n_vec::Vector{Int}, s_vec, i::Int, k::Float64=1.)
    si = s_vec[i]
    accumulator = [0., 0.]
    for j in axes(l_mat)[1]
        if i == j
            continue
        end
        lij = l_mat[i, :] - l_mat[j, :]
        sj = s_vec[j]
        ds = si - sj
        accumulator += f_mag(n_vec, ds) * normalize(lij) / (norm(lij) + 0.1)
    end

    return sqrt(2π)*k/2 * accumulator 
end

function A_int_mat(l_mat::Matrix{Float64}, n_vec::Vector{Int}, s_vec::Vector{Float64}, k::Float64=1.)
    # The loops here & in a_int_mat can likely be simplified by using equal & opposite results for each (i, j) pair.
    A_sum = zeros(Float64, (length(s_vec), 2))
    for i in axes(s_vec)[1]
        A_sum[i, :] += a_int_mat(l_mat, n_vec, s_vec, i, k)
    end

    return A_sum
end

function verlet_step!(x, v, a, a_fcn, Δt::Float64)
    x .+= v .* Δt .+ 0.5 .* a .* Δt^2
    a_new = a_fcn(x)
    v .+= 0.5*(a .+ a_new) .* Δt
    return a_new
end

function init_points(n, mx=(1, 0), my=(0, 1), amplitudex=(1., 0.), amplitudey=(0., 1.), amplituder=(0., 0.), mr=(8, 6))
    s_vec = 0.:2π/n:(2π*(1-1/(n+1)))
    
    x_base = amplitudex[1] * sin.(mx[1]*s_vec) + amplitudex[2] * cos.(mx[2]*s_vec)
    y_base = amplitudey[1] * sin.(my[1]*s_vec) + amplitudey[2] * cos.(my[2]*s_vec)
    
    r = 1 .+ amplituder[1] * sin.(mr[1] * s_vec) + amplituder[2] * cos.(mr[2] * s_vec)
    
    x = r .* x_base
    y = r .* y_base
    
    return hcat(x, y), collect(s_vec)
end


function init_vel_rot(x::Matrix{Float64}, k::Float64)
    v = zeros(size(x))
    for i in axes(x)[1]
        xi = x[i, :]
        az = normalize(R90*xi)
        r = sqrt(norm(xi))
        v[i, :] = k*r*az
    end
    return v
end

function periodic_nearest_neighbor_interpolator(loop::Matrix{Float64})
    N = size(loop, 1)
    
    function interpolator(t::Real)
        t_wrapped = mod(t - 1, N) + 1
        idx = round(Int, t_wrapped)
        idx = ((idx - 1) % N) + 1
        
        return loop[idx, :]
    end
    
    return interpolator
end

T = 1000
n = 200
x, s_vec = init_points(n, (4, 1), (3, 3), (1.4,0.), (1., 1.), (0., 0.), (8, 0))
x .*= hcat(1.4 * ones(n), 1.4 * ones(n))
#x .+= rand(Uniform(-0.02, .02), size(x))
n_vec = [2]
v = init_vel_rot(x, 0.) .+ hcat(0. * ones(n), 0. * ones(n))

function plot_loop(x, xlims=[-2,2], ylims=[-2,2])
    p = plot()
    hspan!(p, [-3,3], color = :deepskyblue2, alpha = 0.4)
    plot!(p, vcat(x[:, 1], x[1, 1]), vcat(x[:, 2], x[1, 2]), size=(1600,1200), xlims=xlims, ylims=ylims, linecolor=:purple, legend=false, axis=([],false), linewidth=6.)
    return p
end
a_fcn = (x) -> A_ela(x, div(n, 5), 1. * T, 3. / n, 0., 100.) + A_int_mat(x, n_vec, collect(s_vec), 10. /length(n_vec))
anim = @animate for i in 1:T
    #println(i)
    verlet_step!(x, v, a_fcn(x), a_fcn, 0.0001)
    p = plot_loop(x, [-3,3], [-9/4,9/4])
    display(p)
end

function cleanstring(input_string::String)
    return replace(input_string, r"[[:punct:]]" => "_")
end

using Dates
gif(anim, "elastic_vids\\elastic"*cleanstring(string(now()))*".mp4", fps = 30)
