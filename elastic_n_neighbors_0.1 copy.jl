using Plots, LinearAlgebra, Distributions

mod1(x, b) = ((x - 1) % b + b) % b + 1

function A_ela(l_mat, num_neighbors=1, k=1., l=1., g=1., r=1.)
    A = zeros(size(l_mat))
    n = size(l_mat, 1)
    for i in 1:n
        for q = 1:num_neighbors
            j = mod1(i-q, n)
            h = mod1(i+q, n)

            d_xy_prev = l_mat[j, :] - l_mat[i, :]
            d_p_hat = d_xy_prev/norm(d_xy_prev)
            d_xy_next = l_mat[h, :] - l_mat[i, :]
            d_n_hat = d_xy_next/norm(d_xy_next)

            A_xy_prev_mag = (norm(d_xy_prev) - q*l)
            A_xy_next_mag = (norm(d_xy_next) - q*l)

            A_xy_prev = A_xy_prev_mag*d_p_hat
            A_xy_next = A_xy_next_mag*d_n_hat

            A[i, :] += (k*q^(1))*(A_xy_prev + A_xy_next)
        end
        A[i, :] += g*l_mat[i, :] - r*l_mat[i, :]*norm(l_mat[i, :])
    end
    return A
end
