using LinearAlgebra, SparseArrays
using ALS
using StatsBase, Distributions
using Plots

## Check spectrum of Y (in particular, σ_1 ~ n, σ_2 ~ √n)
ns = [100, 200, 500, 1000]
k=1
σ1s = zeros(length(ns))
σ2s = zeros(length(ns))
for (j, n) in enumerate(ns)
    m = n 
    U0 = randn(m, k); V0 = randn(n, k)
    println("norm U0 = $(norm(U0)), norm V0 = $(norm(V0))")
    println("norm U0 * V0' = $(norm(U0 * V0'))")
    Ω = sprand(Bool, m, n, 1.)
    Y = Ω .* (U0 * V0' + .1 .* rand(Normal(), m, n))

    # find two largest singular values of Y
    eigvals_Y = eigvals(Matrix(Y * Y'))
    sorted_eigvals_Y = sort(sqrt.(abs.(eigvals_Y)), rev=true)
    println("n = $n: Largest singular values of Y = $(sorted_eigvals_Y[1:2])")
    σ1s[j] = sorted_eigvals_Y[1]
    σ2s[j] = sorted_eigvals_Y[2]
end
plot(ns, σ1s, label="σ_1", xscale=:log10, yscale=:log10)
plot!(ns, ns, label="n", style=:dash, xscale=:log10, yscale=:log10)
plot!(ns, σ2s, label="σ_2", xscale=:log10, yscale=:log10)
plot!(ns, σ2s[1]/sqrt(ns[1]) .* sqrt.(ns), style=:dash,label="sqrt(n)", xscale=:log10, yscale=:log10)


## Check convergence of Gram matrices
n = 1000; k = 2; m = n
U0 = randn(m, k); V0 = randn(n, k)
Ω = sprand(Bool, m, n, .5)
Y = Ω .* (U0 * V0' + .1 .* rand(Normal(), m, n))

U = randn(m, k); V = randn(n, k)
println("initial dist: $(norm(U0 * V0' - U * V'))")
als = ALSIterable(Y, copy(U), copy(V), tol=1e-05, τU=1., τV=1.)
iteration_als = 0
gramU_als = []
gramV_als = []
for (iter, Δ) in enumerate(als)
    iteration_als = iter
    iter % 50 == 0 && println("ALS iteration: $iter, Δ: $Δ")
    push!(gramU_als, als.U' * als.U)
    push!(gramV_als, als.V' * als.V)
end
println("als dist: $(norm(U0 * V0' - als.U * als.V'))")

alsGC = ALSIterable(Y, copy(U), copy(V); Calibrate! =ALS.GramCalibration!, tol=1e-05, τU=1., τV=1.)
iteration_alsGC = 0
gramU_alsGC = []
gramV_alsGC = []
for (iter, Δ) in enumerate(alsGC)
    iteration_alsGC = iter
    iter % 50 == 0 && println("ALS-GC iteration: $iter, Δ: $Δ")
    push!(gramU_alsGC, alsGC.U' * alsGC.U)
    push!(gramV_alsGC, alsGC.V' * alsGC.V)
end
println("alsGC dist: $(norm(U0 * V0' - alsGC.U * alsGC.V'))")

function plot_gram_convergence(gramUs, gramVs, i::Int, j::Int)
    gramU_ij = [gramU[i,j] for gramU in gramUs]
    gramV_ij = [gramV[i,j] for gramV in gramVs]
    plot(gramU_ij, label="U'U_($i,$j)", title="Convergence of Gram Matrices", yscale=:log10)
    plot!(gramV_ij, label="V'V_($i,$j)")
end

plot_gram_convergence(gramU_als, gramV_als, 1, 1)
plot_gram_convergence(gramU_als, gramV_als, 1, 2)
plot_gram_convergence(gramU_als, gramV_als, 2, 2)