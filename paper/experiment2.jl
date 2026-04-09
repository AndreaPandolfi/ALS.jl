using LinearAlgebra, SparseArrays
using Base.Threads
using ALS
using StatsBase, Distributions

Loss(it::ALSIterable) = ALS.Loss(it.Y, it.U, it.V)

function NormCalibration!(it::ALSIterable{T}) where T
    """ Rescale the columns norms of U and V"""
    U = it.U
    V = it.V

    colNormsU = sum(U.^2, dims=1) .|> sqrt |> vec
    colNormsV = sum(V.^2, dims=1) .|> sqrt |> vec

    # We want to rescale U and V s.t. τU * ||U[:, j]||^2 = τV * ||V[:, j]||^2 for all j. 
    # We compute S = (τV/τU)^(1/4) * (colNormsV ./ colNormsU).^(1/2) 
    S = (it.τV / it.τU)^(1/4) * sqrt.(colNormsV ./ colNormsU)
    
    # and rescale U by S and V by 1 ./ S.
    it.U = U * Diagonal(S) 
    it.V = V * Diagonal(1 ./ S)
end

function run_experiment2(π::Real, n::Int, k::Int; m::Int = n, τU=.1, τV=.1, tol=1e-05, maxiter=1000)
    # Define Y as the product U0 * V0' but with probability p of observing each entry
    U0 = randn(m, k); V0 = randn(n, k)
    Ω = sprand(Bool, m, n, π)
    Y = Ω .* (U0 * V0' + .1 .* rand(Normal(), m, n))
    U = randn(m, k); V = randn(n, k)
    
    als = ALSIterable(Y, copy(U), copy(V); τU=τU, τV=τV, tol=tol, maxiter=maxiter)
    iteration_als = 0
    for (iter, Δ) in enumerate(als)
        iteration_als = iter
    end
    
    # U = randn(m, k); V = randn(n, k)
    alsGC = ALSIterable(Y, copy(U), copy(V); Calibrate! =ALS.GramCalibration!, τU=τU, τV=τV, tol=tol, maxiter=maxiter)
    iteration_alsGC = 0
    for (iter, Δ) in enumerate(alsGC)
        iteration_alsGC = iter
    end
    
    alsColGC = ALSIterable(Y, copy(U), copy(V); Calibrate! = NormCalibration!, τU=τU, τV=τV, tol=tol, maxiter=maxiter)
    iteration_alsColGC = 0
    for (iter, Δ) in enumerate(alsColGC)
        iteration_alsColGC = iter
    end

    return Int(iteration_als), Int(iteration_alsGC), Int(iteration_alsColGC), als, alsGC, alsColGC
end

ns = [100, 200, 500, 1000]
n_experiments = 50
p = .25
k = 3
als_iters = zeros(n_experiments, length(ns))
alsGC_iters = zeros(n_experiments, length(ns))
alsColGC_iters = zeros(n_experiments, length(ns))
final_errors_als = zeros(n_experiments, length(ns))
final_errors_alsGC = zeros(n_experiments, length(ns))
final_errors_alsColGC = zeros(n_experiments, length(ns))

for (j, n) in enumerate(ns)
    println("Running experiments for n = $n")
    Threads.@threads for i in 1:n_experiments
        println("Experiment $i")
        iteration_als, iteration_alsGC, iteration_alsColGC, als, alsGC, alsColGC = run_experiment2(p, n÷2, k, m=2*n)
        als_iters[i , j] = iteration_als
        alsGC_iters[i , j] = iteration_alsGC
        alsColGC_iters[i , j] = iteration_alsColGC
        final_errors_als[i , j] = Loss(als)
        final_errors_alsGC[i , j] = Loss(alsGC)
        final_errors_alsColGC[i , j] = Loss(alsColGC)
    end
    # print column averages
    mean(als_iters, dims=1) |> println
    mean(alsColGC_iters, dims=1) |> println
    mean(alsGC_iters, dims=1) |> println
end


fname = joinpath("paper", "output", "als_iters_k=$k.csv")
open(fname, "w") do io
    println(io, "m,n,iterations, final error")
    for (j, n) in enumerate(ns)
        for i in 1:n_experiments
            println(io, "$(2*n),$(n÷2),$(als_iters[i, j]), $(final_errors_als[i, j])")
        end
    end
end

fname2 = joinpath("paper", "output", "alsGC_iters_k=$k.csv")
open(fname2, "w") do io
    println(io, "m,n,iterations, final error")
    for (j, n) in enumerate(ns)
        for i in 1:n_experiments
            println(io, "$(2*n),$(n÷2),$(alsGC_iters[i, j]), $(final_errors_alsGC[i, j])")
        end
    end
end

fname3 = joinpath("paper", "output", "alsColGC_iters_k=$k.csv")
open(fname3, "w") do io
    println(io, "m,n,iterations, final error")
    for (j, n) in enumerate(ns)
        for i in 1:n_experiments
            println(io, "$(2*n),$(n÷2),$(alsColGC_iters[i, j]), $(final_errors_alsColGC[i, j])")
        end
    end
end
