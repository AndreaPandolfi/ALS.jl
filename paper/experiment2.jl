using LinearAlgebra, SparseArrays
using Base.Threads
using DataFrames, CSV
using ALS
using StatsBase, Distributions

cdir = joinpath("paper", "output")
## Auxiliary functions
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

function run_experiment2(π::Real, n::Int, k::Int; m::Int = n, τU=1., τV=1., tol=1e-05, maxiter=1000)
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

    # computing test loss
    Ω_c = dropzeros(.! Ω)
    test_loss_als = ALS.Loss(Ω_c .* (U0 * V0'), als.U, als.V)
    test_loss_alsGC = ALS.Loss(Ω_c .* (U0 * V0'), alsGC.U, alsGC.V)
    test_loss_alsColGC = ALS.Loss(Ω_c .* (U0 * V0'), alsColGC.U, alsColGC.V)

    return Int(iteration_als), Int(iteration_alsGC), Int(iteration_alsColGC), test_loss_als, test_loss_alsGC, test_loss_alsColGC, als, alsGC, alsColGC
end

## Running experiments and saving results
# coefficient of sparsity (i.e. p = n ^ - α)
sparsity = false; α = .3
k = 3

ns = [100, 200, 500, 1000]
n_experiments = 50
τ = 5.

# initialize dataframe with columns m, n, p, k, iterations, train_loss, test_loss
results_als = DataFrame(
    m = Int[], n = Int[], p = Float64[], k = Int[], τ = Float64[], 
    iterations = Int[], train_loss = Float64[], test_loss = Float64[],
)
results_alsGC = copy(results_als)
results_alsColGC = copy(results_als)

for n in ns
    println("Running experiments for n = $n")
    sparsity ? (p = n ^ -α) : (p = .25)

    Threads.@threads for i in 1:n_experiments
        println("Experiment $i")
        iters_als, iters_alsGC, iters_alsColGC, test_als, test_alsGC, test_alsColGC, als, alsGC, alsColGC = run_experiment2(p, n÷2, k, m=2*n, τU=τ, τV=τ)
        push!(results_als, (2*n, n÷2, p, k, τ, iters_als, Loss(als), test_als))
        push!(results_alsGC, (2*n, n÷2, p, k, τ, iters_alsGC, Loss(alsGC), test_alsGC))
        push!(results_alsColGC, (2*n, n÷2, p, k, τ, iters_alsColGC, Loss(alsColGC), test_alsColGC))
    end
end


output = sparsity ? joinpath(cdir, "als_iters_k=$(k)_sparse.csv") : joinpath(cdir, "als_iters_k=$k.csv")
CSV.write(output, results_als)

output2 = sparsity ? joinpath(cdir, "alsGC_iters_k=$(k)_sparse.csv") : joinpath(cdir, "alsGC_iters_k=$k.csv")
CSV.write(output2, results_alsGC)

output3 = sparsity ? joinpath(cdir, "alsColGC_iters_k=$(k)_sparse.csv") : joinpath(cdir, "alsColGC_iters_k=$k.csv")
CSV.write(output3, results_alsColGC)
