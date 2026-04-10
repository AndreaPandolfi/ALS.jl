using Plots
using DataFrames, CSV
using StatsBase
cdir = joinpath("paper", "output")

α = .3
for k in [1, 3], sparsity in [false, true]

    input = sparsity ? joinpath(cdir, "als_iters_k=$(k)_sparse") : joinpath(cdir, "als_iters_k=$k")
    results_als = CSV.read(input * ".csv", DataFrame)

    input2 = sparsity ? joinpath(cdir, "alsGC_iters_k=$(k)_sparse") : joinpath(cdir, "alsGC_iters_k=$k")
    results_alsGC = CSV.read(input2 * ".csv", DataFrame)

    input3 = sparsity ? joinpath(cdir, "alsColGC_iters_k=$(k)_sparse") : joinpath(cdir, "alsColGC_iters_k=$k")
    results_alsColGC = CSV.read(input3 * ".csv", DataFrame)

    function plot_iters!(plt, df::DataFrame, label::String, color::Int)
        summary = combine(
            groupby(df, :n),
            :iterations => mean => :iterations,
            :iterations => (x -> quantile(x, 0.9)) => :upper_decile,
            :iterations => (x -> quantile(x, 0.1)) => :lower_decile
        )

        plot!(plt, 2 .* summary.n, summary.iterations, label="",
            fillrange = summary.upper_decile, alpha=0.2, color=color)
        plot!(plt, 2 .* summary.n, summary.iterations, label="",
            fillrange = summary.lower_decile, alpha=0.2, color=color)
        plot!(plt, 2 .* summary.n, summary.iterations, label=label, linewidth=2, color=color)
    end

    title = sparsity ? "k=$(k), p = n^(-$(α))" : "k=$(k), p = 0.25"
    plt = plot(xlabel = "n", ylabel = "Iterations", ylims=(0, maximum(results_als.iterations) * 1.1), 
        title = title)
    plot_iters!(plt, results_als, "ALS", 1)
    (k>1) && plot_iters!(plt, results_alsColGC, "ALS-ColGC", 2) # for k=1, ALS-ColGC is the same as ALS-GC
    plot_iters!(plt, results_alsGC, "ALS-GC", 3)

    savefig(plt, input * ".pdf")
end