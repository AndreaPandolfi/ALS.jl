import Base: iterate

export ALSIterable, als!

const Factor{T} = Union{Matrix{T}}

mutable struct ALSIterable{T}
    U::Factor{T} # Left latent factor
    V::Factor{T} # Right latent factor
    Y::SparseMatrixCSC{T, Int} # Observation matrix
    τU::T # regularization parameter for U
    τV::T # regularization parameter for V
    Δ::T # Relative difference between the last two iterations. If smaller than tol, we stop.
    Calibrate!::Function # Function to perform calibration step. By default it is the identity
    tol::T
    maxiter::Int

    function ALSIterable(Y::SparseMatrixCSC{Tv,Ti}, U::Factor{Tv}, V::Factor{Tv}; Calibrate!::Function=identity, τU::AbstractFloat=1., τV::AbstractFloat=1., tol::AbstractFloat=1e-05, maxiter::Int=1000) where {Tv, Ti<:Integer}
        @assert size(U, 2) == size(V, 2) "U and V must have the same number of columns (latent factors dimension)"
        
        m, n = size(Y)

        tol = convert(Tv, 1e-5)
        τU = convert(Tv, τU)
        τV = convert(Tv, τV)

        Δ = one(Tv)

        return new{Tv}(U, V, Y, τU, τV, Δ, Calibrate!, tol, maxiter)
    end
    
    
    function ALSIterable(Y::SparseMatrixCSC{Tv,Ti}, nfactors::Int; kwargs...) where {Tv, Ti<:Integer}
        m, n = size(Y)

        U = randn(Tv, m, nfactors)
        V = randn(Tv, n, nfactors)

        return ALSIterable(Y, U, V; kwargs...)
    end
end


mutable struct SoftImputeALSIterable{T}
    U::Factor{T} # Left latent factor
    V::Factor{T} # Right latent factor
    Y::SparseMatrixCSC{T, Int} # Observation matrix
    τU::T # regularization parameter for U
    τV::T # regularization parameter for V
    Δ::T # Relative difference between the last two iterations. If smaller than tol, we stop.
    tol::T
    maxiter::Int
end

@inline converged(it::Union{ALSIterable, SoftImputeALSIterable}) = it.Δ ≤ it.tol

@inline start(it::Union{ALSIterable, SoftImputeALSIterable}) = 0

@inline done(it::Union{ALSIterable, SoftImputeALSIterable}, iteration::Int) = iteration ≥ it.maxiter || converged(it)


function ALSupdate_Ui!(it::ALSIterable{T}, i::Int) where T
    """ Updates the i-th component of the left latent fatctor U given the current estimate 
    of the right factor V."""

    U = it.U; V = it.V
    k = size(U, 2)
    norm_i = zeros(T, k,k)
    num_i = zeros(T, 1,k)
    Yi = it.Y[i,:]
    for (j, Yij) in zip(Yi.nzind, Yi.nzval)
        norm_i += transpose(V[j:j,:])*V[j:j, :]
        num_i += Yij.* V[j:j, :]
    end
    norm_i += Diagonal(it.τU .* ones(k))
    U[i:i,:] .= num_i / norm_i
    return 
end 

function ALSupdate_Vj!(it::ALSIterable{T}, j::Int) where T
    """ Updates the j-th component of the right factor V given the current estimate 
    of the left factor U."""

    U = it.U; V = it.V
    k = size(U, 2)
    norm_j = zeros(T, k,k)
    num_j = zeros(T, 1, k)
    Yj = it.Y[:,j]
    for (i, Yij) in zip(Yj.nzind, Yj.nzval)
        norm_j += transpose(U[i:i,:])*U[i:i, :]
        num_j += Yij.* U[i:i, :]
    end
    norm_j += Diagonal(it.τV .* ones(k))
    V[j:j,:] .=  num_j / norm_j
    return 
end 

function ALSupdate_U!(it::ALSIterable{T}) where T
    """ Updates the left latent factor U given the current estimate of the right factor V."""
    m = size(it.U, 1)
    for i in 1:m
        ALSupdate_Ui!(it, i)
    end
    return 
end

function ALSupdate_V!(it::ALSIterable{T}) where T
    """ Updates the right latent factor V given the current estimate of the left factor U."""
    n = size(it.V, 1)
    for j in 1:n
        ALSupdate_Vj!(it, j)
    end
    return 
end

function update_Δ!(it::Union{ALSIterable, SoftImputeALSIterable}, Uold::Matrix, Vold::Matrix)
    """ Updates the relative difference between the last two iterations."""
    # Compute the relative difference
    Unew = it.U
    Vnew = it.V

    dist_square = FrobeniusDistanceSq(Uold, Vold, Unew, Vnew) / max(lowrankFrob(Uold, Vold), sqrt(eps(typeof(it.Δ)))) 
    it.Δ = sqrt(dist_square)
    return
end

function GramCalibration!(it::ALSIterable{T}) where T
    """ Projects the current estimate of the left and right latent factors onto the Gram manifold."""
    U = it.U
    V = it.V

    qrU = qr(U)
    qrV = qr(V)

    L, S, R = svd(qrU.R * transpose(qrV.R))

    U = (it.τV / it.τU)^(1/4) * Matrix(qrU.Q) * L * Diagonal(sqrt.(S)) 
    # Need to put Matrix(Q) since Q is not stored as a matrix (see https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.QR)
    # Not sure if this is fast.
    V = (it.τU / it.τV)^(1/4) * Matrix(qrV.Q) * R * Diagonal(sqrt.(S))

    it.U = U
    it.V = V
end

###############
##    ALS    ##
###############
function iterate(it::ALSIterable{T}, iteration::Int=start(it)) where T
    # Check for termination first
    if done(it, iteration)
        return nothing
    end

    Uold = copy(it.U)
    Vold = copy(it.V)

    # Update U
    ALSupdate_U!(it)
    # Update V
    ALSupdate_V!(it)

    # Project onto the Gram manifold if required
    it.Calibrate!(it)

    # Update the relative difference with prev. iteration
    update_Δ!(it, Uold, Vold)

    return it.Δ, iteration + 1
end


function als!(Y::SparseMatrixCSC{Tv,Ti}, U::Factor{Tv}, V::Factor{Tv}; Calibrate!::Function=identity, τU::AbstractFloat=1., τV::AbstractFloat=1., tol::AbstractFloat=1e-05, maxiter::Int=1000) where {Tv, Ti<:Integer}
    """ Runs the ALS algorithm for low-rank matrix completion.
    Y: [m,n] observation matrix
    U: starting point for the left latent factor
    V: starting point for the right latent factor
    Calibrate!: function to perform calibration step. By default it is the identity, but it can be set to GramCalibration! for example.
    τU: regularization parameter for U
    τV: regularization parameter for V
    tol: stopping criterion for ALS
    maxiter: maximum number of iterations for ALS

    Returns:
    U: Final estimate for the left latent factor given by ALS
    V: Final estimate for the right latent factor given by ALS
    """
    it = ALSIterable(Y, U, V; Calibrate! =Calibrate!, τU=τU, τV=τV, tol=tol, maxiter=maxiter)
    
    for (iteration, Δ) in enumerate(it)
        iteration % 50 == 0 && println("ALS iteration: $iteration, Δ: $Δ")
    end
    # println("Converged in $iteration iterations")
    return (it.U, it.V)
end
