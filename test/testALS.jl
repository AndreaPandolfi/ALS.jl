using Test
using LinearAlgebra, SparseArrays
using ALS

@testset "Froebenius tools" begin
    n = 1000; k = 4
    A = randn(n, k)
    B = randn(n, k)
    C = randn(n, k)
    D = randn(n, k)
    # Test lowrankFrob
    @test FrobeniusDistanceSq(A, B, C, D) ≈ norm(A*B' - C*D')^2 atol=1e-6
    @test lowrankFrob(A, B) ≈ norm(A*B')^2 atol=1e-6

    # Test Loss
    Y = sprandn(n, n, 0.01)
    L, S, R = svd(Matrix(Y))
    @test ALS.Loss(Y, L * Diagonal(S), Matrix(R)) ≈ 0. atol=1e-6
end;

@testset "ALSIterable" begin
    m = 2000; n = 1000; k = 4; τU=.1; τV=.1
    
    Y = sprandn(m, n, 0.01)

    it = ALSIterable(Y, k, τU=τU, τV=τV, tol=1e-05, maxiter=1000)

    @test size(it.U) == (m, k) || error("U has wrong size")
    @test size(it.V) == (n, k) || error("V has wrong size")

    @test it.maxiter == 1000 || error("maxiter has wrong value")
    @test it.tol == 1e-5 || error("tol has wrong value")
    @test it.τU == τU || error("τU has wrong value")
    @test it.τV == τV || error("τV has wrong value")
end;


@testset "ALS" begin
    m = 500; n = 200; k = 2; τU=.1; τV=.1

    U0 = randn(m, k); V0 = randn(n, k) .+ 2.

    p = .1
    # Define Y as the product U0 * V0' but with probability p of observing each entry
    Ω = sprand(Bool, m, n, p)
    Y = Ω .* (U0 * V0')

    U = randn(m, k); V = randn(n, k)
    
    # it = ALSIterable(Y, copy(U), copy(V), τU=τV, τV=τV, ProjGram=true, tol=1e-07, maxiter=1000)
    # # Check update_Ui
    # i = 1; 
    # norm_i = .1 + sum(Ω[i,:] .* (V .^ 2))
    # num_i = sum(Y[i,:] .* V)
    # newUi = num_i / norm_i

    # ALSupdate_Ui!(it, i)
    
    # it.U[i,1] ≈ newUi || error("update_Ui did not work")  
    
    norm(U0 * V0' - U*V') / norm(U0*V0') |> println
 
    U = randn(m, k); V = randn(n, k)
    U, V = als!(Y, U, V, Calibrate! =identity, tol=1e-6, maxiter=1000, τU=τU, τV=τV)
    @test norm(U0 * V0' - U*V')/norm(U0 * V0') < .1 
    
    U = randn(m, k); V = randn(n, k)
    U, V = als!(Y, U, V, Calibrate! = ALS.GramCalibration!, tol=1e-6, maxiter=1000, τU=τU, τV=τV)
    @test norm(U0 * V0' - U*V')/norm(U0 * V0') < .1
end;

@testset "GramCalibration" begin
    m = 2000; n = 1000; k = 4
    
    Y = sprandn(m, n, 0.01)

    it = ALSIterable(Y, k, Calibrate! = ALS.GramCalibration!)

    @test !(it.τU * it.U' * it.U ≈ it.τV * it.V' * it.V)

    ALS.GramCalibration!(it)

    @test it.τU * it.U' * it.U ≈ it.τV * it.V' * it.V || error("Gram calibration did not work")
end