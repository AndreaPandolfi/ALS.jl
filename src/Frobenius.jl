export FrobeniusDistanceSq, lowrankFrob

function lowrankFrob(U1::Matrix, V1::Matrix, U2::Matrix, V2::Matrix)
    """ Computes Tr((U1*V1')' * (U2*V2')). Intended  when U1, V1, U2, V2 are matrices with many rows but few columns."""
    # FrobInner(U1*V1', U2*V2') = Tr((U1*V1')' * (U2*V2')) = Tr(V2' * V1 * U1' * U2) = Tr((V1' * V2)' * U1' * U2)
    # We use the Julia fuction dot(A, B) = Tr(A' * B)
    return dot(transpose(U1)*U2, transpose(V1)*V2)
end

function lowrankFrob(U::Matrix, V::Matrix)
    """ Computes the squared Frobenius norm of the low-rank matrix U*V'."""
    return lowrankFrob(U, V, U, V)
end

function FrobeniusDistanceSq(U1::Matrix, V1::Matrix, U2::Matrix, V2::Matrix)
    """ Computes the squared Frobenius distance between two low-rank matrices U1*V1' and U2*V2'."""
    # FrobDist(U1*V1', U2*V2')^2 = ||U1*V1' - U2*V2'||_F^2  = ||U1*V1'||_F^2 + ||U2*V2'||_F^2 - 2 * Tr((U1*V1')' * (U2*V2'))
    return lowrankFrob(U1, V1) + lowrankFrob(U2, V2) - 2 * lowrankFrob(U1, V1, U2, V2)
end

function Loss(Y::SparseMatrixCSC, U::Matrix, V::Matrix)
    """ Computes the loss ||Y - U*V'||_F^2 on the support of Y, where Y is a sparse matrix"""
    loss = 0.
    for col in 1:size(Y, 2)
        nzrows = Y.rowval[Y.colptr[col]: Y.colptr[col+1]-1]
        nzvals = Y.nzval[Y.colptr[col]: Y.colptr[col+1]-1]
        for (row, val) in zip(nzrows, nzvals)
            loss += (val - dot(U[row, :], V[col, :]))^2
        end
    end
    return loss
end