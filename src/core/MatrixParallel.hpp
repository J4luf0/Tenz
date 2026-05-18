#ifndef MATRIX_PARALLEL_HPP
#define MATRIX_PARALLEL_HPP

#include "Tensor.hpp"
#include "TensorParallel.hpp"
#include "Matrix.hpp"

namespace gema{

template <typename T>
using MatrixParallel = Matrix<T, TensorParallel<T>>;

// template<class T>
// class MatrixParallel : public Matrix<T, TensorParallel<T>>{

//     private:

//     public:

//     using Matrix<T, TensorParallel<T>>::Matrix;

// };

}

#include "MatrixParallel.tpp"

#endif