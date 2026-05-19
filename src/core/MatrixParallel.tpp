#include <sycl/sycl.hpp>
#include "Matrix.hpp"
#include "Tensor.hpp"
#include "TensorParallel.hpp"

namespace gema{

    template <class T, TensorConcept TensorContainer>
    void Matrix<T, TensorContainer>::matrixMultiplication(const Matrix<T, TensorContainer>& otherMatrix)
    requires (!std::is_same_v<TensorContainer, Tensor<T>>){

        const uint64_t m = getDimensionSizes()[0];
        const uint64_t n = getDimensionSizes()[1];

        const uint64_t otherRows = otherMatrix.getDimensionSizes()[0];
        const uint64_t p = otherMatrix.getDimensionSizes()[1];

        if(n != otherRows){
            throw std::runtime_error("Matrix multiplication dimension mismatch.");
        }

        Matrix<T, TensorContainer> result(m, p);

        T* a = getData();
        const T* b = otherMatrix.getData();
        T* c = result.getData();

        sycl::queue* queue = tensor_.getQueue();

        queue->parallel_for(sycl::range<2>(m, p), [=](sycl::id<2> idx){

            const uint64_t i = idx[0];
            const uint64_t j = idx[1];

            T sum{};

            for(uint64_t k = 0; k < n; ++k){
                sum += a[(i * n) + k] * b[(k * p) + j];
            }

            c[(i * p) + j] = sum;
        }).wait();

        // for(uint64_t i = 0; i < m; ++i){
        //     for(uint64_t j = 0; j < p; ++j){

        //         T sum{};

        //         for(uint64_t k = 0; k < n; ++k){
        //             sum += getItem(i, k) * otherMatrix.getItem(k, j);
        //         }

        //         result.setItem(sum, i, j);
        //     }
        // }

        *this = std::move(result);
    }
    
}