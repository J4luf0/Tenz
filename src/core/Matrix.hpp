#ifndef MATRIX_HPP
#define MATRIX_HPP

#include "Utils.hpp"
#include "TensorConcept.hpp"
#include "Tensor.hpp"

namespace gema{

template<class T, TensorConcept TensorContainer = Tensor<T>>
class Matrix : public AbstractOperation<Matrix<T, TensorContainer>>{

    using DataContainer = TensorContainer::DataContainer;
    using MetadataContainer = TensorContainer::MetadataContainer;

    private:

    TensorContainer tensor_;

    public:

    Matrix(const uint64_t x, const uint64_t y);

    Matrix(const Matrix<T, TensorContainer>& otherMatrix);

    Matrix(Matrix<T, TensorContainer>&& otherMatrix) noexcept;

    Matrix();



    const MetadataContainer& getDimensionSizes() const;

    uint64_t getNumberOfDimensions() const;

    uint64_t getNumberOfItems() const;

    T getItem(const uint64_t x, const uint64_t y);

    void setItem(const T& value, const uint64_t x, const uint64_t y);

    DataContainer& getData();

    //Matrix<T, TensorContainer>& setData(const DataContainer& matrixItems);
    Matrix<T, TensorContainer>& setData(const LinearContainer<T>& matrixItems);

    std::string toString() const;

    template<typename U> friend std::ostream& operator<<(std::ostream& os, const Matrix<U>& matrix);

    void fillWith(const T& fill);

    void transposition();

    void resize(const uint64_t dim1, const uint64_t dim2);

    Matrix<T, TensorContainer>& operator=(const Matrix<T, TensorContainer>& otherMatrix);

    Matrix<T, TensorContainer>& operator=(Matrix<T, TensorContainer>&& otherMatrix) noexcept;

    bool operator==(const Matrix<T, TensorContainer>& otherMatrix) const;

    bool operator!=(const Matrix<T, TensorContainer>& otherMatrix) const;

    std::partial_ordering operator<=>(const Matrix<T, TensorContainer>& otherMatrix) const;


    // #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     inline auto operator OP_SYMBOL(const Tensor<T>& tensor2) const\
    //     requires requires (T a, T b) {a OP_SYMBOL b;};

    // #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     template<typename U> friend inline auto operator OP_SYMBOL(const Tensor<U>& tensor, const U& value)\
    //     requires requires (U a, U b) {a OP_SYMBOL b;};

    // #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    //     template<typename U> friend inline auto operator OP_SYMBOL(const U& value, const Tensor<U>& tensor)\
    //     requires requires (U a, U b) {a OP_SYMBOL b;};

    // #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     void operator OP_SYMBOL##=(const Tensor<T>& tensor2)\
    //     requires requires (T a, T b) {a OP_SYMBOL##= b;};

    // #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
    //     void operator OP_SYMBOL##=(const T& value)\
    //     requires requires (T a, T b) {a OP_SYMBOL##= b;};

    // #define ARITHMETIC_BINARY(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    // ARITHMETIC_BINARY(+)
    // ARITHMETIC_BINARY(-)
    // ARITHMETIC_BINARY(*)
    // ARITHMETIC_BINARY(/)
    // ARITHMETIC_BINARY(|)
    // ARITHMETIC_BINARY(&)
    // ARITHMETIC_BINARY(^)
    // ARITHMETIC_BINARY(%)

    // #define LOGICAL_BINARY(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)

    // LOGICAL_BINARY(&&)
    // LOGICAL_BINARY(||)

    // #define BITSHIFTLIKE(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    // BITSHIFTLIKE(<<)
    // BITSHIFTLIKE(>>)

    // #undef ARITHMETIC_BINARY_ToTrT
    // #undef ARITHMETIC_BINARY_ToVrT
    // #undef ARITHMETIC_BINARY_VoTrT
    // #undef ARITHMETIC_BINARY_ToeT
    // #undef ARITHMETIC_BINARY_ToeV

    // #undef BITSHIFTLIKE
    // #undef LOGICAL_BINARY

    // #undef ARITHMETIC_BINARY

    Matrix<T, TensorContainer> inverse() const;

    void inverseInPlace();

    void matrixMultiplication(const Matrix<T, TensorContainer>& otherMatrix);

    

    template <apply_and_return_callable_parallel<T> C>
    auto applyAndReturn(const Matrix<T, TensorContainer>& tensor2, C&& operation) const;
    
    template <typename A, typename B, apply_and_return_callable_parallel<T> C> 
    static auto applyAndReturn(const A& operand1, const B& operand2, C&& operation)
    requires(a_or_b_or_both_a<A, B, T, Matrix<T, TensorContainer>>);

    template <apply_callable_parallel<T> C>
    void apply(const Matrix<T, TensorContainer>& tensor2, C&& operation);

    template <apply_callable_parallel<T> C>
    static void apply(Matrix<T, TensorContainer>& operand1, const Matrix<T, TensorContainer>& operand2, C&& operation);

    template <apply_callable_parallel<T> C>
    static void apply(Matrix<T, TensorContainer>& operand1, const T& operand2, C&& operation);

    template <apply_reverse_callable_parallel<T> C>
    static void apply(const T& operand1, Matrix<T, TensorContainer>& operand2, C&& operation);

    template <foreach_and_return_callable_parallel<T> C>
    auto forEachAndReturn(C&& operation) const;

    template <foreach_and_return_callable_parallel<T> C>
    static auto forEachAndReturn(const Matrix<T, TensorContainer>& matrix, C&& operation);

    template <foreach_callable_parallel<T> C> 
    void forEach(C&& operation);

    template <foreach_callable_parallel<T> C>
    static void forEach(Matrix<T, TensorContainer>& matrix, C&& operation);


    
    virtual ~Matrix();



    protected:

    MetadataContainer getCoords(uint64_t itemIndex) const;

    uint64_t getIndex(span_view<uint64_t> coordinates) const;

};

}

#include "Matrix.tpp"

#endif