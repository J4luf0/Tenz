//#include "Matrix.hpp"

namespace gema {

    
    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer>::Matrix(const uint64_t x, const uint64_t y)
    : tensor_({x, y}){

    }

    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer>::Matrix(const Matrix<T, TensorContainer>& otherMatrix)
    : tensor_(otherMatrix.tensor_){

    }

    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer>::Matrix(Matrix<T, TensorContainer>&& otherMatrix) noexcept
    : tensor_(std::move(otherMatrix.tensor_)){

    }
    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer>::Matrix(){

    }

    template <class T, TensorConcept TensorContainer>
    const Matrix<T, TensorContainer>::MetadataContainer& Matrix<T, TensorContainer>::getDimensionSizes() const{
        return tensor_.getDimensionSizes();
    }

    template <class T, TensorConcept TensorContainer>
    uint64_t Matrix<T, TensorContainer>::getNumberOfDimensions() const{
        return tensor_.getNumberOfDimensions();
    }

    template <class T, TensorConcept TensorContainer>
    uint64_t Matrix<T, TensorContainer>::getNumberOfItems() const{
        return tensor_.getNumberOfItems();
    }

    template <class T, TensorConcept TensorContainer>
    T Matrix<T, TensorContainer>::getItem(const uint64_t x, const uint64_t y) const{
        uint64_t index = getIndex({x, y});
        return tensor_.getDataContainer().get(index);
    }

    template <class T, TensorConcept TensorContainer>
    void Matrix<T, TensorContainer>::setItem(const T& value, const uint64_t x, const uint64_t y){
        uint64_t index = getIndex({x, y});
        tensor_.getDataContainer().set(index, value);
    }

    template <class T, TensorConcept TensorContainer>
    T* Matrix<T, TensorContainer>::getData(){
        return tensor_.getData();
    }
    
    template <class T, TensorConcept TensorContainer>
    const T* Matrix<T, TensorContainer>::getData() const {
        return tensor_.getData();
    }


    // template <class T, TensorConcept TensorContainer>
    // Matrix<T, TensorContainer>& Matrix<T, TensorContainer>::setData(const DataContainer& matrixItems){
    //     tensor_.setData(matrixItems);
    //     return *this;
    // }

    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer> &Matrix<T, TensorContainer>::setData(const LinearContainer<T> &matrixItems){
        tensor_.setData(matrixItems);
        return *this;
    }

    template <class T, TensorConcept TensorContainer>
    std::string Matrix<T, TensorContainer>::toString() const{
        return tensor_.toString();
    }

    template <typename U>
    std::ostream& operator<<(std::ostream& os, const Matrix<U>& matrix){
        return os << matrix.toString();
    }

    template <class T, TensorConcept TensorContainer>
    void Matrix<T, TensorContainer>::fillWith(const T& fill){
        tensor_.fillWith(fill);
    }

    template <class T, TensorConcept TensorContainer>
    void Matrix<T, TensorContainer>::transposition(){
        tensor_.transposition(0, 1);
    }

    template <class T, TensorConcept TensorContainer>
    void Matrix<T, TensorContainer>::resize(const uint64_t dim1, const uint64_t dim2){
        tensor_.resize({dim1, dim2});
    }

    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer>& Matrix<T, TensorContainer>::operator=(const Matrix<T, TensorContainer>& otherMatrix){
        tensor_ = otherMatrix.tensor_;
        return *this;
    }

    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer>& Matrix<T, TensorContainer>::operator=(Matrix<T, TensorContainer>&& otherMatrix) noexcept{
        tensor_ = std::move(otherMatrix.tensor_);
        return *this;
    }

    template <class T, TensorConcept TensorContainer>
    bool Matrix<T, TensorContainer>::operator==(const Matrix<T, TensorContainer>& otherMatrix) const{
        return tensor_ == otherMatrix.tensor_;
    }
    
    template <class T, TensorConcept TensorContainer>
    bool Matrix<T, TensorContainer>::operator!=(const Matrix<T, TensorContainer>& otherMatrix) const{
        return !(*this == otherMatrix);
    }

    template <class T, TensorConcept TensorContainer>
    std::partial_ordering Matrix<T, TensorContainer>::operator<=>(const Matrix<T, TensorContainer>& otherMatrix) const{
        return tensor_ <=> otherMatrix.tensor_;
    }

    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer> Matrix<T, TensorContainer>::inverse() const{

        const uint64_t n = getDimensionSizes()[0];

        if(n != getDimensionSizes()[1]){
            throw std::runtime_error("Inverse requires square matrix.");
        }

        DefaultEquals<T> defaultEquals;

        // Left side = original matrix
        Matrix<T, TensorContainer> left(*this);

        // Right side = identity matrix
        Matrix<T, TensorContainer> right(n, n);
        right.fillWith(static_cast<T>(0));

        for(uint64_t i = 0; i < n; ++i){
            right.setItem(static_cast<T>(1), i, i);
        }

        // Gauss-Jordan elimination
        for(uint64_t pivot = 0; pivot < n; ++pivot){

            T pivotValue = left.getItem(pivot, pivot);

            if(defaultEquals(pivotValue, static_cast<T>(0))){
                throw std::runtime_error("Matrix is singular.");
            }

            std::vector<T> leftPivotRow(n);
            std::vector<T> rightPivotRow(n);

            for(uint64_t col = 0; col < n; ++col){
                leftPivotRow[col] = left.getItem(pivot, col) / pivotValue;
                rightPivotRow[col] = right.getItem(pivot, col) / pivotValue;
            }

            // Normalize pivot row
            for(uint64_t col = 0; col < n; ++col){
                left.setItem(leftPivotRow[col], pivot, col);
                right.setItem(rightPivotRow[col], pivot, col);
            }

            // Eliminate other rows
            for(uint64_t row = 0; row < n; ++row){

                if(row == pivot) continue;

                T factor = left.getItem(row, pivot);

                std::vector<T> leftCurrentRow(n);
                std::vector<T> rightCurrentRow(n);

                for(uint64_t col = 0; col < n; ++col){
                    leftCurrentRow[col] = left.getItem(row, col);
                    rightCurrentRow[col] = right.getItem(row, col);
                }

                for(uint64_t col = 0; col < n; ++col){
                    left.setItem(leftCurrentRow[col] - (factor * leftPivotRow[col]), row, col);
                    right.setItem(rightCurrentRow[col] - (factor * rightPivotRow[col]), row, col);
                }
            }
        }

        return right;
    }

    template <class T, TensorConcept TensorContainer>
    void Matrix<T, TensorContainer>::inverseInPlace(){
        *this = inverse();
    }

    template <class T, TensorConcept TensorContainer>
    void Matrix<T, TensorContainer>::matrixMultiplication(const Matrix<T, TensorContainer>& otherMatrix)
    requires (std::is_same_v<TensorContainer, Tensor<T>>){

        const uint64_t m = getDimensionSizes()[0];
        const uint64_t n = getDimensionSizes()[1];

        const uint64_t otherRows = otherMatrix.getDimensionSizes()[0];
        const uint64_t p = otherMatrix.getDimensionSizes()[1];

        if(n != otherRows){
            throw std::runtime_error("Matrix multiplication dimension mismatch.");
        }

        Matrix<T, TensorContainer> result(m, p);

        for(uint64_t i = 0; i < m; ++i){
            for(uint64_t j = 0; j < p; ++j){

                T sum{};

                for(uint64_t k = 0; k < n; ++k){
                    sum += getItem(i, k) * otherMatrix.getItem(k, j);
                }

                result.setItem(sum, i, j);
            }
        }

        *this = std::move(result);
    }

    template <class T, TensorConcept TensorContainer>
    template <apply_and_return_callable_parallel<T> C>
    auto Matrix<T, TensorContainer>::applyAndReturn(const Matrix<T, TensorContainer>& tensor2, C&& operation) const{
        return Matrix<T, TensorContainer>::applyAndReturn(*this, tensor2, operation);
    }

    template <class T, TensorConcept TensorContainer>
    template <typename A, typename B, apply_and_return_callable_parallel<T> C>
    auto Matrix<T, TensorContainer>::applyAndReturn(const A &operand1, const B &operand2, C &&operation)
        requires(a_or_b_or_both_a<A, B, T, Matrix<T, TensorContainer>>)
    {

        using opReturnType = decltype(operation(std::declval<T>(), std::declval<T>()));
        Matrix<opReturnType, TensorContainer> newMatrix{};

        if constexpr(std::is_same_v<A, B>){
            newMatrix.tensor_ = TensorContainer::applyAndReturn(operand1.tensor_, operand2.tensor_, operation);
        }else if constexpr(std::is_same_v<A, T>){
            newMatrix.tensor_ = TensorContainer::applyAndReturn(operand1, operand2.tensor_, operation);
        }else if constexpr(std::is_same_v<B, T>){
            newMatrix.tensor_ = TensorContainer::applyAndReturn(operand1.tensor_, operand2, operation);
        }
        
        return newMatrix;
    }

    template <class T, TensorConcept TensorContainer>
    template <apply_callable_parallel<T> C>
    void Matrix<T, TensorContainer>::apply(const Matrix<T, TensorContainer>& tensor2, C&& operation){
        Matrix<T, TensorContainer>::apply(*this, tensor2, operation);
    }

    template <class T, TensorConcept TensorContainer>
    template <apply_callable_parallel<T> C>
    void Matrix<T, TensorContainer>::apply(Matrix<T, TensorContainer>& operand1, const Matrix<T, TensorContainer>& operand2, C&& operation){
        TensorContainer::apply(operand1.tensor_, operand2.tensor_, operation);
    }

    template <class T, TensorConcept TensorContainer>
    template <apply_callable_parallel<T> C>
    void Matrix<T, TensorContainer>::apply(Matrix<T, TensorContainer>& operand1, const T& operand2, C&& operation){
        TensorContainer::apply(operand1.tensor_, operand2, operation);
    }

    template <class T, TensorConcept TensorContainer>
    template <apply_reverse_callable_parallel<T> C>
    void Matrix<T, TensorContainer>::apply(const T& operand1, Matrix<T, TensorContainer>& operand2, C&& operation){
        TensorContainer::apply(operand1, operand2.tensor_, operation);
    }

    template <class T, TensorConcept TensorContainer>
    template <foreach_and_return_callable_parallel<T> C>
    auto Matrix<T, TensorContainer>::forEachAndReturn(C&& operation) const{
        return Matrix<T, TensorContainer>::forEachAndReturn(*this, operation);
    }

    template <class T, TensorConcept TensorContainer>
    template <foreach_and_return_callable_parallel<T> C>
    auto Matrix<T, TensorContainer>::forEachAndReturn(const Matrix<T, TensorContainer>& matrix, C&& operation){
        
        using opReturnType = decltype(operation(std::declval<T>()));
        Matrix<opReturnType, TensorContainer> newMatrix{};

        newMatrix.tensor_ = TensorContainer::forEachAndReturn(matrix.tensor_, operation);

        return newMatrix;
    }

    template <class T, TensorConcept TensorContainer>
    template <foreach_callable_parallel<T> C>
    void Matrix<T, TensorContainer>::forEach(C&& operation){
        Matrix<T, TensorContainer>::forEach(*this, operation);
    }

    template <class T, TensorConcept TensorContainer>
    template <foreach_callable_parallel<T> C>
    void Matrix<T, TensorContainer>::forEach(Matrix<T, TensorContainer>& matrix, C&& operation){
        TensorContainer::forEach(matrix.tensor_, operation);
    }

    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer>::~Matrix(){
        
    }

    template <class T, TensorConcept TensorContainer>
    Matrix<T, TensorContainer>::MetadataContainer Matrix<T, TensorContainer>::getCoords(uint64_t itemIndex) const{

        const MetadataContainer& dimensionSizes = tensor_.getDimensionSizes();
        
        MetadataContainer coords(2, dimensionSizes.getMemoryBackend());

        coords[0] = itemIndex / dimensionSizes[1];
        coords[1] = itemIndex % dimensionSizes[1];

        return coords;
    }

    template <class T, TensorConcept TensorContainer>
    uint64_t Matrix<T, TensorContainer>::getIndex(span_view<uint64_t> coordinates) const{
        const MetadataContainer& dimensionSizes = tensor_.getDimensionSizes();
        return (coordinates[0] * dimensionSizes[1]) + coordinates[1];
    }
}
