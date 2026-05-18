#include <vector>

#include <sycl/sycl.hpp>

#include "Utils.hpp"
#include "TensorParallel.hpp"
#include "MemoryBackendUSM.hpp"

namespace gema{
   
    
    template <class T>
    TensorParallel<T>::TensorParallel(const LinearContainer<uint64_t>& newDimensionSizes)
    : tensor_(newDimensionSizes.copyToBackend(MetadataBackend(queue_)), DataBackend(queue_)){
        // this->tensor_ = Tensor<T>(MemoryBackendUSM<T, usmDataKind_>(queue_));
        // this->dimensionSizes_ = LinearContainer<uint64_t>(newTensorDimensionSizes, MemoryBackendUSM<uint64_t, usmDataKind_>(queue_));
        // this->update();
    }

    // template <class T>
    // TensorParallel<T>::TensorParallel(const MetadataContainer& newDimensionSizes, const DataContainer& newData)
    // : tensor_(newDimensionSizes, newData){

    // }


    // template <class T>
    // TensorParallel<T>::TensorParallel(span_view<uint64_t> newDimensionSizes)
    // : tensor_(newDimensionSizes, DataBackend(queue_), MetadataBackend(queue_)){

    // }

    template <class T>
    TensorParallel<T>::TensorParallel(const LinearContainer<uint64_t>& newDimensionSizes, const LinearContainer<T>& newData)
    : tensor_(
        newDimensionSizes.copyToBackend(MetadataBackend(queue_)),
        newData.copyToBackend(DataBackend(queue_))
    ){

    }

    template <class T>
    TensorParallel<T>::TensorParallel(const TensorParallel<T>& otherTensor)
    : tensor_(otherTensor.tensor_){
        //queue_ = otherTensor.queue_;
    }

    template <class T>
    TensorParallel<T>::TensorParallel(TensorParallel<T>&& otherTensor) noexcept 
    : tensor_(std::move(otherTensor.tensor_)){
        //queue_ = std::move(otherTensor.queue_);
    }

    template <class T>
    template <typename OtherTensor>
    TensorParallel<T>::TensorParallel(OtherTensor* otherTensor)
    : tensor_(&(otherTensor->getTensor())){
        //*queue_ = otherTensor->getQueue();
    }

    template <class T>
    TensorParallel<T>::TensorParallel()
    : tensor_(DataBackend(queue_), MetadataBackend(queue_)){
        
    }

    template <class T>
    TensorParallel<T>::~TensorParallel(){
        
    }

    template <class T>
    TensorParallel<T>& TensorParallel<T>::operator=(const TensorParallel<T>& otherTensor){
        tensor_ = otherTensor.tensor_;
        queue_ = otherTensor.queue_;
        return *this;
    }

    template <class T>
    TensorParallel<T>& TensorParallel<T>::operator=(TensorParallel<T>&& otherTensor) noexcept {
        tensor_ = std::move(otherTensor.tensor_);
        queue_ = std::move(otherTensor.queue_);
        return *this;
    }

    template <class T>
    const TensorParallel<T>::MetadataContainer& TensorParallel<T>::getDimensionSizes() const {
        return tensor_.getDimensionSizes();
    }

    template <class T>
    uint64_t TensorParallel<T>::getNumberOfDimensions() const {
        return tensor_.getNumberOfDimensions();
    }

    template <class T>
    uint64_t TensorParallel<T>::getNumberOfItems() const {
        return tensor_.getNumberOfItems();
    }

    template <class T>
    const Tensor<T, typename TensorParallel<T>::DataBackend, typename TensorParallel<T>::MetadataBackend>& 
    TensorParallel<T>::getTensor() const {
        return tensor_;
    }

    template <class T>
    sycl::queue* TensorParallel<T>::getQueue(){
        return queue_;
    }

    template <class T>
    T TensorParallel<T>::getItem(span_view<uint64_t> coordinates){
        //return tensor_.getItem(coordinates);
        const uint64_t index = tensor_.getIndex(coordinates);
        return tensor_.getDataContainer().get(index);
    }

    template <class T>
    void TensorParallel<T>::setItem(const T& value, span_view<uint64_t> coordinates){

        const uint64_t index = tensor_.getIndex(coordinates);
        tensor_.getDataContainer().set(index, value);

        // T* placeToSave = tensor_.getData() + tensor_.getIndex(coordinates);

        // queue_->submit([&](sycl::handler& h){
        //     h.single_task([=](){
        //         *placeToSave = value;
        //     });
        // }).wait();
    }

    template <class T>
    T* TensorParallel<T>::getData(){
        return tensor_.getData();
    }

    template <class T>
    const T* TensorParallel<T>::getData() const{
        return tensor_.getData();
    }

    template <class T>
    TensorParallel<T>& TensorParallel<T>::setData(const LinearContainer<T>& tensorItems){
        tensor_.setData(tensorItems.copyToBackend(DataBackend(queue_)));
        return *this;
    }

    // template <class T>
    // TensorParallel<T> &TensorParallel<T>::setData(const DataContainer& tensorItems){
    //     tensor_.setData(tensorItems);
    //     return *this;
    // }

    template <class T>
    bool TensorParallel<T>::isValidCoordinates(span_view<uint64_t> coords) const {
        return tensor_.isValidCoordinates(coords);
    }

    template <class T>
    bool TensorParallel<T>::isValidCoordinates(
        span_view<uint64_t> coords, 
        span_view<uint64_t> dimensionSizes
    ){
        return Tensor<T>::isValidCoordinates(coords, dimensionSizes);
    }

    template <class T>
    bool TensorParallel<T>::isEquilateral() const {
        return tensor_.isEquilateral();
    }

    template <class T>
    void TensorParallel<T>::fillWith(const T& fill){
        //tensor_.fillWith(fill);

        forEach([=](T& item){
            item = fill;
        });
    }

    template <typename U>
    std::ostream& operator<<(std::ostream &os, const TensorParallel<U>& tensor){
        return os << tensor.toString();
    }

    template <class T>
    std::string TensorParallel<T>::toString() const {
        //return tensor_.toString();

        const uint64_t itemCount = tensor_.getNumberOfItems();
        LinearContainer<uint64_t, MetadataBackend> dimensionSizes = tensor_.getDimensionSizes();

        std::vector<std::string> openingBrackets(itemCount);
        std::fill(openingBrackets.begin(), openingBrackets.end(), "");

        std::vector<std::string> closingBrackets(itemCount);
        std::fill(closingBrackets.begin(), closingBrackets.end(), "");

        uint64_t dimensionProduct = itemCount;

        for(uint64_t i = 0; i < dimensionSizes.size(); i++){ // Identity endianness

            for(uint64_t j = 0; j < itemCount; ++j){

                if(j % dimensionProduct == 0){
                    openingBrackets[j] += "{";
                }

                if(j % dimensionProduct == (dimensionProduct - 1)){
                    closingBrackets[j] += "}";
                }
            }

            dimensionProduct /= dimensionSizes[i];
        }

        std::string output = "";
        LinearContainer<T> dataContainer = tensor_.getDataContainer().copyToBackend(MemoryBackend<T>());

        for(uint64_t i = 0; i < itemCount; ++i){
            output += 
                std::format("{}{}{}{}", openingBrackets[i], dataContainer[i], closingBrackets[i], (((i + 1) >= itemCount) ? "" : ", "));
        }

        return output; 
    }

    template <class T>
    bool TensorParallel<T>::operator==(const TensorParallel<T>& otherTensor) const {
        return tensor_ == otherTensor.tensor_;
    }

    template <class T>
    bool TensorParallel<T>::operator!=(const TensorParallel<T>& otherTensor) const {
        return tensor_ != otherTensor.tensor_;
    }

    template <class T>
    std::partial_ordering TensorParallel<T>::operator<=>(const TensorParallel<T>& otherTensor) const{
        return tensor_ <=> otherTensor.tensor_;
    }

    template <class T>
    TensorParallel<T> TensorParallel<T>::transpositionAndReturn(const uint64_t dim1, const uint64_t dim2) const {

        TensorParallel<T> newTensor(*this);
        newTensor.transposition();
        return newTensor;

        //return tensor_.transpositionAndReturn(dim1, dim2);//never use

        // if(dim1 == dim2) return TensorParallel<T>(this);

        // // Copying the dimensionSizes
        // // Change assigment to just construction of correct size
        // std::vector<uint64_t> transposedDimensionSizes = this->dimensionSizes_; 

        // // Swapping the dimension sizes
        // transposedDimensionSizes[dim1] = this->dimensionSizes_[dim2]; 
        // transposedDimensionSizes[dim2] = this->dimensionSizes_[dim1];
        
        // // Initializing the new tensor
        // TensorParallel<T> tensorTransposed = TensorParallel<T>(transposedDimensionSizes);

        // // std::vector<uint64_t> original, switched;
        // // original.resize(dimensionSizes_.size());
        // // switched.resize(dimensionSizes_.size());

        // const uint64_t dimensionCount = this->dimensionSizes_.size();

        // const T* rawOldDimensionSizes = sycl::malloc(dimensionCount, *queue_, usmDataKind_);
        // queue_->memcpy(rawOldDimensionSizes, this->dimensionSizes_.data(), sizeof(uint64_t) * dimensionCount);

        // const T* rawData = this->tensor_.getData();
        // T* rawTransposedData = tensorTransposed.getData();

        // T* originalCoords = sycl::malloc(dimensionCount, *queue_, usmDataKind_);
        // T* switchedCoords = sycl::malloc(dimensionCount, *queue_, usmDataKind_);

        // queue_->parallel_for(this->tensor_.size(), [=](sycl::id<1> idx){

        //     size_t i = idx[0];

        //     // The swap of two desired coordinates
        //     for(uint64_t j = 0; j < dimensionCount; j++){
        //         switchedCoords[j] = originalCoords[j];
        //     }
            
        //     switchedCoords[dim1] = originalCoords[dim2];
        //     switchedCoords[dim2] = originalCoords[dim1];

        //     rawTransposedData[tensorTransposed.getIndex(switchedCoords)] = std::move(rawData[i]);

        //     Tensor<T>::incrementCoords(switchedCoords, rawOldDimensionSizes);

        // }).wait();

        // return tensorTransposed;
    }

    template <class T>
    void TensorParallel<T>::transposition(const uint64_t dim1, const uint64_t dim2){

        //tensor_.transposition(dim1, dim2);

        if(dim1 == dim2) return;

        // Copying the dimensionSizes
        // Change assigment to just construction of correct size
        const MetadataContainer oldDimensionSizes = tensor_.getDimensionSizes();
        span_view<uint64_t> oldDimensionSizesView{oldDimensionSizes};

        // Swapping the dimension sizes
        const uint64_t temporaryDimensionSize1 = tensor_.getDimensionSizes()[dim1];
        tensor_.getDimensionSizes()[dim1] = tensor_.getDimensionSizes()[dim2];
        tensor_.getDimensionSizes()[dim2] = temporaryDimensionSize1;

        span_view<uint64_t> newDimensionSizesView{tensor_.getDimensionSizes()};

        const uint64_t itemCount = tensor_.updateInnerState();
        const uint64_t dimensionCount = tensor_.getNumberOfDimensions();
        
        // Initializing the new data
        DataContainer newData(itemCount, DataBackend(queue_));

        T* oldDataRaw = tensor_.getData();
        T* newDataRaw = newData.data();

        //std::cout << "here transposition" << std::endl;

        uint64_t* coordsBuffer = sycl::malloc_device<uint64_t>(itemCount * dimensionCount, *queue_);

        queue_->parallel_for(itemCount, [=](sycl::id<1> idx){

            size_t i = idx[0];

            uint64_t* coords = coordsBuffer + i * dimensionCount;
            Tensor<T>::getCoords(i, oldDimensionSizesView, coords);

            uint64_t temporaryCoord1 = coords[dim1];
            coords[dim1] = coords[dim2];
            coords[dim2] = temporaryCoord1;

            span_view<uint64_t> coordsView{coords, dimensionCount};

            newDataRaw[Tensor<T>::getIndex(coordsView, newDimensionSizesView)] = std::move(oldDataRaw[i]);

        }).wait();

        sycl::free(coordsBuffer, *queue_);

        tensor_.getDataContainer() = std::move(newData);
    }

    template <class T>
    void TensorParallel<T>::resize(const LinearContainer<uint64_t>& newDimensionSizes){

        const MetadataContainer oldDimensionSizes = tensor_.getDimensionSizes();
        span_view<uint64_t> oldDimensionSizesView(oldDimensionSizes);
        const uint64_t dimensionCount = oldDimensionSizes.size();
        const uint64_t oldItemCount = tensor_.getNumberOfItems();

        tensor_.getDimensionSizes() = newDimensionSizes.copyToBackend(MetadataBackend(queue_));

        const uint64_t newItemCount = tensor_.updateInnerState();
        span_view<uint64_t> newDimensionSizesView(tensor_.getDimensionSizes());

        DataContainer newData(newItemCount, tensor_.getDataContainer().getMemoryBackend());

        T* oldDataRaw = tensor_.getData();
        T* newDataRaw = newData.data();

        uint64_t* coordsBuffer = sycl::malloc_device<uint64_t>(oldItemCount * dimensionCount, *queue_);

        queue_->parallel_for(oldItemCount, [=](sycl::id<1> idx){

            size_t i = idx[0];

            uint64_t* coords = coordsBuffer + i * dimensionCount;
            Tensor<T, DataBackend, MetadataBackend>::getCoords(i, oldDimensionSizesView, coords);
            span_view<uint64_t> coordsView = {coords, dimensionCount};
            
            if(Tensor<T, DataBackend, MetadataBackend>::isValidCoordinates(coordsView, newDimensionSizesView)){
                uint64_t destinationIndex = 
                    Tensor<T, DataBackend, MetadataBackend>::getIndex(coordsView, newDimensionSizesView);
                newDataRaw[destinationIndex] = std::move(oldDataRaw[i]);
            }
        }).wait();
        
        sycl::free(coordsBuffer, *queue_);

        tensor_.getDataContainer() = std::move(newData);
    }

    template <class T>
    void TensorParallel<T>::resize(const uint64_t newDimensionSize, const uint64_t dimensionIndex){
        // too many copies, just temporary solution
        LinearContainer<uint64_t> newDimensionSizes = tensor_.dimensionSizes_.copyToBackend(MemoryBackend<T>());
        newDimensionSizes[dimensionIndex] = newDimensionSize;
        resize(newDimensionSizes);
    }

    template <class T>
    void TensorParallel<T>::addDimension(const uint64_t newDimensionSize, const uint64_t putBefore){

    }

    template <class T>
    void TensorParallel<T>::removeDimension(const uint64_t removedDimensionIndex){

    }

    template <class T>
    template <apply_and_return_callable_parallel<T> C>
    auto TensorParallel<T>::applyAndReturn(const TensorParallel<T>& tensor2, C&& operation) const {
        return TensorParallel<T>::applyAndReturn(*this, tensor2, std::forward<C>(operation));
    }

    template <class T>
    template <typename A, typename B, apply_and_return_callable_parallel<T> C>
    /*static*/ auto TensorParallel<T>::applyAndReturn(const A& operand1, const B& operand2, C&& operation)
    requires(tensor_or_t_or_bothtensor_parallel<A, B, T>){

        using opReturnType = decltype(operation(std::declval<T>(), std::declval<T>()));
        const TensorParallel<T>* tensorOperand = type_pick<TensorParallel<T>>(operand1, operand2);
        sycl::queue* queue = tensorOperand->queue_;

        TensorParallel<opReturnType> resultTensor = TensorParallel<opReturnType>(tensorOperand);
        opReturnType* resultRawData = resultTensor.getData();

        const T* operand1Raw;
        const T* operand2Raw;

        if constexpr (std::is_same_v<A, B>){
            operand1Raw = operand1.getData();
            operand2Raw = operand2.getData();
        }else if constexpr (std::is_same_v<A, T>){
            operand2Raw = operand2.getData();
        }else if constexpr (std::is_same_v<B, T>){
            operand1Raw = operand1.getData();
        }

        queue->parallel_for(tensorOperand->getNumberOfItems(), [=](sycl::id<1> idx){

            size_t i = idx[0];
            if constexpr (std::is_same_v<A, B>){
                resultRawData[i] = operation(operand1Raw[i], operand2Raw[i]);
            }else if constexpr (std::is_same_v<A, T>){
                resultRawData[i] = operation(operand1, operand2Raw[i]);
            }else if constexpr (std::is_same_v<B, T>){
                resultRawData[i] = operation(operand1Raw[i], operand2);
            }
        }).wait();

        return resultTensor;
    }

    template <class T>
    template <apply_callable_parallel<T> C>
    void TensorParallel<T>::apply(const TensorParallel<T>& tensor2, C&& operation){
        TensorParallel<T>::apply(*this, tensor2, std::forward<C>(operation));
    }

    template <class T>
    template <apply_callable_parallel<T> C>
    /*static*/ void TensorParallel<T>::apply(TensorParallel<T>& operand1, const TensorParallel<T>& operand2, C&& operation){

        sycl::queue* queue = operand1.queue_;
        T* operand1Raw = operand1.getData();
        const T* operand2Raw = operand2.getData();

        queue->parallel_for(operand1.getNumberOfItems(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            operation(operand1Raw[i], operand2Raw[i]);
        }).wait();
    }

    template <class T>
    template <apply_callable_parallel<T> C>
    /*static*/ void TensorParallel<T>::apply(TensorParallel<T>& operand1, const T& operand2, C&& operation){

        sycl::queue* queue = operand1.queue_;
        T* operand1Raw = operand1.getData();

        queue->parallel_for(operand1.getNumberOfItems(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            operation(operand1Raw[i], operand2);
        }).wait();
    }

    template <class T>
    template <apply_reverse_callable_parallel<T> C>
    /*static*/ void TensorParallel<T>::apply(const T& operand1, TensorParallel<T>& operand2, C&& operation){

        sycl::queue* queue = operand2.queue_;
        T* operand2Raw = operand2.getData();

        queue->parallel_for(operand2.getNumberOfItems(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            operation(operand1, operand2Raw[i]);
        }).wait();
    }

    template <class T>
    template <foreach_and_return_callable_parallel<T> C>
    auto TensorParallel<T>::forEachAndReturn(C&& operation) const{
        return TensorParallel<T>::forEachAndReturn(*this, std::forward<C>(operation));
    }

    template <class T>
    template <foreach_and_return_callable_parallel<T> C>
    /*static*/ auto TensorParallel<T>::forEachAndReturn(const TensorParallel<T>& tensor, C&& operation){

        const T* operandRaw = tensor.getData();
        sycl::queue* queue = tensor.queue_;

        using opReturnType = decltype(operation(std::declval<T>()));
        TensorParallel<opReturnType> resultTensor = TensorParallel<opReturnType>(&tensor);
        opReturnType* resultRawData = resultTensor.getData();

        queue->parallel_for(tensor.getNumberOfItems(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            resultRawData[i] = operation(operandRaw[i]);
        }).wait();

        return resultTensor;
    }

    template <class T>
    template <foreach_callable_parallel<T> C>
    void TensorParallel<T>::forEach(C&& operation){
        TensorParallel<T>::forEach(*this, std::forward<C>(operation));
    }

    template <class T>
    template <foreach_callable_parallel<T> C>
    /*static*/ void TensorParallel<T>::forEach(TensorParallel<T>& tensor, C&& operation){

        sycl::queue* queue = tensor.queue_;
        T* operandRaw = tensor.getData();

        queue->parallel_for(tensor.getNumberOfItems(), [=](sycl::id<1> idx){
            size_t i = idx[0];
            operation(operandRaw[i]);
        }).wait();
    }

    template <class T>
    template <apply_to_item_callable<T> C>
    void TensorParallel<T>::applyToItem(span_view<uint64_t> coords, C&& operation){

        T* operationItem = tensor_.getData() + tensor_.getIndex(coords);

        queue_->submit([&](sycl::handler& h){
            h.single_task([=](){
                operation(*operationItem);
            });
        }).wait();
    }

    // template <class T>
    // uint64_t TensorParallel<T>::getIndex(span_view<uint64_t> coordinates) const{
    //     return TensorParallel<T>::getIndex(coordinates, tensor_.getDimensionSizes());
    // }

    // template <class T>
    // uint64_t TensorParallel<T>::getIndex(span_view<uint64_t> coordinates, span_view<uint64_t> dimensionSizes){

    //     uint64_t* hostCoordinates = sycl::malloc_host<uint64_t>(coordinates.size());
    //     queue_->memcpy(host, coordinates.data(), coordinates.size() * sizeof(uint64_t));

    //     uint64_t itemIndex = 0;
    //     const uint64_t dimensionCount = dimensionSizes.size();
    //     for (size_t i = 0; i < dimensionCount; ++i){
    //         itemIndex = itemIndex * dimensionSizes[i] + hostCoordinates[i];
    //     }

    //     return itemIndex;
    // }
}