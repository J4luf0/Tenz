#ifndef TENSOR_PARALLEL_HPP
#define TENSOR_PARALLEL_HPP

#include <sycl/sycl.hpp>

#include "MemoryBackendConcept.hpp"
#include "MemoryBackendUSM.hpp"
#include "TensorConcept.hpp"
#include "Tensor.hpp"

namespace gema{

template<class T>//, MemoryBackendConcept<T> IMemoryBackend = MemoryBackendUSM<T, sycl::usm::alloc::device>
class TensorParallel;

// Concept that checks if type X is of type T or TensorParallel<T>. Useful for operator overloads.
template <typename X, class T>
concept is_tensorparallel_or_t = std::is_same_v<X, T> || std::is_same_v<X, TensorParallel<T>>;

template<typename C, class T>
concept apply_to_item_callable = std::is_invocable_r_v<void, C, T&> && sycl::is_device_copyable_v<std::remove_cvref_t<C>>;

template <typename A, typename B, class T>
concept tensor_or_t_or_bothtensor_parallel = 
    (std::is_same_v<std::remove_cvref_t<A>, TensorParallel<T>> && std::is_same_v<std::remove_cvref_t<B>, TensorParallel<T>>) ||
    (std::is_same_v<std::remove_cvref_t<A>, TensorParallel<T>> && std::is_same_v<std::remove_cvref_t<B>, T>) ||
    (std::is_same_v<std::remove_cvref_t<A>, T> && std::is_same_v<std::remove_cvref_t<B>, TensorParallel<T>>);


template<class T>
class TensorParallel : /*public Tensor<T>,*/public AbstractOperation<TensorParallel<T>>{

    static_assert(sycl::is_device_copyable_v<T>);
    static_assert(std::is_trivially_copyable_v<T>);

    public:

    inline static sycl::queue queueGlobal_{sycl::property::queue::in_order{}};

    constexpr static sycl::usm::alloc usmDataKind_ = sycl::usm::alloc::device;
    constexpr static sycl::usm::alloc usmMetadataKind_ = sycl::usm::alloc::shared;

    using DataBackend = MemoryBackendUSM<T, usmDataKind_>;
    using MetadataBackend = MemoryBackendUSM<uint64_t, usmMetadataKind_>;

    using DataContainer = LinearContainer<T, DataBackend>;
    using MetadataContainer = LinearContainer<uint64_t, MetadataBackend>;

    private:

    sycl::queue* queue_ = &queueGlobal_;

    Tensor<T, DataBackend, MetadataBackend> tensor_{DataBackend(queue_), MetadataBackend(queue_)};

    public:

    template<typename U>
    using type = TensorParallel<U>;

    using value_type = T;
    //using memory_backend = MemoryBackendUSM<T, sycl::usm::alloc::device>;


    TensorParallel(const LinearContainer<uint64_t>& newDimensionSizes);

    //TensorParallel(const MetadataContainer& newDimensionSizes, const DataContainer& newData);

    //TensorParallel(span_view<uint64_t> newDimensionSizes);

    TensorParallel(const LinearContainer<uint64_t>& newDimensionSizes, const LinearContainer<T>& newData);// = delete;

    TensorParallel(const TensorParallel<T>& otherTensor);

    TensorParallel(TensorParallel<T>&& otherTensor) noexcept;

    template <typename OtherTensor>
    TensorParallel(OtherTensor* otherTensor);

    TensorParallel();

    ~TensorParallel();

    TensorParallel<T>& operator=(const TensorParallel<T>& otherTensor);
    
    TensorParallel<T>& operator=(TensorParallel<T>&& otherTensor) noexcept;



    const MetadataContainer& getDimensionSizes() const;

    uint64_t getNumberOfDimensions() const;

    uint64_t getNumberOfItems() const;

    const Tensor<T, DataBackend, MetadataBackend>& getTensor() const;
    sycl::queue* getQueue();


    T getItem(span_view<uint64_t> coordinates);

    void setItem(const T& value, span_view<uint64_t> coordinates);

    T* getData();
    const T* getData() const;

    TensorParallel<T>& setData(const LinearContainer<T>& tensorItems);
    //TensorParallel<T>& setData(const DataContainer& tensorItems);


    bool isValidCoordinates(span_view<uint64_t> coords) const;

    static bool isValidCoordinates(span_view<uint64_t> coords, span_view<uint64_t> dimensionSizes);

    bool isEquilateral() const;

    void fillWith(const T& fill);

    template<typename U> 
    friend std::ostream& operator<<(std::ostream& os, const TensorParallel<U>& tensor);

    std::string toString() const;

    bool operator==(const TensorParallel<T>& otherTensor) const;

    bool operator!=(const TensorParallel<T>& otherTensor) const;

    std::partial_ordering operator<=>(const TensorParallel<T>& otherTensor) const;




    TensorParallel<T> transpositionAndReturn(const uint64_t dim1 = 0, const uint64_t dim2 = 1) const;

    void transposition(const uint64_t dim1 = 0, const uint64_t dim2 = 1);

    void resize(const LinearContainer<uint64_t>& newDimensionSizes);

    void resize(const uint64_t newDimensionSize, const uint64_t dimensionIndex);
    
    void addDimension(const uint64_t newDimensionSize, const uint64_t putBefore);

    void removeDimension(const uint64_t removedDimensionIndex);



    template <apply_and_return_callable_parallel<T> C>
    auto applyAndReturn(const TensorParallel<T>& tensor2, C&& operation) const;
    
    template <typename A, typename B, apply_and_return_callable_parallel<T> C> 
    static auto applyAndReturn(const A& operand1, const B& operand2, C&& operation)
    requires(tensor_or_t_or_bothtensor_parallel<A, B, T>);

    template <apply_callable_parallel<T> C>
    void apply(const TensorParallel<T>& tensor2, C&& operation);

    // template <typename A, typename B, apply_callable_parallel<T> C> 
    // static void apply(A& operand1, const B& operand2, C&& operation)
    // requires(tensor_or_t_or_bothtensor_parallel<A, B, T>);

    template <apply_callable_parallel<T> C>
    static void apply(TensorParallel<T>& operand1, const TensorParallel<T>& operand2, C&& operation);

    template <apply_callable_parallel<T> C>
    static void apply(TensorParallel<T>& operand1, const T& operand2, C&& operation);

    template <apply_reverse_callable_parallel<T> C>
    static void apply(const T& operand1, TensorParallel<T>& operand2, C&& operation);


    template <foreach_and_return_callable_parallel<T> C>
    auto forEachAndReturn(C&& operation) const;

    template <foreach_and_return_callable_parallel<T> C>
    static auto forEachAndReturn(const TensorParallel<T>& tensor, C&& operation);

    template <foreach_callable_parallel<T> C> 
    void forEach(C&& operation);

    template <foreach_callable_parallel<T> C>
    static void forEach(TensorParallel<T>& tensor, C&& operation);

    template <apply_to_item_callable<T> C>
    void applyToItem(span_view<uint64_t> coords, C&& operation);




    // LinearContainer<uint64_t, MetadataBackend> getCoords(uint64_t itemIndex) const;
    // static void getCoords(uint64_t itemIndex, span_view<uint64_t> dimensionSizes, uint64_t* coordsBuffer);

    // uint64_t getIndex(span_view<uint64_t> coordinates) const;
    // static uint64_t getIndex(span_view<uint64_t> coordinates, span_view<uint64_t> dimensionSizes);

};

}

#include "TensorParallel.tpp"

#endif