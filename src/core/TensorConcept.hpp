#ifndef TENSOR_CONCEPTS_HPP
#define TENSOR_CONCEPTS_HPP

#include <cstdint>
#include <type_traits>
#include <vector>

#include "Utils.hpp"
#include "LinearContainer.hpp"

namespace gema {



// Forward declaration for the concepts.
template<class T, MemoryBackendConcept<T> DataMB = MemoryBackend<T>, 
MemoryBackendConcept<uint64_t> MetadataMB = MemoryBackend<uint64_t>>
class Tensor;


// Concept that checks if given type is tensor or derivate of it.
template <typename Type, class T>
concept TensorType = std::is_same_v<Type, Tensor<T>> || std::derived_from<Type, Tensor<T>>;


// template <typename C, typename A, typename B, class T>
// concept apply_callable = (std::is_same_v<A, T> && std::is_invocable_r_v<void, C, const T&, T&>) ||
//                          (std::is_same_v<B, T> && std::is_invocable_r_v<void, C, T&, const T&>) || 
//                          ((!std::is_same_v<A, T> && !std::is_same_v<B, T>) && std::is_invocable_r_v<void, C, T&, const T&>);

// Tensor computation callables

/// Checks for void(T&, const T&) invocable signature.
template <typename C, class T>
concept apply_callable = std::is_invocable_r_v<void, C, T&, const T&>;

/// Checks for void(const T&, T&) invocable signature.
template <typename C, class T>
concept apply_reverse_callable = std::is_invocable_r_v<void, C, const T&, T&>;

/// Checks for T(const T&, const T&) invocable signature.
template <typename C, class T>
//concept apply_and_return_callable = std::is_invocable_r_v<T, C, const T&, const T&>;
concept apply_and_return_callable = std::is_invocable_v<C, const T&, const T&>;

/// Checks for void(T&) invocable signature.
template <typename C, class T>
concept foreach_callable = std::is_invocable_r_v<void, C, T&>;

/// Checks for void(T&, const std::vector<uint64_t>&) invocable signature.
template <typename C, class T>
concept foreach_coord_callable = std::is_invocable_r_v<void, C, T&, const LinearContainer<uint64_t>&>;

// Checks for T(const T&) invocable signature.
template <typename C, class T>
//concept foreach_and_return_callable = std::is_invocable_r_v<T, C, const T&>;
concept foreach_and_return_callable = std::is_invocable_v<C, const T&>;


/// Checks for void(T&, const T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept apply_callable_parallel = apply_callable<C, T> ;//&& std::is_trivially_copyable_v<C>;

/// Checks for void(const T&, T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept apply_reverse_callable_parallel = apply_reverse_callable<C, T> ;//&& std::is_trivially_copyable_v<C>;

/// Checks for T(const T&, const T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept apply_and_return_callable_parallel = apply_and_return_callable<C, T> ;//&& std::is_trivially_copyable_v<C>;

/// Checks for void(T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept foreach_callable_parallel = foreach_callable<C, T> ;//&& std::is_trivially_copyable_v<C>;

/// Checks for void(T&, const std::vector<uint64_t>&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept foreach_coord_callable_parallel = foreach_coord_callable<C, T> ;//&& std::is_trivially_copyable_v<C>;

// Checks for T(const T&) invocable signature and invocable being trivially copyable.
template <typename C, class T>
concept foreach_and_return_callable_parallel = foreach_and_return_callable<C, T> ;//&& std::is_trivially_copyable_v<C>;


template <template <typename> class TensorT, class T>
concept tensor_computation_interface = requires (TensorT<T> tensor, T t){
    {true};
};

using Coords = LinearContainer<uint64_t>;

//template <template <typename> class Candidate, class T>
template <typename Candidate, typename T = typename Candidate::value_type>
concept TensorConcept = requires(
    Candidate tensor, 
    const Candidate ctensor, 
    Candidate otherTensor, 
    T t,
    const Candidate& tensorInput,
    Candidate& tensorInputOutput,
    const T& itemInput,
    const Coords& coordsInput
) {

    requires std::same_as<typename Candidate::value_type, T>;

    requires std::constructible_from<Candidate, const Coords&>;
    requires std::copy_constructible<Candidate>;
    requires std::move_constructible<Candidate>;
    requires std::default_initializable<Candidate>;

    { otherTensor = tensor } -> std::same_as<Candidate&>;
    { otherTensor = std::move(tensor) } -> std::same_as<Candidate&>;

    //{ ctensor.getDimensionSizes() } -> std::same_as<const LinearContainer<uint64_t>&>;
    { ctensor.getDimensionSizes() } -> std::convertible_to<std::span<const uint64_t>>;
    { ctensor.getNumberOfDimensions() } -> std::same_as<uint64_t>;
    { ctensor.getNumberOfItems() } -> std::same_as<uint64_t>;

    { tensor.getItem(coordsInput) } -> std::convertible_to<T>;//todo: is T really convertible to T? nope duh
    { tensor.setItem(itemInput, coordsInput) } -> std::same_as<void>;
    { tensor.getData() } -> std::convertible_to<T*>;
    { ctensor.getData() } -> std::convertible_to<const T*>;
    //{ tensor.setData(std::declval<const LinearContainer<T>&>()) };

    { ctensor.isValidCoordinates(coordsInput) } -> std::convertible_to<bool>;
    { Candidate::isValidCoordinates(coordsInput, coordsInput) } -> std::convertible_to<bool>;
    { ctensor.isEquilateral() } -> std::convertible_to<bool>;

    { ctensor == tensor } -> std::convertible_to<bool>;
    { ctensor != tensor } -> std::convertible_to<bool>;

    { ctensor.transpositionAndReturn(
        std::declval<const uint64_t>(), 
        std::declval<const uint64_t>()
    ) } -> std::same_as<Candidate>;
    { tensor.transposition(
        std::declval<const uint64_t>(), 
        std::declval<const uint64_t>()
    ) } -> std::same_as<void>;

    { tensor.resize(coordsInput) } -> std::same_as<void>;
    { tensor.addDimension(std::declval<const uint64_t>(), std::declval<const uint64_t>()) } -> std::same_as<void>;
    { tensor.removeDimension(std::declval<const uint64_t>()) } -> std::same_as<void>;

    { Candidate::applyAndReturn(
        tensorInput, 
        itemInput, 
        [](const T&, const T&) -> T {}
        //std::declval<T(const T&, const T&)>()
    ) } -> std::same_as<Candidate>;
    { Candidate::applyAndReturn(
        itemInput, 
        tensorInput,
        [](const T&, const T&) -> T {}
        //std::declval<T(const T&, const T&)>()
    ) } -> std::same_as<Candidate>;
    { Candidate::applyAndReturn(
        tensorInput, 
        tensorInput, 
        [](const T&, const T&) -> T {}
        //std::declval<T(const T&, const T&)>()
    ) } -> std::same_as<Candidate>;

    { Candidate::apply(
        tensorInputOutput, 
        itemInput, 
        [](T&, const T&) -> void {}
        //std::declval<void(T&, const T&)>()
    ) } -> std::same_as<void>;
    { Candidate::apply(
        itemInput, 
        tensorInputOutput, 
        [](const T&, T&) -> void {}
        //std::declval<void(const T&, T&)>()
    ) } -> std::same_as<void>;
    { Candidate::apply(
        tensorInputOutput, 
        tensorInput,
        [](T&, const T&) -> void {}
        //std::declval<void(T&, const T&)>()
    ) } -> std::same_as<void>;

    { Candidate::forEachAndReturn(
        tensorInput, 
        [](const T&) -> T {}
        //std::declval<T(const T&)>()
    ) } -> std::same_as<Candidate>;
    { Candidate::forEach(
        tensorInputOutput, 
        [](T&) -> void {}
        //std::declval<void(T&)>()
    ) } -> std::same_as<void>;
};


}



#endif