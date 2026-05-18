#include <algorithm>
#include <bit>
#include <cmath>
#include <compare>
#include <cstdint>
#include <format>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

#include "MemoryBackendConcept.hpp"
#include "TensorConcept.hpp"
#include "Tensor.hpp"

//#include "Tensor.hpp" // Not needed, just keep it for intelisense

namespace gema{

    // TODO: choose what to delete
    template<class T>
    concept has_to_string = requires(T& t) {
        { t.to_string() } -> std::convertible_to<std::string>;
    };

    template<typename T>
    concept has_ostream = requires(T& t, std::ostream& os) {
        { os << t } -> std::same_as<std::ostream&>;
    };
}

// STD specializations of formatter
namespace std {

    template <class T>
    struct formatter<gema::Tensor<T>, char> {

        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();  // No custom format specs supported
        }

        auto format(const gema::Tensor<T>& tensor, format_context& ctx) const {
            // Use tensor.toString() to get the string representation
            return format_to(ctx.out(), "{}", tensor.toString());
        }
    };

    template <class T>
    struct formatter<gema::Tensor<T>*, char> {

        constexpr auto parse(format_parse_context& ctx) {
            return ctx.begin();  // No custom format specs supported
        }

        auto format(const gema::Tensor<T>* tensor, format_context& ctx) const {
            // Use tensor.toString() to get the string representation
            return format_to(ctx.out(), "{}", tensor->toString());
        }
    };
}

namespace gema {

    constexpr int maxLoopCount = 65536; // Will be probably unused.

    // Template helpers:
    
    // Converts from floating types to integral of same size
    template <typename F>
    struct to_integral {
        using type = std::conditional_t<sizeof(F) == 1, uint8_t,
                     std::conditional_t<sizeof(F) == 2, uint16_t,
                     std::conditional_t<sizeof(F) == 4, uint32_t, 
                     std::conditional_t<sizeof(F) == 8, uint64_t,
                     void>>>>;
    };

    // If F is floating point, it will get converted to integral and if not, then just returned
    template <typename F>
    struct integral_if_float{
        using type = std::conditional_t<std::is_floating_point_v<F>, typename to_integral<F>::type, F>;
    };

    // If argument is floating type, it will get bitcasted to integral of same size, otherwise just returned
    template <typename T>
    inline constexpr typename integral_if_float<T>::type bitcast_if_float(const T& value){

        // use this if the bitcast wont get optimized away to nonfloating types
        typename integral_if_float<T>::type valueBits;

        if constexpr(std::is_floating_point<T>::value){
            valueBits = std::bit_cast<typename integral_if_float<T>::type>(value);
        }else{
            valueBits = value;
        }

        return valueBits;
    }

    // Takes two parameters and returns length of tensor_ of the first one to be type Tensor<T>
    template <typename T, typename A, typename B>
    consteval uint64_t tensor_size(const A& operand1, const B& operand2){

        if constexpr (std::is_same_v<A, Tensor<T>>){
            return operand1.tensor_.size();
        } else if constexpr (std::is_same_v<B, Tensor<T>>){
            return operand2.tensor_.size();
        }else{
            return 1;
        }
    }

    // Returns first argument of two arguments that matches given type
    template <typename X, typename T, typename A, typename B>
    consteval Tensor<T>* type_pick_b(const A& operand1, const B& operand2){
        
        if constexpr (std::is_same_v<A, X>){
            return &operand1;
        } else if constexpr (std::is_same_v<B, X>){
            return &operand2;
        } else {
            return nullptr;
        }
    }

    // Returns first argument that matches given type
    template <typename X, typename F, typename... R>
    inline const X* type_pick(const F& first, const R&... rest) { 

        if constexpr (std::is_same_v<std::decay_t<F>, X>) {
            return (&first);
        } else if constexpr (sizeof...(rest) > 0) {
            return type_pick<X>(rest...);
        } else {
            return nullptr; // No match found
        }
    }
    
    // Returns first type of two type arguments that matches given type
    template <typename X, typename A, typename B, typename T>
    struct first_of_specified{
        using type =    std::conditional<std::is_same_v<A, X>, A,
                        std::conditional<std::is_same_v<B, X>, B, 
                        void>>;
    };

    template<typename T>
    inline decltype(auto) total_dereference(T&& x) {
        if constexpr (std::is_pointer_v<std::remove_reference_t<T>>) {
            return deref_all(*x);  // dereference one level and recurse
        } else {
            return std::forward<T>(x);  // base case: not a pointer
        }
    }

    

    



    // STATIC PRIVATE DEFAULT VALUES: -----------------------------------------------------------------------------------------

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // inline std::function<EqualsCallable<T>> Tensor<T, DataMB, MetadataMB>::defaultEquals_ = [] (const T& a, const T& b) {
        
    //     if constexpr (std::is_floating_point<T>::value) {
    //         T epsilon = std::numeric_limits<T>::epsilon();
    //         return std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)));
    //     } else {
    //         return a == b;
    //     }
    // };


    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // inline std::function<OrderCallable<T>> Tensor<T, DataMB, MetadataMB>::defaultOrder_ = [] (const T& a, const T& b) {

    //     if constexpr (std::is_floating_point<T>::value) {
    //         T epsilon = std::numeric_limits<T>::epsilon();
    //         if (std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)))){
    //             return 0;
    //         }else if(a > b){
    //             return 1;
    //         }else{
    //             return -1;
    //         }

    //     } else if constexpr (std::is_integral<T>::value){
    //         return (a != b) * ((a > b) + -(a < b));
    //     }else{
    //         return 0;
    //     }
    // };






    // PUBLIC METHODS: --------------------------------------------------------------------------------------------------------

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>::Tensor(const MetadataContainer& newDimensionSizes) 
    : dimensionSizes_(newDimensionSizes){
        update();
    }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // Tensor<T, DataMB, MetadataMB>::Tensor(span_view<uint64_t> newDimensionSizes)
    // : dimensionSizes_(newDimensionSizes){

    // }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>::Tensor(const MetadataContainer& newDimensionSizes,
    const DataMB& memoryBackend)
    : tensor_(memoryBackend), dimensionSizes_(newDimensionSizes), dimensionJumps_(newDimensionSizes.getMemoryBackend()){ 
        update();
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>::Tensor(const DataMB& memoryBackend, const MetadataMB& metadataBackend)
    : tensor_(memoryBackend), dimensionSizes_(metadataBackend), dimensionJumps_(metadataBackend){
        
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    inline Tensor<T, DataMB, MetadataMB>::Tensor
    (const MetadataContainer& newDimensionSizes, const DataContainer& newData) 
    : dimensionSizes_(newDimensionSizes), tensor_(newData), dimensionJumps_(newDimensionSizes.getMemoryBackend()){

        // Check actual capacity of dimensions to tensorData

        update();
    }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // Tensor<T, DataMB, MetadataMB>::Tensor(
    //     span_view<uint64_t> newDimensionSizes, 
    //     const DataMB& dataBackend, 
    //     const MetadataMB& metadataBackend
    // ) : dimensionSizes_(newDimensionSizes, metadataBackend), dimensionJumps_(metadataBackend), tensor_(dataBackend){

    //     update();
    // }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>::Tensor(const Tensor<T, DataMB, MetadataMB>& otherTensor)
    :   tensor_(otherTensor.tensor_), 
        dimensionSizes_(otherTensor.dimensionSizes_),
        dimensionJumps_(otherTensor.dimensionJumps_)
    {
        //*this = otherTensor;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>::Tensor(Tensor<T, DataMB, MetadataMB>&& otherTensor) noexcept
    :   tensor_(std::move(otherTensor.tensor_)), 
        dimensionSizes_(std::move(otherTensor.dimensionSizes_)),
        dimensionJumps_(std::move(otherTensor.dimensionJumps_))
    {
        //*this = std::move(otherTensor);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <typename OtherTensor>
    Tensor<T, DataMB, MetadataMB>::Tensor(const OtherTensor* otherTensor)
    :   tensor_(DataMB(otherTensor->tensor_.getMemoryBackend())),
        dimensionSizes_(otherTensor->dimensionSizes_.getMemoryBackend()),
        dimensionJumps_(otherTensor->dimensionJumps_.getMemoryBackend())
    {

        tensor_.resize(otherTensor->tensor_.size());
        dimensionSizes_ = otherTensor->dimensionSizes_;
        dimensionJumps_ = otherTensor->dimensionJumps_;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>::Tensor(){

    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    const LinearContainer<uint64_t, MetadataMB>& Tensor<T, DataMB, MetadataMB>::getDimensionSizes() const{
        return dimensionSizes_;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    uint64_t Tensor<T, DataMB, MetadataMB>::getNumberOfDimensions() const{
        return dimensionSizes_.size();
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    uint64_t Tensor<T, DataMB, MetadataMB>::getNumberOfItems() const{
        return tensor_.size();
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    T& Tensor<T, DataMB, MetadataMB>::getItem(span_view<uint64_t> coordinates){

        return tensor_[getIndex(coordinates)];
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::setItem(const T& value, span_view<uint64_t> coordinates){

        uint64_t itemIndex = getIndex(coordinates);
        tensor_[itemIndex] = value;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    T* Tensor<T, DataMB, MetadataMB>::getData(){
        return tensor_.data();
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    const T* Tensor<T, DataMB, MetadataMB>::getData() const {
        return tensor_.data();
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>::DataContainer& Tensor<T, DataMB, MetadataMB>::getDataContainer(){
        return tensor_;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    const Tensor<T, DataMB, MetadataMB>::DataContainer& Tensor<T, DataMB, MetadataMB>::getDataContainer() const{
        return tensor_;
    }

    // Secure version will need to check for correct tensorItems size
    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>& Tensor<T, DataMB, MetadataMB>::setData(const LinearContainer<T, DataMB>& tensorItems){
        tensor_ = tensorItems;
        return *this;
    }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // void Tensor<T, DataMB, MetadataMB>::setEquals(const std::function<EqualsCallable<T>>& equals){
        
    //     userEquals_ = equals;
    //     equals_ = &userEquals_;
    // }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // void Tensor<T, DataMB, MetadataMB>::setOrder(const std::function<OrderCallable<T>>& order){

    //     userOrder_ = order;
    //     order_ = &userOrder_;
    // }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    bool Tensor<T, DataMB, MetadataMB>::isValidCoordinates(span_view<uint64_t> coords) const{
        return Tensor<T, DataMB, MetadataMB>::isValidCoordinates(coords, dimensionSizes_);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    /*static*/ bool Tensor<T, DataMB, MetadataMB>::isValidCoordinates(
        span_view<uint64_t> coords, 
        span_view<uint64_t> dimensionSizes
    ){

        if(dimensionSizes.size() != coords.size()) return false;

        for(uint64_t i = 0; i < dimensionSizes.size(); i++){
            if(coords[i] >= dimensionSizes[i]) return false;
        }

        return true;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    bool Tensor<T, DataMB, MetadataMB>::isEquilateral() const{
        return std::adjacent_find(dimensionSizes_.begin(), dimensionSizes_.end(), std::not_equal_to<int>()) == dimensionSizes_.end();
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    std::string Tensor<T, DataMB, MetadataMB>::toString() const{

        std::vector<std::string> openingBrackets(tensor_.size());
        std::fill(openingBrackets.begin(), openingBrackets.end(), "");

        std::vector<std::string> closingBrackets(tensor_.size());
        std::fill(closingBrackets.begin(), closingBrackets.end(), "");

        uint64_t dimensionProduct = tensor_.size();

        //for(uint64_t i = dimensionSizes_.size() - 1; (i >= 0) && (i < dimensionSizes_.size()); --i){ // Opposite endianness
        for(uint64_t i = 0; i < dimensionSizes_.size(); i++){ // Identity endianness

            for(uint64_t j = 0; j < tensor_.size(); ++j){

                if(j % dimensionProduct == 0){
                    openingBrackets[j] += "{";
                }

                if(j % dimensionProduct == (dimensionProduct - 1)){
                    closingBrackets[j] += "}";
                }
            }

            dimensionProduct /= dimensionSizes_[i];
        }

        std::string output = "";

        for(uint64_t i = 0; i < tensor_.size(); ++i){
            output += 
                std::format("{}{}{}{}", openingBrackets[i], tensor_[i], closingBrackets[i], (((i + 1) >= tensor_.size()) ? "" : ", "));
        }

        return output; 
    }

    template <typename U, MemoryBackendConcept<U> MB>
    std::ostream& operator<<(std::ostream& os, const Tensor<U, MB>& tensor){
        return os << tensor.toString();
    }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // void Tensor<T, DataMB, MetadataMB>::parse(const std::string& tensorString, const std::function<const T(const std::string&)>& parseItem){
        
    //     uint64_t i;
    //     for(i = 0; tensorString[i] == '{'; ++i){}
    //     std::vector<int> parsedDimensionSizes(i);

    // }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::fillWith(const T& value){

        // could use assign method on everything thus dodging the specialization but std::fill is probably faster
        //tensor_.assign(tensor_.size(), value);

        // std::fill(tensor_.begin(), tensor_.end(), value); // probably better optimalized

        /*for(T& item : tensor_){
            item = value;
        }*/

        // Or maybe this?
        tensor_.fill(value);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB> Tensor<T, DataMB, MetadataMB>::transpositionAndReturn(
        const uint64_t dim1, const uint64_t dim2
    ) const {

        if(dim1 == dim2) return Tensor<T, DataMB, MetadataMB>(dimensionSizes_);

        // Copying the dimensionSizes
        // Change assigment to just construction of correct size
        LinearContainer<uint64_t, MetadataMB> transposedDimensionSizes = dimensionSizes_; 

        // Swapping the dimension sizes
        transposedDimensionSizes[dim1] = dimensionSizes_[dim2]; 
        transposedDimensionSizes[dim2] = dimensionSizes_[dim1];
        
        // Initializing the new tensor
        Tensor<T, DataMB, MetadataMB> tensorTransposed = 
            Tensor<T, DataMB, MetadataMB>(transposedDimensionSizes, tensor_.getMemoryBackend());

        LinearContainer<uint64_t> original, switched;
        original.resize(dimensionSizes_.size());
        switched.resize(dimensionSizes_.size());
        original.fill(0);

        // Looping through elements in tensor and swapping the desired coordinates
        for(uint64_t i = 0; i < tensor_.size(); ++i){
            
            // The swap of two desired coordinates
            switched = original;
            switched[dim1] = original[dim2];
            switched[dim2] = original[dim1];

            // Works until now, check the getIndex function if it actually works properly
            tensorTransposed.tensor_[tensorTransposed.getIndex(switched)] = tensor_[i];

            //std::cout << original[0] << " " << original[1] << std::endl;
            Tensor<T, DataMB, MetadataMB>::incrementCoords(original, dimensionSizes_);
        }

        return tensorTransposed;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::transposition(const uint64_t dim1, const uint64_t dim2){

        if(dim1 == dim2) return;

        // Copying the dimensionSizes
        // Change assigment to just construction of correct size
        const LinearContainer<uint64_t> oldDimensionSizes = dimensionSizes_.copyToBackend(MemoryBackend<uint64_t>());

        // Swapping the dimension sizes
        const uint64_t temporaryDimensionSize1 = dimensionSizes_[dim1];
        dimensionSizes_[dim1] = dimensionSizes_[dim2];
        dimensionSizes_[dim2] = temporaryDimensionSize1;

        const uint64_t itemCount = updateDimensionJump();
        
        // Initializing the new data
        LinearContainer<T, DataMB> newTensorData(itemCount, tensor_.getMemoryBackend());

        LinearContainer<uint64_t> original, switched;
        original.resize(dimensionSizes_.size());
        switched.resize(dimensionSizes_.size());
        original.fill(0);

        // Looping through elements in tensor and swapping the desired coordinates
        for(uint64_t i = 0; i < itemCount; ++i){
            
            //if (original[dim1] == original[dim2]) [[unlikely]] continue;

            // The swap of two desired coordinates
            switched = original;
            switched[dim1] = original[dim2];
            switched[dim2] = original[dim1];

            // Works until now, check the getIndex function if it actually works properly
            newTensorData[getIndex(switched)] = std::move(tensor_[i]);

            Tensor<T, DataMB, MetadataMB>::incrementCoords(original, oldDimensionSizes);
        }

        tensor_ = std::move(newTensorData);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::resize(const LinearContainer<uint64_t>& newDimensionSizes){

        const LinearContainer<uint64_t> oldDimensionSizes = dimensionSizes_;
        dimensionSizes_ = LinearContainer(newDimensionSizes);
        const uint64_t newItemCount = updateDimensionJump();

        // New allocation because it is likely anyway, even if tensor_.resize() would be used
        // Because even change of 1 to any dimension size likely means multiplicative increase/decrease in item count
        LinearContainer<T> newTensor(newItemCount);

        LinearContainer<uint64_t> currentCoordsSource(oldDimensionSizes.size());
        currentCoordsSource.fill(0);

        for(uint64_t i = 0; i < tensor_.size(); i++){

            if(isValidCoordinates(currentCoordsSource)){
                uint64_t destinationIndex = getIndex(currentCoordsSource);
                newTensor[destinationIndex] = std::move(tensor_[i]);
            }

            incrementCoords(currentCoordsSource, oldDimensionSizes);
        }
        
        tensor_ = std::move(newTensor);
    }

    // template<class T>
    // void Tensor<T, DataMB, MetadataMB>::resize(uint64_t newSize, uint64_t dim) {
    //     auto oldSizes = dimensionSizes_;
    //     dimensionSizes_[dim] = newSize;
    //     uint64_t newCount = updateDimensionJump();

    //     LinearContainer<T> newTensor(newCount);

    //     std::vector<uint64_t> coords(oldSizes.size(), 0);

    //     while (true) {
    //         // check if coords fit into new tensor
    //         if (coords[dim] < newSize) {
    //             uint64_t oldIdx = getIndex(coords, oldSizes);
    //             uint64_t newIdx = getIndex(coords, dimensionSizes_);
    //             newTensor[newIdx] = std::move(tensor_[oldIdx]);
    //         }

    //         // increment coords
    //         int d = coords.size() - 1;
    //         while (d >= 0) {
    //             coords[d]++;
    //             if (coords[d] < oldSizes[d]) break;
    //             coords[d] = 0;
    //             d--;
    //         }
    //         if (d < 0) break;
    //     }

    //     tensor_ = std::move(newTensor);
    // }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::resize(const uint64_t newDimensionSize, const uint64_t dimensionIndex) {

        LinearContainer<uint64_t> newDimensionSizes = dimensionSizes_;
        newDimensionSizes[dimensionIndex] = newDimensionSize;
        resize(newDimensionSizes);

        // const std::vector<uint64_t> oldDimensionSizes = dimensionSizes_;
        // dimensionSizes_[dimensionIndex] = newDimensionSize;
        // const uint64_t newItemCount = updateDimensionJump();

        // // New allocation because it is likely anyway, even if tensor_.resize() would be used
        // // Because even change of 1 to any dimension size likely means multiplicative increase/decrease in item count
        // LinearContainer<T> newTensor(newItemCount);

        // std::vector<uint64_t> currentCoordsSource(oldDimensionSizes.size(), 0);

        // for(uint64_t i = 0; i < tensor_.size(); i++){

        //     if(isValidCoordinates(currentCoordsSource)){
        //         uint64_t destinationIndex = getIndex(currentCoordsSource);
        //         newTensor[destinationIndex] = std::move(tensor_[i]);
        //     }

        //     incrementCoords(currentCoordsSource, oldDimensionSizes);
        // }
        
        // tensor_ = std::move(newTensor);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::addDimension(const uint64_t newDimensionSize, const uint64_t putBefore){

        const LinearContainer<uint64_t, MetadataMB> oldDimensionSizes = dimensionSizes_; //todo: probably delete
        const LinearContainer<uint64_t, MetadataMB> oldDimensionJumps = dimensionJumps_; //todo: probably delete

        dimensionSizes_.insert(dimensionSizes_.begin() + putBefore, newDimensionSize);

        const uint64_t newItemCount = updateDimensionJump();

        LinearContainer<T, DataMB> newTensor(newItemCount);

        uint64_t j = 0;
        uint64_t jCounter = 0;
        const uint64_t moveAmount = dimensionJumps_[putBefore];
        const uint64_t skipAmount = moveAmount * (dimensionSizes_[putBefore] - 1);
        for(uint64_t i = 0; i < tensor_.size(); i++, j++){

            newTensor[j] = std::move(tensor_[i]);

            if(++jCounter >= moveAmount)[[unlikely]]{
                j += skipAmount;
                jCounter = 0;
            }
        }

        // std::vector<uint64_t> currentCoordsSource(oldDimensionSizes.size(), 0);

        // for(uint64_t i = 0; i < tensor_.size(); i++){

        //     if(isValidCoordinates(currentCoordsSource)){
        //         uint64_t destinationIndex = getIndex(currentCoordsSource);
        //         newTensor[destinationIndex] = std::move(tensor_[i]);
        //     }

        //     incrementCoords(currentCoordsSource, oldDimensionSizes);
        // }

        tensor_ = std::move(newTensor);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::removeDimension(const uint64_t removedDimensionIndex){

        const LinearContainer<uint64_t, MetadataMB> oldDimensionSizes = dimensionSizes_;
        const LinearContainer<uint64_t, MetadataMB> oldDimensionJumps = dimensionJumps_;

        dimensionSizes_.erase(dimensionSizes_.begin() + removedDimensionIndex);

        const uint64_t newItemCount = updateDimensionJump();

        LinearContainer<T, DataMB> newTensor(newItemCount);
        
        uint64_t j = 0;
        uint64_t jCounter = 0;
        const uint64_t moveAmount = oldDimensionJumps[removedDimensionIndex];
        const uint64_t skipAmount = moveAmount * (oldDimensionSizes[removedDimensionIndex] - 1);
        for(uint64_t i = 0; i < newTensor.size(); i++, j++){

            newTensor[i] = std::move(tensor_[j]);

            if(++jCounter >= moveAmount)[[unlikely]]{
                j += skipAmount;
                jCounter = 0;
            }
        }

        // std::vector<uint64_t> currentCoordsSource(oldDimensionSizes.size(), 0);

        // for(uint64_t i = 0; i < tensor_.size(); i++){

        //     if(isValidCoordinates(currentCoordsSource)){
        //         uint64_t destinationIndex = getIndex(currentCoordsSource);
        //         newTensor[destinationIndex] = std::move(tensor_[i]);
        //     }

        //     incrementCoords(currentCoordsSource, oldDimensionSizes);
        // }

        tensor_ = std::move(newTensor);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <binary_operation_on_items<T> I>
    void Tensor<T, DataMB, MetadataMB>::collapseDimension(const uint64_t dimensionIndex, const I &binaryOperation)
    {
    }

    // SPECIAL OPERATOR OVERLOADS ---------------------------------------------------------------------------------------------
    // Does not need macros.
    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>& Tensor<T, DataMB, MetadataMB>::operator=(const Tensor<T, DataMB, MetadataMB>& otherTensor){
        
        this->tensor_ = otherTensor.tensor_;
        this->dimensionSizes_ = otherTensor.dimensionSizes_;
        this->dimensionJumps_ = otherTensor.dimensionJumps_;

        // TODO: decide if to actually copy this
        // TODO: can it be done without if statements?
        // Argument for yes: tensor that does not have these functions defined is not comparable
        // if(tensor2.userEquals_){
        //     setEquals(tensor2.userEquals_);
        // }

        // if(tensor2.userOrder_){
        //     setOrder(tensor2.userOrder_);
        // }

        // TODO: decide what to do with tensorOutput and itemOutput
        return *this;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>& Tensor<T, DataMB, MetadataMB>::operator=(Tensor<T, DataMB, MetadataMB>&& otherTensor) noexcept{
        
        tensor_ = std::move(otherTensor.tensor_);
        dimensionSizes_ = std::move(otherTensor.dimensionSizes_);
        dimensionJumps_ = std::move(otherTensor.dimensionJumps_);
        
        return *this;
    }

    // Do not simplify
    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    bool Tensor<T, DataMB, MetadataMB>::operator==(const Tensor<T, DataMB, MetadataMB>& otherTensor) const{

        // Values should be compared first, as tensors of same dimensions are more likely to be compared
        //return (this->tensor_ == otherTensor.tensor_) && (this->dimensionSizes_ == otherTensor.dimensionSizes_);

        // Too complicated
        /*if(this->tensor_.size() != otherTensor.tensor_.size()){
            return false;
        }

        // Check if all items are equal
        return std::equal(this->tensor_.begin(), this->tensor_.end(), otherTensor.tensor_.begin(), [&](const auto& a, const auto& b){

            return compareItems(a, b);
        });*/



        // This is the implementation similar to std::vector::operator== workings, but with possibility of custom comparison function
        // return std::equal(this->tensor_.begin(), this->tensor_.end(), otherTensor.tensor_.begin(), [this](const auto& a, const auto& b){

        //     return (*equals_)(a, b); //was: compareItems

        // }) && (this->dimensionSizes_ == otherTensor.dimensionSizes_); // Could be also: !(this->tensor_.size() - tensor2.tensor_.size())
    
        return (dimensionSizes_ == otherTensor.dimensionSizes_) && (tensor_ == otherTensor.tensor_);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    bool Tensor<T, DataMB, MetadataMB>::operator!=(const Tensor<T, DataMB, MetadataMB>& otherTensor) const
    {
        return !(*this == otherTensor);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    std::partial_ordering Tensor<T, DataMB, MetadataMB>::operator<=>(const Tensor<T, DataMB, MetadataMB>& otherTensor) const {

        if(dimensionSizes_ != otherTensor.dimensionSizes_) return std::partial_ordering::unordered;

        return tensor_ <=> otherTensor.tensor_;
    }


    // OPERATOR OVERLOADS -----------------------------------------------------------------------------------------------------
    // Operator overload implemetations are often repetetive. To reduce code duplicates, macros are created for the overloads
    // and their variants. A naming convention has been created to include in macro name information about its abstract
    // signature.
    //
    // Notation explanation:
    // The last letters in macro name (after the last _) mean type of operation (similar to function signature).
    // T - Tensor<T>
    // V - value of type T from Tensor<T>
    // o - operation
    // oe - operation in place
    // r - results in
    //
    // Examples: 
    // ToTrT means Tensor performing Operation with Tensor Resulting in new Tensor (Tensor operation Tensor = Tensor).
    // ToeV means Tensor performing Operation with Value in place (Tensor oepration= Value).

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // constexpr inline auto& force_dereference(T& pointer){//bad

    //     if constexpr (std::is_pointer_v<T>){
    //         return *pointer;
    //     } else if constexpr (
    //         std::is_same_v<T, std::unique_ptr<typename T::element_type>> || 
    //         std::is_same_v<T, std::shared_ptr<typename T::element_type>>)
    //     {
    //         return *pointer;
    //     } else{
    //         return pointer;
    //     }
    // }

    // // ARITHMETIC BINARY GENERIC MACRO ----------------------------------------------------------------------------------------
    // // Artihmetic binary is an binary operation on two arithmetic types (or ones with overloaded operators acting like 
    // // arithmetic).
    // #define ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    // /**/\
    //     template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>\
    //     inline auto Tensor<T, DataMB, MetadataMB>::operator OP_SYMBOL(const Tensor<T>& tensor2) const\
    //     requires requires (T a, T b) {a OP_SYMBOL b;}{\
    // /**/\
    //         return applyAndReturn(*this, tensor2, [](const T& tensorItem, const T& tensor2Item){\
    //                 return tensorItem OP_SYMBOL tensor2Item;\
    //         });\
    //     }\
    // /**/

    // #define ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    // /**/\
    //     template<class T>\
    //     inline auto operator OP_SYMBOL(const Tensor<T>& tensor, const T& value)\
    //     requires requires (T a, T b) {a OP_SYMBOL b;}{\
    // /**/\
    //         return Tensor<T, DataMB, MetadataMB>::forEachAndReturn(tensor, [&value](const T& item){\
    //             return item OP_SYMBOL value;\
    //         });\
    //     }\
    // /**/

    // #define ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)\
    // /**/\
    //     template<class T>\
    //     inline auto operator OP_SYMBOL(const T& value, const Tensor<T>& tensor)\
    //     requires requires (T a, T b) {a OP_SYMBOL b;}{\
    // /**/\
    //         /* Do not delegate switched argument operator! While on numbers set the operation would be often commutative, */\
    //         /* it is not guaranteed to be so on every type and operation!*/\
    //         return Tensor<T, DataMB, MetadataMB>::forEachAndReturn(tensor, [&value](const T& item){\
    //             return value OP_SYMBOL item;\
    //         });\
    //     }\
    // /**/

    // #define ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    // /**/\
    //     template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>\
    //     void Tensor<T, DataMB, MetadataMB>::operator OP_SYMBOL##=(const Tensor<T>& tensor2)\
    //     requires requires (T a, T b) {a OP_SYMBOL##= b;}{\
    // /**/\
    //         apply(tensor2, [](T& tensorItem, const T& tensor2Item){\
    //             tensorItem OP_SYMBOL##= tensor2Item;\
    //         });\
    //     }\
    // /**/

    // #define ARITHMETIC_BINARY_ToeV(OP_SYMBOL)\
    // /**/\
    //     template<class T>\
    //     void Tensor<T, DataMB, MetadataMB>::operator OP_SYMBOL##=(const T& value)\
    //     requires requires (T a, T b) {a OP_SYMBOL##= b;}{\
    // /**/\
    //         forEach([&value](T& item){\
    //             item OP_SYMBOL##= value;\
    //         });\
    //     }\
    // /**/

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

    // // Some logical overloads for binary operations are not making sense for logical operators
    // #define LOGICAL_BINARY(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_VoTrT(OP_SYMBOL)

    // LOGICAL_BINARY(&&)
    // LOGICAL_BINARY(||)

    // // Some bitwise overloads for binary operations are not making sense for bitshift
    // #define BITSHIFTLIKE(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToTrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToVrT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeT(OP_SYMBOL)\
    //     ARITHMETIC_BINARY_ToeV(OP_SYMBOL)

    // BITSHIFTLIKE(<<)
    // BITSHIFTLIKE(>>)

    // #undef ARITHMETIC_BINARY_ToTrT // Macros no longer needed
    // #undef ARITHMETIC_BINARY_ToVrT
    // #undef ARITHMETIC_BINARY_VoTrT
    // #undef ARITHMETIC_BINARY_ToeT
    // #undef ARITHMETIC_BINARY_ToeV

    // #undef BITSHIFTLIKE
    // #undef LOGICAL_BINARY

    // #undef ARITHMETIC_BINARY

    // Needed specialization for % because it is not normally supported for floating types

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // inline auto Tensor<T, DataMB, MetadataMB>::operator %(const Tensor<T>& tensor2) const{
    //     return applyAndReturn(*this, tensor2, [](const T& tensorItem, const T& tensor2Item){
            
    //         if constexpr(std::is_floating_point_v<T>){
    //             return std::fmod(tensorItem, tensor2Item);
    //         }else{
    //             return tensorItem % tensor2Item;
    //         }
    //     });
    // }

    // template<class T>
    // inline auto operator %(const Tensor<T>& tensor, const T& value){
    //     return Tensor<T, DataMB, MetadataMB>::forEachAndReturn(tensor, [&value](const T& item){
    //         //return item OP_SYMBOL value;
    //         if constexpr(std::is_floating_point_v<T>){
    //             return std::fmod(item, value);
    //         }else{
    //             return item % value;
    //         }
    //     });
    // }

    // template<class T>
    // inline auto operator %(const T& value, const Tensor<T>& tensor){
    //     /* Do not delegate switched argument operator! While on numbers set the operation would be often commutative, */
    //     /* it is not guaranteed to be so on every type and operation!*/
    //     return Tensor<T, DataMB, MetadataMB>::forEachAndReturn(tensor, [&value](const T& item){
    //         //return value OP_SYMBOL item;
    //         if constexpr(std::is_floating_point_v<T>){
    //             return std::fmod(value, item);
    //         }else{
    //             return value % item;
    //         }
    //     });
    // }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // void Tensor<T, DataMB, MetadataMB>::operator %=(const Tensor<T>& tensor2){
    //     apply(tensor2, [](T& tensorItem, const T& tensor2Item){
    //         //tensorItem %= tensor2Item;
    //         if constexpr(std::is_floating_point_v<T>){
    //             tensorItem = std::fmod(tensorItem, tensor2Item);
    //         }else{
    //             tensorItem %= tensor2Item;
    //         }
    //     });
    // }

    // template<class T>
    // void Tensor<T, DataMB, MetadataMB>::operator %=(const T& value){
    //     forEach([&value](T& item){
    //         //item %= value;
    //         if constexpr(std::is_floating_point_v<T>){
    //             item = std::fmod(item, value);
    //         }else{
    //             item %= value;
    //         }
    //     });
    // }


    // UNARY OPERATION GENERIC MACRO ------------------------------------------------------------------------------------------

    // #define UNARY_OPERATION(OP_SYMBOL)\
    //     template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>\
    //     auto Tensor<T, DataMB, MetadataMB>::operator OP_SYMBOL() const\
    //     requires requires (T a) {OP_SYMBOL a;}{\
    // /**/\
    //         return forEachAndReturn([](const T& item){\
    //             return OP_SYMBOL item;\
    //         });\
    //     }\
    // /**/

    // UNARY_OPERATION(~)
    // UNARY_OPERATION(!)
    // UNARY_OPERATION(+)
    // UNARY_OPERATION(-)

    // #undef UNARY_OPERATION


    
    // #define UNARY_OPERATION_INPLACE(OP_SYMBOL, OP_NAME)\
    //     template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>\
    //     void Tensor<T, DataMB, MetadataMB>::OP_NAME##InPlace(){\
    //         forEach([](T& item){\
    //             item = OP_SYMBOL item;\
    //         });\
    //     }

    // UNARY_OPERATION_INPLACE(~, complement)
    // UNARY_OPERATION_INPLACE(+, plus)
    // UNARY_OPERATION_INPLACE(-, opposite)

    // #undef UNARY_OPERATION_INPLACE




    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // void Tensor<T, DataMB, MetadataMB>::complementInPlace(){
    //     forEach([](T& item){
    //         item = ~item;
    //     });
    // }
    
    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // void Tensor<T, DataMB, MetadataMB>::plusInPlace(){
    //     forEach([](T& item){
    //         item = +item;
    //     });
    // }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // void Tensor<T, DataMB, MetadataMB>::oppositeInPlace(){
    //     forEach([](T& item){
    //         item = -item;
    //     });
    // }



    // #define PREFIX_POSTFIX(OP_SYMBOL)\
    //     template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>\
    //     Tensor<T>& Tensor<T, DataMB, MetadataMB>::operator OP_SYMBOL(){\
    // /**/\
    //         forEach([](T& item){\
    //             OP_SYMBOL item;\
    //         });\
    // /**/\
    //         return *this;\
    //     }\
    // /**/\
    //     template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>\
    //     Tensor<T> Tensor<T, DataMB, MetadataMB>::operator OP_SYMBOL(int) const{\
    //         Tensor<T> temporary(*this);\
    //         operator++();\
    //         return temporary;\
    //     }\
    // /**/

    // PREFIX_POSTFIX(++)
    // PREFIX_POSTFIX(--)

    // #undef PREFIX_POSTFIX



    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <apply_and_return_callable<T> C>
    auto Tensor<T, DataMB, MetadataMB>::applyAndReturn(const Tensor<T>& tensor2, C&& operation) const {

        //std::transform(tensor_.begin(), tensor_.end(), tensor2.tensor_.begin(), resultTensor->tensor_.begin(), operation);
        return Tensor<T, DataMB, MetadataMB>::applyAndReturn(*this, tensor2, std::forward<C>(operation));
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <typename A, typename B, apply_and_return_callable<T> C>
    /*static*/ auto gema::Tensor<T, DataMB, MetadataMB>::applyAndReturn(const A& operand1, const B& operand2, C&& operation)
    requires(tensor_or_t_or_bothtensor<A, B, T>){
        
        using opReturnType = decltype(operation(std::declval<T>(), std::declval<T>()));
        const Tensor<T>* tensorOperand = type_pick<Tensor<T>>(operand1, operand2);
        Tensor<opReturnType> resultTensor = Tensor<opReturnType>(tensorOperand->getDimensionSizes());

        opReturnType* resultTensorData = resultTensor.getData();

        //#pragma GCC ivdep
        for(uint64_t i = 0; i < tensorOperand->tensor_.size(); ++i){

            if constexpr (std::is_same_v<A, B>){
                resultTensorData[i] = operation(operand1.tensor_[i], operand2.tensor_[i]);
            }else if constexpr (std::is_same_v<A, T>){
                resultTensorData[i] = operation(operand1, operand2.tensor_[i]);
            }else if constexpr (std::is_same_v<B, T>){
                resultTensorData[i] = operation(operand1.tensor_[i], operand2);
            }
        }

        return resultTensor;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <apply_callable<T> C>
    void Tensor<T, DataMB, MetadataMB>::apply(const Tensor<T>& tensor2, C&& operation){

        Tensor<T, DataMB, MetadataMB>::apply(*this, tensor2, std::forward<C>(operation));
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <apply_callable<T> C>
    /*static*/ void Tensor<T, DataMB, MetadataMB>::apply(Tensor<T>& operand1, const Tensor<T>& operand2, C&& operation){

        for(uint64_t i = 0; i < operand1.tensor_.size(); ++i){
            operation(operand1.tensor_[i], operand2.tensor_[i]);
        }
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <apply_callable<T> C>
    /*static*/ void Tensor<T, DataMB, MetadataMB>::apply(Tensor<T>& operand1, const T& operand2, C&& operation){
        
        for(uint64_t i = 0; i < operand1.tensor_.size(); ++i){
            operation(operand1.tensor_[i], operand2);
        }
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <apply_reverse_callable<T> C>
    /*static*/ void Tensor<T, DataMB, MetadataMB>::apply(const T& operand1, Tensor<T>& operand2, C&& operation){
        
        for(uint64_t i = 0; i < operand2.tensor_.size(); ++i){
            operation(operand1, operand2.tensor_[i]);
        }
    }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // template <typename A, typename B, apply_callable<T> C> 
    // void Tensor<T, DataMB, MetadataMB>::apply(A& operand1, B& operand2, C&& operation) // static
    // requires(tensor_or_t_or_bothtensor<A, B, T>){

    //     const Tensor<T>* tensorOperand = type_pick<Tensor<T>>(operand1, operand2);

    //     //#pragma GCC ivdep
    //     for(uint64_t i = 0; i < tensorOperand->tensor_.size(); ++i){

    //         if constexpr (std::is_same_v<A, B>){
    //             operation(operand1.tensor_[i], operand2.tensor_[i]);
    //         }else if constexpr (std::is_same_v<A, T>){
    //             operation(operand1, operand2.tensor_[i]);
    //         }else if constexpr (std::is_same_v<B, T>){
    //             operation(operand1.tensor_[i], operand2);
    //         }
    //     }
        
    // }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <foreach_and_return_callable<T> C>
    auto Tensor<T, DataMB, MetadataMB>::forEachAndReturn(C&& operation) const {

        return Tensor<T, DataMB, MetadataMB>::forEachAndReturn(*this, std::forward<C>(operation));
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <foreach_and_return_callable<T> C>
    /*static*/ auto Tensor<T, DataMB, MetadataMB>::forEachAndReturn(const Tensor<T>& tensor, C&& operation)
    {
        using opReturnType = decltype(operation(std::declval<T>()));
        Tensor<opReturnType> resultTensor = Tensor<opReturnType>(tensor.getDimensionSizes()); 

        opReturnType* resultTensorData = resultTensor.getData();

        //#pragma GCC ivdep
        for(uint64_t i = 0; i < tensor.tensor_.size(); ++i){
            resultTensorData[i] = operation(tensor.tensor_[i]);
        }

        return resultTensor;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <foreach_callable<T> C>
    void Tensor<T, DataMB, MetadataMB>::forEach(C&& operation){

        Tensor<T, DataMB, MetadataMB>::forEach(*this, std::forward<C>(operation));
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    template <foreach_callable<T> C>
    /*static*/ void Tensor<T, DataMB, MetadataMB>::forEach(Tensor<T>& tensor, C&& operation){

        //#pragma GCC ivdep
        for(uint64_t i = 0; i < tensor.tensor_.size(); ++i){

            operation(tensor.tensor_[i]);
        }
        
        // 2.
        //std::transform(tensor.tensor_.begin(), tensor.tensor_.end(), tensor.tensor_.begin(), apply);
    }

    
    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    bool Tensor<T, DataMB, MetadataMB>::incrementCoords(std::span<uint64_t> coordinates, std::span<const uint64_t> dimensionSizes){

        uint64_t lastCoordIndex = coordinates.size() - 1;

        uint64_t coordToIncrement = lastCoordIndex;
        while((dimensionSizes[coordToIncrement] <= (coordinates[coordToIncrement] + 1))){
            coordinates[coordToIncrement--] = 0;
            //coordToIncrement--;

            if(coordToIncrement > lastCoordIndex){
                //std::fill(coordinates.begin(), coordinates.end(), 0);
                return true;
            }
        }

        coordinates[coordToIncrement++]++;

        // while(coordToIncrement <= lastCoordIndex){
        //     coordinates[coordToIncrement++] = 0;
        // }

        return false;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    uint64_t Tensor<T, DataMB, MetadataMB>::updateInnerState(){
        return updateDimensionJump();
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    Tensor<T, DataMB, MetadataMB>::~Tensor()
    {
        //
    }



    // PRIVATE METHODS: -------------------------------------------------------------------------------------------------------

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    LinearContainer<uint64_t, MetadataMB> Tensor<T, DataMB, MetadataMB>::getCoords(uint64_t itemIndex) const{

        LinearContainer<uint64_t, MetadataMB> coordinates(dimensionSizes_.size());

        // uint64_t divisor = tensor_.size();
        
        // for(uint64_t i = 0; i < dimensionSizes_.size(); ++i){

        //     divisor /= dimensionSizes_[i];
        //     coordinates[i] = itemIndex / divisor;
        //     itemIndex %= divisor;
        // }

        //uint64_t remaining = itemIndex;

        for (size_t i = 0; i < dimensionSizes_.size(); ++i) {
            uint64_t coord = itemIndex / dimensionJumps_[i];
            coordinates[i] = coord;
            itemIndex -= coord * dimensionJumps_[i];
        }

        return coordinates;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    /*static*/ void Tensor<T, DataMB, MetadataMB>::getCoords(
        uint64_t itemIndex, 
        span_view<uint64_t> dimensionSizes, 
        uint64_t* coordsBuffer
    ){

        // uint64_t dimensionJump = 1;
        // for(const uint64_t dimensionSize : dimensionSizes){
        //     dimensionJump *= dimensionSize;
        // }

        // for (size_t i = 0; i < dimensionSizes.size(); ++i) {
        //     dimensionJump /= dimensionSizes[i];
        //     uint64_t coord = itemIndex / dimensionJump;
        //     coordsBuffer[i] = coord;
        //     itemIndex -= coord * dimensionJump;
        // }

        

        const uint64_t dimensionCount = dimensionSizes.size();

        for(size_t i = dimensionCount; i-- > 0;){
            coordsBuffer[i] = itemIndex % dimensionSizes[i];
            itemIndex /= dimensionSizes[i];
        }
    }

    // template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    // uint64_t Tensor<T, DataMB, MetadataMB>::getIndex(std::initializer_list<uint64_t> coordinates) const {
    //     return getIndex(std::span<const uint64_t>{coordinates.begin(), coordinates.size()});
    // }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    uint64_t Tensor<T, DataMB, MetadataMB>::getIndex(span_view<uint64_t> coordinates) const {
        
    //     uint64_t itemIndex = 0;
    //     //uint64_t dimensionProduct = 1;

    //    for(uint64_t i = dimensionSizes_.size() - 1; i < dimensionSizes_.size(); --i){

    //         itemIndex += coordinates[i] * dimensionJumps_[i];
    //         //dimensionProduct *= dimensionSizes_[i];
    //    }

        return Tensor<T, DataMB, MetadataMB>::getIndex(coordinates, dimensionSizes_);
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    /*static*/ uint64_t Tensor<T, DataMB, MetadataMB>::getIndex(
        span_view<uint64_t> coordinates, 
        span_view<uint64_t> dimensionSizes
    ){

        uint64_t itemIndex = 0;
        const uint64_t dimensionCount = dimensionSizes.size();
        for (size_t i = 0; i < dimensionCount; ++i){
            itemIndex = itemIndex * dimensionSizes[i] + coordinates[i];
        }

        return itemIndex;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    LinearContainer<T> Tensor<T, DataMB, MetadataMB>::transposition_(const int dim1, const int dim2) const {
        return LinearContainer<T>();
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    LinearContainer<uint64_t> Tensor<T, DataMB, MetadataMB>::littleGetCoords(int itemIndex) const{

        LinearContainer<uint64_t> coordinates;
        coordinates.resize(dimensionSizes_.size());
        uint64_t divisor = tensor_.size();
        
        for(uint64_t i = dimensionSizes_.size() - 1; i < dimensionSizes_.size(); --i){

            divisor /= dimensionSizes_[i];
            coordinates[i] = itemIndex / divisor;
            itemIndex %= divisor;
        }

        return coordinates;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    int Tensor<T, DataMB, MetadataMB>::littleGetIndex(const LinearContainer<uint64_t>& coordinates) const{
        
        int itemIndex = 0;
        int dimensionProduct = 1;

       for(uint64_t i = 0; i < dimensionSizes_.size(); ++i){

            itemIndex += coordinates[i] * dimensionProduct;
            dimensionProduct *= dimensionSizes_[i];
       }

        return itemIndex;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    uint64_t Tensor<T, DataMB, MetadataMB>::updateNumberOfItems(){

        const uint64_t itemCount = 
            std::accumulate(dimensionSizes_.begin(), dimensionSizes_.end(), uint64_t{1}, std::multiplies<uint64_t>());
        tensor_.resize(itemCount);

        return itemCount;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    uint64_t Tensor<T, DataMB, MetadataMB>::updateDimensionJump(){

        const uint64_t dimensionCount = dimensionSizes_.size();
        if(dimensionJumps_.size() != dimensionCount){
            dimensionJumps_.resize(dimensionCount);
        }

        uint64_t jump = 1;
        for(uint64_t i = dimensionCount; i-- > 0;){
            dimensionJumps_[i] = jump;
            jump *= dimensionSizes_[i];
        }

        return jump;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::update(){
        tensor_.resize(updateDimensionJump());
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    inline bool Tensor<T, DataMB, MetadataMB>::compareItems(const T& a, const T& b) const requires(!std::is_floating_point<T>::value){
        
        return a == b;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    inline bool Tensor<T, DataMB, MetadataMB>::compareItems(const T a, const T b) const requires(std::is_floating_point<T>::value){
        
        T veightedEpsilon = std::numeric_limits<T>::epsilon();
        return std::fabs(a - b) < veightedEpsilon;
    }

    template <class T, MemoryBackendConcept<T> DataMB, MemoryBackendConcept<uint64_t> MetadataMB>
    void Tensor<T, DataMB, MetadataMB>::defaultFunctions(){

        //equals_ = &defaultEquals_;

        /*if constexpr (std::is_floating_point<T>::value){

            equals_ = [](const T a, const T b){
                T epsilon = std::numeric_limits<T>::epsilon();
                return std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)));
                
                // this does not work: std::fabs(a - b) < (epsilon * std::max(std::fabs(a), std::fabs(b)));
                // but this does: std::fabs(a - b) < epsilon;
                // luckily this works: std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)));
            };

        }else{

            equals_ = [](const T& a, const T& b){
                return a == b;
            };
        }*/
    }
}
