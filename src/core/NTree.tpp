#include "NTree.hpp"

namespace gema{

    
    template <class T, TensorConcept TensorContainer>
    NTree<T, TensorContainer>::NTree(
        const LinearContainer<uint64_t>& newTensorDimensionSizes, 
        const LinearContainer<LinearContainer<uint64_t>>& newNTreeRestraints
    ) : tensor_(newTensorDimensionSizes), dimensionRestraints_(newNTreeRestraints){
        updateVirtualMask();
    }

    template <class T, TensorConcept TensorContainer>
    NTree<T, TensorContainer>::NTree(const NTree<T>& otherNTree)
    :   tensor_(otherNTree.tensor_), 
        dimensionRestraints_(otherNTree.dimensionRestraints_), 
        indexMask_(otherNTree.indexMask_){

    }

    template <class T, TensorConcept TensorContainer>
    inline NTree<T, TensorContainer>::NTree(NTree<T>&& otherNTree) noexcept
    :   tensor_(std::move(otherNTree.tensor_)), 
        dimensionRestraints_(std::move(otherNTree.dimensionRestraints_)), 
        indexMask_(std::move(otherNTree.indexMask_)){

    }

    template <class T, TensorConcept TensorContainer>
    NTree<T, TensorContainer>::NTree(){

    }

    template <class T, TensorConcept TensorContainer>
    uint64_t NTree<T, TensorContainer>::getNumberOfItems() const{

        uint64_t itemCounting = 1;
        for(const uint64_t dimensionSize : tensor_.getDimensionSizes()){
            itemCounting *= dimensionSize;
        }

        return itemCounting;
    }

    template <class T, TensorConcept TensorContainer>
    T NTree<T, TensorContainer>::getItem(span_view<uint64_t> coordinates){
        return tensor_.getItem(coordinates);
    }

    template <class T, TensorConcept TensorContainer>
    void NTree<T, TensorContainer>::setItem(const T& value, span_view<uint64_t> coordinates){
        tensor_.setItem(value, coordinates);
    }

    template <class T, TensorConcept TensorContainer>
    T* NTree<T, TensorContainer>::getData(){
        return tensor_.getData();
    }

    template <class T, TensorConcept TensorContainer>
    void NTree<T, TensorContainer>::fillWith(const T& fill){
        tensor_.fill(fill);
    }

    template <class T, TensorConcept TensorContainer>
    bool NTree<T, TensorContainer>::isValidCoordinates(std::span<const uint64_t> coords) const {

        if (coords.size() > tensor_.getNumberOfDimensions()) return false;
        if (coords.size() == 0) return true;

        LinearContainer<uint64_t> coordRestraints = restrainsFromCoords(coords);

        for (uint64_t i = 0; i < coords.size(); i++){
            if (coords[i] >= coordRestraints[i]) return false;
        }

        return true;
    }

    template <class T, TensorConcept TensorContainer>
    bool NTree<T, TensorContainer>::isValidCoordinatesFast(std::span<const uint64_t> coords) const{

        if (coords.size() != tensor_.getNumberOfDimensions()) return false;

        uint64_t index = tensor_.getIndex(coords);
        return indexMask_[index];
    }

    template <class T, TensorConcept TensorContainer>
    std::array<uint64_t, 2> NTree<T, TensorContainer>::getLocalRange(std::span<const uint64_t> coords) const{

        LinearContainer<uint64_t> lowestStructureFirst{lowestStructureFirst};
        // lowestStructureFirst.fill(0);
        // Array.Copy(coords, lowestStructureFirst, coords.size());

        std::array<uint64_t, 2> newRange;
        newRange[0] = tensor_.getIndex(lowestStructureFirst);
        newRange[1] = autoGetLength(coords);

        return newRange;
    }

    template <class T, TensorConcept TensorContainer>
    uint64_t NTree<T, TensorContainer>::autoGetLength(std::span<const uint64_t> coords) const{

        std::span<const uint64_t> dimensionSizes = tensor_.getDimensionSizes();
        uint64_t dimensionCount = tensor_.getNumberOfDimensions();

        uint64_t numOfMissingCoordinates = dimensionCount - coords.size();
        uint64_t numberOfItems = 1;

        for(uint64_t i = 0; i < numOfMissingCoordinates; i++){
            numberOfItems *= dimensionSizes[dimensionCount - 1 - i];
        }

        return numberOfItems;
    }

    template <class T, TensorConcept TensorContainer>
    void NTree<T, TensorContainer>::updateVirtualMask(){

        indexMask_.resize(tensor_.getNumberOfItems());

        LinearContainer<uint64_t> currentCoords(tensor_.getNumberOfDimensions());
        currentCoords.fill(0ul);

        for(uint64_t i = 0; i < indexMask_.size(); i++){

            indexMask_[i] = isValidCoordinates(currentCoords);

            Tensor<T>::incrementCoords(currentCoords, tensor_.getDimensionSizes());
        }
    }

    template <class T, TensorConcept TensorContainer>
    LinearContainer<uint64_t> NTree<T, TensorContainer>::restrainsFromCoords(std::span<const uint64_t> coords) const{
        
        uint64_t restraintIndex = 0;
        uint64_t restraint;
        LinearContainer<uint64_t> restraints{};

        for (uint64_t i = 0; i < coords.size(); i++){
            
            // This is ugly, fix later
            if (restraintIndex >= dimensionRestraints_[i].size()){
                // LinearContainer<uint64_t> bad{coords.size()};
                // bad.fill(-1);
                LinearContainer<uint64_t> bad{0};
                return bad;
            }

            restraint = dimensionRestraints_[i][restraintIndex];
            restraints.push_back(restraint);

            uint64_t sumOfPreceding = 0;
            for(uint64_t j = 0; j < restraintIndex; j++){
                sumOfPreceding += dimensionRestraints_[i][j];
            }
            restraintIndex = coords[i] + sumOfPreceding;
        }

        return restraints;
    }

    template <class T, TensorConcept TensorContainer>
    bool NTree<T, TensorContainer>::isRestraintsValid(
        const LinearContainer<LinearContainer<uint64_t>>& restraints, 
        std::span<const uint64_t> dimensionSizes
    ){

        if (restraints.size() != dimensionSizes.size()) return false;
        if (restraints[0].size() != 1) return false;

        uint64_t prevRestraintSum = restraints[0][0];

        for (uint64_t i = 1; i < restraints.size(); ++i){

            std::span<const uint64_t> restraint = restraints[i];
            if (restraint.size() != prevRestraintSum) return false;

            prevRestraintSum = 0;
            for (uint64_t j = 0; j < restraint.size(); ++j){
                if (restraint[j] < 1 || restraint[j] > dimensionSizes[i]) return false;
                prevRestraintSum += restraint[j];
            }
        }

        return true;
    }
}