#include "NTree.hpp"

namespace gema{

    
    template <class T, TensorConcept TensorContainer>
    NTree<T, TensorContainer>::NTree(
        const LinearContainer<uint64_t>& newTensorDimensionSizes, 
        const LinearContainer<LinearContainer<uint64_t>>& newNTreeRestraints
    ) : tensor_(newTensorDimensionSizes), dimensionRestraints_(newNTreeRestraints){

    }

    
    template <class T, TensorConcept TensorContainer>
    uint64_t NTree<T, TensorContainer>::getNumberOfItems() const{
        //return tensor_.;
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
    void NTree<T, TensorContainer>::updateVirtualMask()
    {

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
}