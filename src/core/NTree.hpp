#ifndef NTREE_HPP
#define NTREE_HPP

#include "LinearContainer.hpp"
#include "Tensor.hpp"

namespace gema{

template<class T, TensorConcept TensorContainer = Tensor<T>>
class NTree{

    public:

    using DataContainer = TensorContainer::DataContainer;
    using MetadataContainer = TensorContainer::MetadataContainer;
    using NTreeMetadataContainer = LinearContainer<NTreeMetadataContainer>;

    private:

    TensorContainer tensor_;

    LinearContainer<LinearContainer<uint64_t>> dimensionRestraints_;

    LinearContainer<bool> indexMask_;



    public:

    NTree(
        const LinearContainer<uint64_t>& newTensorDimensionSizes,  
        const LinearContainer<LinearContainer<uint64_t>>& newNTreeRestraints
    );

    NTree(const NTree<T>& otherNTree);

    NTree(NTree<T>&& otherNTree) noexcept;

    NTree();



    uint64_t getNumberOfItems() const;

    T getItem(span_view<uint64_t> coordinates);

    void setItem(const T& value, span_view<uint64_t> coordinates);

    T* getData();

    NTree& setData(const LinearContainer<T>& tensorItems);

    void fillWith(const T& fill);

    template <foreach_callable<T> C> 
    void forEach(C&& operation);

    template <foreach_callable<T> C>
    static void forEach(NTree<T>& tensor, C&& operation);

    bool isValidCoordinates(std::span<const uint64_t> coords) const;

    std::array<uint64_t, 2> getLocalRange(std::span<const uint64_t> coords) const;

    private:

    void updateVirtualMask();

    LinearContainer<uint64_t> restrainsFromCoords(std::span<const uint64_t> coords) const;


};

}

#include "NTree.tpp"

#endif