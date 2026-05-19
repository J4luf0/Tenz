#ifndef NTREE_HPP
#define NTREE_HPP

#include "LinearContainer.hpp"
#include "Tensor.hpp"

namespace gema{

template<class T>
class NTree{

    private:

    Tensor<T> tensor_;

    std::vector<uint64_t> dimensionRestraints_;

    std::vector<bool> indexMask_;



    public:

    NTree(const std::vector<uint64_t>& newTensorDimensionSizes,  const std::vector<uint64_t>& newNTreeRestraints);

    NTree(const NTree<T>& otherNTree);

    NTree(NTree&& otherNTree);

    NTree();



    uint64_t getNumberOfItems() const;

    T& getItem(const std::vector<uint64_t>& coordinates);

    void setItem(const T& value, const std::vector<uint64_t>& coordinates);

    LinearContainer<T>& getData();

    Tensor<T>& setData(const LinearContainer<T>& tensorItems);

    void fillWith(const T& fill);

    template <foreach_callable<T> C> 
    inline void forEach(C&& operation);

    template <foreach_callable<T> C>
    static void forEach(Tensor<T>& tensor, C&& operation);

    bool CoordsValid(const std::vector<uint64_t>& coords) const;

    std::array<uint64_t, 2> getLocalRange(const std::vector<uint64_t>& coords) const;

    private:






};

}

#include "NTree.tpp"

#endif