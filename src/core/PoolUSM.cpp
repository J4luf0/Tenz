#include "PoolUSM.hpp"

namespace gema {

    void PoolUSM::freePool(){

        std::lock_guard lock(poolMutex_);

        for(const auto& keyValue : memoryPool_){
            for(const PoolBlock& block : keyValue.second){
                sycl::free(block.ptr, *(keyValue.first.queue));
            }
        }
        memoryPool_.clear();
    }

    void* PoolUSM::allocate(size_t bytes, sycl::queue* queue, sycl::usm::alloc kind, size_t alignment){

        if(bytes == 0) return nullptr;

        PoolKey key{bytes, alignment, kind, queue};

        {
            std::lock_guard lock(poolMutex_);

            auto it = memoryPool_.find(key);

            // if(it->second.empty()){
            //     memoryPool_.erase(it);
            // }

            if(it != memoryPool_.end() && !it->second.empty()){

                void* ptr = it->second.back().ptr;
                it->second.pop_back();
                PoolUSM::poolHit_.fetch_add(1, std::memory_order_relaxed);
                return ptr;
            }
        }
                
        PoolUSM::poolMiss_.fetch_add(1, std::memory_order_relaxed);

        return sycl::aligned_alloc(alignment, bytes, *queue, kind);
    }

    void PoolUSM::deallocate(void* pos, size_t bytes, sycl::queue* queue, sycl::usm::alloc kind, size_t alignment){

        if(pos == nullptr) return;

        PoolKey key{bytes, alignment, kind, queue};

        std::lock_guard lock(poolMutex_);

        memoryPool_[key].push_back(PoolBlock{pos});
    }

}