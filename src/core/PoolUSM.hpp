#ifndef POOL_USM_HPP
#define POOL_USM_HPP

#include <cstddef>
#include <unordered_map>

#include <sycl/sycl.hpp>

namespace gema {

class PoolUSM{

    struct PoolKey {
        size_t bytes;
        size_t alignment;
        sycl::usm::alloc kind;
        sycl::queue* queue;

        bool operator==(const PoolKey& other) const {
            return bytes == other.bytes && alignment == other.alignment && kind == other.kind && queue == other.queue;
        }
    };

    struct PoolBlock {
        void* ptr;
    };

    struct PoolKeyHash {

        size_t operator()(const PoolKey& k) const {

            size_t h1 = std::hash<size_t>{}(k.bytes);
            size_t h2 = std::hash<size_t>{}(k.alignment);
            size_t h3 = std::hash<int>{}(static_cast<int>(k.kind));
            size_t h4 = std::hash<sycl::queue>{}(*(k.queue));

            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };

    struct Cleanup {

        ~Cleanup() {
            freePool();
        }
    };
    static inline Cleanup cleanup_;

    static inline std::unordered_map<PoolKey, std::vector<PoolBlock>, PoolKeyHash> memoryPool_;
    static inline std::mutex poolMutex_;
    static inline std::atomic<size_t> instanceCount_ = 0;

    static void freePool();

    public:

    static inline std::atomic<size_t> poolHit_ = 0;
    static inline std::atomic<size_t> poolMiss_ = 0;
    
    static void* allocate(size_t nBytes, sycl::queue* queue, sycl::usm::alloc kind, size_t alignment);
    static void deallocate(void* pos, size_t nBytes, sycl::queue* queue, sycl::usm::alloc kind, size_t alignment);
    
};

}

#endif