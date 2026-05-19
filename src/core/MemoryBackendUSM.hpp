#ifndef MEMORY_BACKEND_USM_HPP
#define MEMORY_BACKEND_USM_HPP

#include <cstddef>
#include <unordered_map>

#include <sycl/sycl.hpp>

#include "MemoryBackend.hpp"

namespace gema {

template<class T, sycl::usm::alloc Kind, size_t Alignment = 64>
class MemoryBackendUSM : public MemoryBackend<T, Alignment> {

    private:

    struct PoolKey {
        size_t bytes;
        size_t alignment;
        sycl::usm::alloc kind;
        sycl::context context;

        bool operator==(const PoolKey& other) const {
            return bytes == other.bytes && alignment == other.alignment && kind == other.kind && context == other.context;
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
            size_t h4 = std::hash<sycl::context>{}(k.context);

            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
        }
    };

    static inline std::unordered_map<PoolKey, std::vector<PoolBlock>, PoolKeyHash> memoryPool_;
    static inline std::mutex poolMutex_;
    static inline std::atomic<size_t> instanceCount_ = 0;

    static void freePool();

    public:

    sycl::queue* queue_ = nullptr;

    template<typename U>
    using type = MemoryBackendUSM<U, Kind>;
    using value_type = T;

    MemoryBackendUSM(sycl::queue* queue_);
    MemoryBackendUSM(const MemoryBackendUSM<T, Kind, Alignment>& memoryBackend);
    template <typename U> 
    MemoryBackendUSM(const MemoryBackendUSM<U, Kind, Alignment>& memoryBackend) requires (!std::is_same_v<U, T>);
    MemoryBackendUSM(MemoryBackendUSM<T, Kind, Alignment>&& memoryBackend) noexcept;
    //MemoryBackendUSM() = delete;
    MemoryBackendUSM();
    ~MemoryBackendUSM();

    MemoryBackendUSM<T, Kind, Alignment>& operator=(const MemoryBackendUSM<T, Kind, Alignment>& memoryBackend);
    MemoryBackendUSM<T, Kind, Alignment>& operator=(MemoryBackendUSM<T, Kind, Alignment>&& memoryBackend) noexcept;

    T* allocate(size_t n) const;
    void deallocate(T* pos, size_t n) const;

    void construct_at(T* pos, const T& value) const;
    void destroy_at(T* pos) const;
    void destroy(T* first, T* last) const;

    T* uninitialized_copy(const T* first, const T* last, T* dest) const;
    T* uninitialized_move(T* first, T* last, T* dest) const;
    void uninitialized_default_construct(T* first, T* last) const;
    T* uninitialized_fill_n(T* dest, size_t count, const T& value) const;

    void copy(T* dest, const T* src, size_t count) const;
    T* memory_set(T* dest, size_t ch, size_t count ) const;
    T* fill(T* dest, const T& value, size_t count ) const;
    bool equals(const T* a, const T* b, size_t count) const;
    std::partial_ordering compare(const T* a, const T* b, size_t count) const;

    void set_value(T* dest, const uint64_t index, const T& value) const;
    T get_value(const T* dest, const uint64_t index) const;

    // template <typename U>
    // MemoryBackendUSM<U, Kind, Alignment> copy_with_type() const;

    void copy_to_host(T* dest, const T* src, size_t count) const;
    void copy_from_host(T* dest, const T* src, size_t count) const;
};

}

#include "MemoryBackendUSM.tpp"

#endif