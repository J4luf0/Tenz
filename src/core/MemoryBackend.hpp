#ifndef MEMORY_BACKEND_HPP
#define MEMORY_BACKEND_HPP

#include <cstddef>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <type_traits>

#include "MemoryBackendConcept.hpp"

namespace gema {

template<class T, size_t Alignment = 64>
class MemoryBackend {

    public:

    template<typename U>
    using type = MemoryBackend<U>;
    using value_type = T;

    MemoryBackend();

    MemoryBackend(const MemoryBackend<T, Alignment>& memoryBackend);
    template <typename U>
    MemoryBackend(const MemoryBackend<U, Alignment>& memoryBackend) requires (!std::is_same_v<U, T>);
    MemoryBackend(MemoryBackend<T, Alignment>&& memoryBackend) noexcept;
    MemoryBackend<T, Alignment>& operator=(const MemoryBackend<T, Alignment>& memoryBackend);
    MemoryBackend<T, Alignment>& operator=(MemoryBackend<T, Alignment>&& memoryBackend) noexcept;

    T* allocate(size_t n) const;
    void deallocate(T* pos, size_t n) const;

    void construct_at(T* pos, const T& value) const;
    void destroy_at(T* pos) const;
    void destroy(T* first, T* last) const;

    T* uninitialized_copy(const T* first, const T* last, T* dest) const;
    //iterator uninitialized_copy(const_iterator first, const_iterator last, iterator dest);

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
    // MemoryBackend<U> copy_with_type() const;

    void copy_to_host(T* dest, const T* src, size_t count) const;
    void copy_from_host(T* dest, const T* src, size_t count) const;

    // Methods out of concept

    template <MemoryBackendConcept<T> DestBackend, MemoryBackendConcept<T> SrcBackend>
    static void copy_to_backend(
        T* dest, const DestBackend& destBackend, const T* src, const SrcBackend& srcBackend, const uint64_t n
    );
};

}

#include "MemoryBackend.tpp"

#endif