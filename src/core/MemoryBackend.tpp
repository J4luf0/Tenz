#include <cstring>
#include <iterator>

#include "Utils.hpp"
#include "MemoryBackend.hpp"

//#define T_ALLOC_ALIGN class T, sycl::usm::alloc MemoryType, size_t Alignment

namespace gema {

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>::MemoryBackend(){

    }

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>::MemoryBackend(const MemoryBackend<T, Alignment>& memoryBackend){

    }

    template <class T, size_t Alignment>
    template <typename U>
    MemoryBackend<T, Alignment>::MemoryBackend(const MemoryBackend<U, Alignment>& memoryBackend)
    requires (!std::is_same_v<U, T>){

    }

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>::MemoryBackend(MemoryBackend<T, Alignment>&& memoryBackend) noexcept {

    }

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>&
    MemoryBackend<T, Alignment>::operator=(const MemoryBackend<T, Alignment>& memoryBackend) {
        return *this;
    }

    template <class T, size_t Alignment>
    MemoryBackend<T, Alignment>& 
    MemoryBackend<T, Alignment>::operator=(MemoryBackend<T, Alignment>&& memoryBackend) noexcept {
        return *this;
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::allocate(size_t n) const {

        if(n == 0) return nullptr;

        size_t bytes = n * sizeof(T);

        void* ptr = ::operator new(bytes, std::align_val_t{Alignment});
        return static_cast<T*>(ptr);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::deallocate(T* pos, size_t n) const {

        ::operator delete(pos, std::align_val_t{Alignment});
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::construct_at(T* pos, const T& value) const {
        std::construct_at(pos, value);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::destroy_at(T* pos) const {
        std::destroy_at(pos);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::destroy(T* first, T* last) const {
        std::destroy(first, last);
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::uninitialized_copy(const T* first, const T* last, T* dest) const {
        return std::uninitialized_copy(first, last, dest);
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::uninitialized_move(T* first, T* last, T* dest) const {
        return std::uninitialized_move(first, last, dest);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::uninitialized_default_construct(T* first, T* last) const {
        std::uninitialized_default_construct(first, last);
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::uninitialized_fill_n(T* dest, size_t count, const T& value) const {
        return std::uninitialized_fill_n(dest, count, value);
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::copy(T* dest, const T* src, size_t count) const {

        if constexpr(std::is_trivially_copyable_v<T>){
            std::memcpy(dest, src, count * sizeof(T));
        }else{
            for(size_t i = 0; i < count; ++i){
                dest[i] = src[i];
            }
        }
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::memory_set(T* dest, size_t ch, size_t count) const {

        static_assert(std::is_trivially_copyable_v<T>);
        
        return static_cast<T*>(std::memset(dest, ch, count * sizeof(T)));
    }

    template <class T, size_t Alignment>
    T* MemoryBackend<T, Alignment>::fill(T* dest, const T& value, size_t count) const{

        for(size_t i = 0; i < count; ++i){
            dest[i] = value;
        }

        return dest;
    }

    template <class T, size_t Alignment>
    bool MemoryBackend<T, Alignment>::equals(const T *a, const T *b, size_t count) const{

        if constexpr(std::is_trivially_copyable_v<T> && std::has_unique_object_representations_v<T>){
            return std::memcmp(a, b, count * sizeof(T)) == 0;
        }else{

            DefaultEquals<T> equals;

            for(uint64_t i = 0; i < count; i++){

                bool result = equals(a[i], b[i]);
                if(result != true){
                    return result;
                }
            }
        }

        return true;
    }

    template <class T, size_t Alignment>
    std::partial_ordering MemoryBackend<T, Alignment>::compare(const T* a, const T* b, size_t count) const {

        if constexpr(std::is_trivially_copyable_v<T> && std::has_unique_object_representations_v<T>){

            int result = std::memcmp(a, b, count * sizeof(T));

            if(result == 0){
                return std::partial_ordering::equivalent;
            }else if(result > 0){
                return std::partial_ordering::greater;
            }else{
                return std::partial_ordering::less;
            }
            //return std::memcmp(a, b, count * sizeof(T));
        }else{

            DefaultOrder<T> order;

            for(uint64_t i = 0; i < count; i++){

                std::partial_ordering result = order(a[i], b[i]);
                if(result != std::partial_ordering::equivalent){
                    return result;
                }
            }
        }

        return std::partial_ordering::equivalent;
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::set_value(T* dest, const uint64_t index, const T& value) const{
        dest[index] = value;
    }

    template <class T, size_t Alignment>
    T MemoryBackend<T, Alignment>::get_value(const T* dest, const uint64_t index) const{
        return dest[index];
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::copy_to_host(T* dest, const T* src, size_t count) const {
        //std::memcpy(dest, src, count);

        if constexpr(std::is_trivially_copyable_v<T>) {

            std::memcpy(dest, src, count * sizeof(T));

        } else {

            for(size_t i = 0; i < count; ++i){
                dest[i] = src[i];
            }
        }
    }

    template <class T, size_t Alignment>
    void MemoryBackend<T, Alignment>::copy_from_host(T* dest, const T* src, size_t count) const {
        //std::memcpy(dest, src, count);

        if constexpr(std::is_trivially_copyable_v<T>) {

            std::memcpy(dest, src, count * sizeof(T));

        } else {

            for(size_t i = 0; i < count; ++i){
                dest[i] = src[i];
            }
        }
    }

    // template <class T, size_t Alignment>
    // template <typename U>
    // MemoryBackend<U> MemoryBackend<T, Alignment>::copy_with_type() const{
    //     return MemoryBackend<U>();
    // }

    template <class T, size_t Alignment>
    template <MemoryBackendConcept<T> DestBackend, MemoryBackendConcept<T> SrcBackend>
    void MemoryBackend<T, Alignment>::copy_to_backend(
        T* dest, const DestBackend& destBackend, const T* src, const SrcBackend& srcBackend, const uint64_t n)
    {

        if constexpr(std::is_same_v<DestBackend, MemoryBackend<T>>){
            srcBackend.copy_to_host(dest, src, n);
        }else if constexpr(std::is_same_v<SrcBackend, MemoryBackend<T>>){
            destBackend.copy_from_host(dest, src, n);
        }else{
            MemoryBackend<T> hostBackend;
            T* host = hostBackend.allocate(n);
            srcBackend.copy_to_host(host, src, n);
            destBackend.copy_from_host(dest, host, n);
            hostBackend.deallocate(host, n);
        }
    }
}

#undef T_ALLOC_ALIGN