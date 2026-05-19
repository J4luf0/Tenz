#include <compare>
#include <type_traits>
#include <sycl/sycl.hpp>

#include "Utils.hpp"
#include "PoolUSM.hpp"
#include "MemoryBackendUSM.hpp"

//#define T_ALLOC_ALIGN class T, sycl::usm::alloc MemoryType, std::size_t Alignment

namespace gema {

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>::MemoryBackendUSM(sycl::queue* queue_){
        this->queue_ = queue_;
        //MemoryBackendUSM<T, Kind, Alignment>::instanceCount_.fetch_add(1, std::memory_order_relaxed);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>::MemoryBackendUSM(const MemoryBackendUSM<T, Kind, Alignment>& otherBackend)
    : queue_(otherBackend.queue_){
        //MemoryBackendUSM<T, Kind, Alignment>::instanceCount_.fetch_add(1, std::memory_order_relaxed);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    template <typename U>
    MemoryBackendUSM<T, Kind, Alignment>::MemoryBackendUSM(const MemoryBackendUSM<U, Kind, Alignment>& otherBackend)
    requires (!std::is_same_v<U, T>){
        queue_ = otherBackend.queue_;
        //MemoryBackendUSM<T, Kind, Alignment>::instanceCount_.fetch_add(1, std::memory_order_relaxed);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>::MemoryBackendUSM(MemoryBackendUSM<T, Kind, Alignment>&& otherBackend) noexcept
    : queue_(otherBackend.queue_){
        //otherBackend.queue_ = nullptr;
        //MemoryBackendUSM<T, Kind, Alignment>::instanceCount_.fetch_add(1, std::memory_order_relaxed);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>::MemoryBackendUSM(){
        //MemoryBackendUSM<T, Kind, Alignment>::instanceCount_.fetch_add(1, std::memory_order_relaxed);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>::~MemoryBackendUSM(){

        // if(MemoryBackendUSM<T, Kind, Alignment>::instanceCount_.fetch_sub(1, std::memory_order_acq_rel) == 1){
        //     MemoryBackendUSM<T, Kind, Alignment>::freePool();
        // }
    }

    // template <class T, sycl::usm::alloc Kind, size_t Alignment>
    // void MemoryBackendUSM<T, Kind, Alignment>::freePool(){
    //     for(const auto& keyValue : memoryPool_){
    //         for(const PoolBlock& block : keyValue.second){
    //             sycl::free(block.ptr, *(keyValue.first.queue));
    //         }
    //     }
    //     memoryPool_.clear();
        
    //     // std::cout << "pool hit: " << MemoryBackendUSM<T, Kind, Alignment>::poolHit_.load() << std::endl
    //     //           << "pool miss: " << MemoryBackendUSM<T, Kind, Alignment>::poolMiss_.load() << std::endl;

    // }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>& MemoryBackendUSM<T, Kind, Alignment>::operator=(
    const MemoryBackendUSM<T, Kind, Alignment>& otherBackend) {
        queue_ = otherBackend.queue_;
        return *this;
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    MemoryBackendUSM<T, Kind, Alignment>& MemoryBackendUSM<T, Kind, Alignment>::operator=(
    MemoryBackendUSM<T, Kind, Alignment>&& otherBackend) noexcept {
        queue_ = std::move(otherBackend.queue_);
        return *this;
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::allocate(size_t n) const {

        // if(n == 0) return nullptr;

        // //std::size_t bytes = n * sizeof(T);

        // return sycl::aligned_alloc<T>(Alignment, n, *queue_, Kind);



        // if(n == 0) return nullptr;

        // PoolKey key{n * sizeof(T), Alignment, Kind, queue_};

        // {
        //     std::lock_guard lock(poolMutex_);

        //     auto it = memoryPool_.find(key);

        //     if(it != memoryPool_.end() && !it->second.empty()){

        //         void* ptr = it->second.back().ptr;
        //         it->second.pop_back();
        //         MemoryBackendUSM<T, Kind, Alignment>::poolHit_.fetch_add(1, std::memory_order_relaxed);
        //         return static_cast<T*>(ptr);
        //     }
        // }
                
        // MemoryBackendUSM<T, Kind, Alignment>::poolMiss_.fetch_add(1, std::memory_order_relaxed);

        // return sycl::aligned_alloc<T>(Alignment, n, *queue_, Kind);

        return static_cast<T*>(PoolUSM::allocate(n * sizeof(T), queue_, Kind, Alignment));
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::deallocate(T* pos, size_t n) const {

        //sycl::free(pos, *queue_);



        // if(pos == nullptr) return;

        // PoolKey key{n * sizeof(T), Alignment, Kind, queue_};

        // std::lock_guard lock(poolMutex_);

        // memoryPool_[key].push_back(PoolBlock{static_cast<void*>(pos)});

        PoolUSM::deallocate(static_cast<void*>(pos), n * sizeof(T), queue_, Kind, Alignment);
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::construct_at(T* pos, const T& value) const {

        // queue_->submit([&](sycl::handler& h){
        //     h.single_task([=](){
        //         new (pos) T(value);
        //     });
        // }).wait();

        if constexpr (Kind == sycl::usm::alloc::device) {
            queue_->submit([&](sycl::handler& h){
                h.single_task([=](){
                    new (pos) T(value);
                });
            }).wait();
        } else {
            std::construct_at(pos, value);
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::destroy_at(T* pos) const {

        // queue_->submit([&](sycl::handler& h){
        //     h.single_task([=](){
        //         pos->~T();
        //     });
        // }).wait();

        if constexpr(std::is_trivially_destructible_v<T>) return;

        if constexpr (Kind == sycl::usm::alloc::device) {
            queue_->submit([&](sycl::handler& h){
                h.single_task([=](){
                    pos->~T();
                });
            }).wait();
        } else {
            std::destroy_at(pos);
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::destroy(T* first, T* last) const {

        // size_t n = last - first;
        // queue_->parallel_for(n, [=](auto i){
        //     (first + i)->~T();
        // }).wait();

        if constexpr(std::is_trivially_destructible_v<T>) return;

        if constexpr (Kind == sycl::usm::alloc::device) {
            size_t n = last - first;
            queue_->parallel_for(n, [=](auto i){
                (first + i)->~T();
            }).wait();
        } else {
            std::destroy(first, last);
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::uninitialized_copy(const T* first, const T* last, T* dest) const {

        size_t n = last - first;

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->memcpy(dest, first, n * sizeof(T)).wait();

        } else {

            if constexpr(std::is_trivially_copyable_v<T>) {
                std::memcpy(dest, first, n * sizeof(T));
            } else {
                std::uninitialized_copy(first, last, dest);
            }
        }

        return dest + n;
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::uninitialized_move(T* first, T* last, T* dest) const {

        size_t n = last - first;

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->memcpy(dest, first, n * sizeof(T)).wait();

        } else {

            if constexpr(std::is_trivially_copyable_v<T>) {
                std::memcpy(dest, first, n * sizeof(T));
            } else {
                std::uninitialized_move(first, last, dest);
            }
        }

        return dest + n;
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::uninitialized_default_construct(T* first, T* last) const {

        if constexpr (Kind == sycl::usm::alloc::device) {

            size_t n = last - first;
            queue_->parallel_for(n, [=](auto i){
                new (first + i) T();
            }).wait();

        } else {

            std::uninitialized_default_construct(first, last);
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::uninitialized_fill_n(T* dest, size_t count, const T& value) const {

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->parallel_for(count, [=](auto i){
                new (dest + i) T(value);
            }).wait();

        } else {

            std::uninitialized_fill_n(dest, count, value);
        }

        return dest + count;
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::copy(T* dest, const T* src, size_t count) const {

        //queue_->memcpy(dest, src, count * sizeof(T)).wait();

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->memcpy(dest, src, count * sizeof(T)).wait();

        } else {

            if constexpr(std::is_trivially_copyable_v<T>) {
                std::memcpy(dest, src, count * sizeof(T));
            } else {
                for(size_t i = 0; i < count; ++i){
                    dest[i] = src[i];
                }
            }
        }
    }
    
    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::memory_set(T* dest, size_t ch, size_t count) const {

        //queue_->memset(dest, ch, count * sizeof(T)).wait();

        if constexpr (Kind == sycl::usm::alloc::device) {

            queue_->memset(dest, ch, count * sizeof(T)).wait();

        } else {

            std::memset(dest, ch, count * sizeof(T));
        }

        return dest;
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T* MemoryBackendUSM<T, Kind, Alignment>::fill(T* dest, const T& value, size_t count) const{
        return queue_->fill<T>(dest, value, count).wait();
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    bool MemoryBackendUSM<T, Kind, Alignment>::equals(const T *a, const T *b, size_t count) const{

        // Older but simpler/more reliable implementation
        // bool* workingResult = sycl::malloc_shared<bool>(1, *queue_);
        // *workingResult = true;

        // queue_->submit([&](sycl::handler& h){
        //     h.single_task([=](){
                
        //         DefaultEquals<T> equals;
        //         for(uint64_t i = 0; i < count; i++){

        //             bool result = equals(a[i], b[i]);
        //             if(result != true){
        //                 *workingResult = result;
        //                 return;
        //             }
        //         }
        //     });
        // }).wait();

        // bool finalResult = *workingResult;
        // sycl::free(workingResult, *queue_);

        // return finalResult;

        uint64_t* deviceAcumulator = sycl::malloc_device<uint64_t>(1, *queue_);

        uint64_t initialValue = 0;
        queue_->memcpy(deviceAcumulator, &initialValue, sizeof(uint64_t)).wait();

        queue_->parallel_for(count, [=](sycl::id<1> idx){
            size_t i = idx[0];

            DefaultEquals<T> equals;

            if(!equals(a[i], b[i])){

                sycl::atomic_ref<
                    uint64_t,
                    sycl::memory_order::relaxed,
                    sycl::memory_scope::device,
                    sycl::access::address_space::global_space
                > atomicFlag(*deviceAcumulator);

                atomicFlag.store(1);
            }

            // bool result = equals(a[i], b[i]);
            // if(result != true){
            //     *workingResult += 1;
            // }
        }).wait();

        uint64_t hostAcumulator;
        queue_->memcpy(&hostAcumulator, deviceAcumulator, sizeof(uint64_t)).wait();

        bool finalResult = (hostAcumulator == 0);

        sycl::free(deviceAcumulator, *queue_);

        return finalResult;
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    std::partial_ordering MemoryBackendUSM<T, Kind, Alignment>::compare(const T* a, const T* b, size_t count) const {

        std::partial_ordering* workingResult = sycl::malloc_shared<std::partial_ordering>(1, *queue_);
        *workingResult = std::partial_ordering::equivalent;

        queue_->submit([&](sycl::handler& h){
            h.single_task([=](){
                
                DefaultOrder<T> order;
                for(uint64_t i = 0; i < count; i++){

                    std::partial_ordering result = order(a[i], b[i]);
                    if(result != std::partial_ordering::equivalent){
                        *workingResult = result;
                        return;
                    }
                }
            });
        }).wait();

        std::partial_ordering finalResult = *workingResult;
        sycl::free(workingResult, *queue_);

        return finalResult;



        // if constexpr (Kind == sycl::usm::alloc::device) {

        //     // fallback → copy to host (expensive!)
        //     uint64_t countBytes = count * sizeof(T);
        //     T* tmpA = new T[count];
        //     T* tmpB = new T[count];

        //     queue_->memcpy(tmpA, a, count * sizeof(T)).wait();
        //     queue_->memcpy(tmpB, b, count * sizeof(T)).wait();

        //     int result = std::memcmp(tmpA, tmpB, count * sizeof(T));

        //     delete[] tmpA;
        //     delete[] tmpB;

        //     return result;

        // } else {

        //     if constexpr(std::is_trivially_copyable_v<T>) {
        //         return std::memcmp(a, b, count * sizeof(T));
        //     } else {
        //         for(size_t i = 0; i < count; ++i){
        //             if(a[i] < b[i]) return -1;
        //             if(a[i] > b[i]) return 1;
        //         }
        //         return 0;
        //     }
        // }
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::set_value(T* dest, const uint64_t index, const T& value) const {

        T* placeToSave = dest + index;

        queue_->submit([&](sycl::handler& h){
            h.single_task([=](){
                *placeToSave = value;
            });
        }).wait();
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    T MemoryBackendUSM<T, Kind, Alignment>::get_value(const T* dest, const uint64_t index) const {

        T* sharedTmp = sycl::malloc_shared<T>(1, *queue_);

        queue_->submit([&](sycl::handler& h){
            h.single_task([=](){
                new (sharedTmp) T(dest[index]);
            });
        }).wait();

        T result = *sharedTmp;

        std::destroy_at(sharedTmp);

        sycl::free(sharedTmp, *queue_);

        return result;
    }

    
    // template <class T, sycl::usm::alloc Kind, size_t Alignment>
    // template <typename U>
    // MemoryBackendUSM<U, Kind, Alignment> MemoryBackendUSM<T, Kind, Alignment>::copy_with_type() const{
    //     return MemoryBackendUSM<U, Kind, Alignment>(queue_);
    // }



    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::copy_to_host(T* dest, const T* src, size_t count) const {

        if constexpr(std::is_trivially_copyable_v<T>) {
            queue_->memcpy(dest, src, count * sizeof(T)).wait();
        } else {

            // raw bytes to temporary host storage
            std::unique_ptr<std::byte[]> rawBuffer(new std::byte[count * sizeof(T)]);

            queue_->memcpy(
                rawBuffer.get(),
                src,
                count * sizeof(T)
            ).wait();

            // to host objects
            T* tmp = reinterpret_cast<T*>(rawBuffer.get());

            // safe copy to destination
            for(size_t i = 0; i < count; ++i) {

                if constexpr(std::is_copy_assignable_v<T>) {
                    dest[i] = tmp[i];
                } else {
                    std::construct_at(dest + i, tmp[i]);
                }

            }
        }
    }

    template <class T, sycl::usm::alloc Kind, size_t Alignment>
    void MemoryBackendUSM<T, Kind, Alignment>::copy_from_host(T* dest, const T* src, size_t count) const {
        //queue_->memcpy(dest, src, count * sizeof(T)).wait();

        if constexpr(std::is_trivially_copyable_v<T>) {
            queue_->memcpy(dest, src, count * sizeof(T)).wait();
        } else {

            // temporary shared storage
            T* tmp = sycl::malloc_shared<T>(
                count,
                *queue_
            );

            try {

                // host copy to shared memory
                for(size_t i = 0; i < count; ++i) {
                    std::construct_at(tmp + i, src[i]);
                }

                // device-side assignment / construction
                queue_->submit([&](sycl::handler& h){

                    h.parallel_for(
                        sycl::range<1>(count),
                        [=](sycl::id<1> idx){

                            size_t i = idx[0];

                            if constexpr(std::is_copy_assignable_v<T>) {
                                dest[i] = tmp[i];
                            } else {
                                new (dest + i) T(tmp[i]);
                            }
                        }
                    );

                }).wait();

                // cleanup shared objects
                for(size_t i = 0; i < count; ++i) {
                    std::destroy_at(tmp + i);
                }

                sycl::free(tmp, *queue_);

            } catch(...) {

                for(size_t i = 0; i < count; ++i) {
                    std::destroy_at(tmp + i);
                }

                sycl::free(tmp, *queue_);

                throw;
            }
        }
    }
}

#undef T_ALLOC_ALIGN