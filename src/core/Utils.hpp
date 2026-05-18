#ifndef UTILS_HPP
#define UTILS_HPP

#include <concepts>
#include <cstdint>
#include <span>
#include <cmath>
#include <compare>

namespace gema{

template<typename T, std::size_t Extent = std::dynamic_extent>
class span_view : public std::span<const T, Extent> {
    using Base = std::span<const T, Extent>;

public:
    using Base::Base;

    constexpr span_view(std::initializer_list<T> il)
        requires (Extent == std::dynamic_extent)
        : Base(il.begin(), il.size())
    {}

    constexpr operator std::span<const T, Extent>() const noexcept {
        return static_cast<Base>(*this);
    }
};


template<typename T>
struct DefaultEquals {

    bool operator()(const T& a, const T& b) const {
        if constexpr(std::is_floating_point_v<T>) {
            T epsilon = std::numeric_limits<T>::epsilon();
            return std::fabs(a - b) <= epsilon * std::max(std::fabs(a), std::fabs(b));
        } else {
            return a == b;
        }
    }
};


template<typename T>
struct DefaultOrder {

    std::partial_ordering operator()(const T& a, const T& b) const {

        if constexpr(std::is_floating_point_v<T>) {
            T epsilon = std::numeric_limits<T>::epsilon();
            if (std::fabs(a - b) <= (epsilon * std::max(std::fabs(a), std::fabs(b)))){
                return std::partial_ordering::equivalent;
            }else if(a > b){
                return std::partial_ordering::greater;
            }else{
                return std::partial_ordering::less;
            }

            //return (a > b) ? 1 : -1;

        } else if constexpr(std::three_way_comparable<T>){
            return a <=> b;
            //auto cmp = (a <=> b);
            //return (cmp > 0) - (cmp < 0);
        }else if constexpr(requires {
                { a == b } -> std::convertible_to<bool>;
                { a < b } -> std::convertible_to<bool>;
            }
        ){
            if(a == b) return std::partial_ordering::equivalent;
            return (a < b) ? std::partial_ordering::less : std::partial_ordering::greater;
        }else{
            return std::partial_ordering::unordered;
        }
        
        
        // else if constexpr(std::is_integral_v<T>) {
        //     return (a != b) * ((a > b) - (a < b));
        // } else {
        //     return a <=> b;
        // }
    }
};


template <typename A, typename B, class T, class Type>
concept a_or_b_or_both_a = 
    (std::is_same_v<std::remove_cvref_t<A>, T> && std::is_same_v<std::remove_cvref_t<B>, Type>) ||
    (std::is_same_v<std::remove_cvref_t<B>, T> && std::is_same_v<std::remove_cvref_t<A>, Type>) ||
    (std::is_same_v<std::remove_cvref_t<A>, std::remove_cvref_t<B>> && std::is_same_v<std::remove_cvref_t<A>, Type>);

}

#endif

