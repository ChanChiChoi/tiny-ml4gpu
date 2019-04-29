/*
 * the file borrowed from pybind11
 * */

#pragma once

#if !defined(NAMESPACE_BEGIN)
#    define NAMESPACE_BEGIN(name) namespace name {
#endif

#if !defined(NAMESPACE_END)
#    define NAMESPACE_END(name) }
# endif

#include <vector>

enum class Dtype{INT,FLOAT};


//================below borrowed from pybind11/detail/common.h=====
template <typename...> struct void_t_impl { using type = void; };
template <typename... Ts> using void_t = typename void_t_impl<Ts...>::type;

template <bool B, typename T = void> using enable_if_t = typename std::enable_if<B, T>::type;

// Check if T looks like an input iterator
template <typename T, typename = void> struct is_input_iterator : std::false_type {};
template <typename T>
struct is_input_iterator<T, void_t<decltype(*std::declval<T &>()), decltype(++std::declval<T &>())>>
    : std::true_type {};

template <typename T>
class any_container {
    std::vector<T> v;
public:
    any_container() = default;

    // Can construct from a pair of iterators
    template <typename It, typename = enable_if_t<is_input_iterator<It>::value>>
    any_container(It first, It last) : v(first, last) { }

    // Implicit conversion constructor from any arbitrary container type with values convertible to T
    template <typename Container, typename = enable_if_t<std::is_convertible<decltype(*std::begin(std::declval<const Container &>())), T>::value>>
    any_container(const Container &c) : any_container(std::begin(c), std::end(c)) { }

    // initializer_list's aren't deducible, so don't get matched by the above template; we need this
    // to explicitly allow implicit conversion from one:
    template <typename TIn, typename = enable_if_t<std::is_convertible<TIn, T>::value>>
    any_container(const std::initializer_list<TIn> &c) : any_container(c.begin(), c.end()) { }

    // Avoid copying if given an rvalue vector of the correct type.
    any_container(std::vector<T> &&v) : v(std::move(v)) { }

    // Moves the vector out of an rvalue any_container
    operator std::vector<T> &&() && { return std::move(v); }

    // Dereferencing obtains a reference to the underlying vector
    std::vector<T> &operator*() { return v; }
    const std::vector<T> &operator*() const { return v; }

    // -> lets you call methods on the underlying vector
    std::vector<T> *operator->() { return &v; }
    const std::vector<T> *operator->() const { return &v; }
};
//======================



