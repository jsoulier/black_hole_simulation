/*
 * This is free and unencumbered software released into the public domain.
 * 
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 * 
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * For more information, please refer to <https://unlicense.org>
 */

#pragma once

#include <savepoint/fwd.hpp>

#include <memory>
#include <ranges>
#include <utility>
#include <type_traits>

template<typename T>
concept SavepointRawPointer = std::is_pointer_v<T>;

template<typename T>
struct SavepointUniquePointerImpl : std::false_type {};

template<typename T, typename Deleter>
struct SavepointUniquePointerImpl<std::unique_ptr<T, Deleter>> : std::true_type {};

template<typename T>
concept SavepointUniquePointer = SavepointUniquePointerImpl<T>::value;

template<typename T>
struct SavepointSharedPointerImpl : std::false_type {};

template<typename T>
struct SavepointSharedPointerImpl<std::shared_ptr<T>> : std::true_type {};

template<typename T>
concept SavepointSharedPointer = SavepointSharedPointerImpl<T>::value;

template<typename T>
concept SavepointStdPointer = SavepointUniquePointer<T> || SavepointSharedPointer<T>;

template<typename T>
concept SavepointPointer = SavepointRawPointer<T> || SavepointStdPointer<T>;

template<typename T>
struct SavepointPairImpl : std::false_type {};

template<typename First, typename Second>
struct SavepointPairImpl<std::pair<First, Second>> : std::true_type {};

template<typename T>
concept SavepointPair = SavepointPairImpl<T>::value;

// Is a resizable container with insert
template<typename T>
concept SavepointDynamicRange = requires(T item)
{
    item.insert(std::ranges::end(item), std::declval<typename T::value_type>());
};

// Is not the former with subscript
template<typename T>
concept SavepointStaticRange = !SavepointDynamicRange<T> && requires(T item)
{
    item[0] = std::declval<typename T::value_type>();
};

template<typename T>
concept SavepointRange = std::ranges::range<T>;

// For ensuring we don't accidentally use the wrong Read or Write on a visitor
template<typename T>
concept SavepointReadableWritable = !std::is_same_v<T, SavepointVisitor> && !std::is_base_of_v<SavepointBase, T>;

template<typename T>
concept SavepointFreeVisit = requires(SavepointVisitor visitor, T item) { { SavepointVisit(visitor, item) }; };

template<typename T>
concept SavepointMemberVisit = requires(SavepointVisitor visitor, T item) { { item.Visit(visitor) }; };

// Not a pointer and no Visit implementation provided
template<typename T>
concept SavepointMemcpyable = !SavepointPointer<T> && !SavepointFreeVisit<T> && !SavepointMemberVisit<T>;
