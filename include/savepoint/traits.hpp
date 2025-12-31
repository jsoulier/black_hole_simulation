#pragma once

#include <savepoint/fwd.hpp>

#include <memory>
#include <ranges>
#include <utility>
#include <type_traits>

/** @cond INTERNAL */

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
concept SavepointPointer = std::is_pointer_v<T> || SavepointStdPointer<T>;

template<typename T>
struct SavepointPairImpl : std::false_type {};

template<typename First, typename Second>
struct SavepointPairImpl<std::pair<First, Second>> : std::true_type {};

template<typename T>
concept SavepointPair = SavepointPairImpl<T>::value;

template<typename T>
concept SavepointDynamicRange = requires(T item) { item.insert(std::ranges::end(item), std::declval<typename T::value_type>()); };

template<typename T>
concept SavepointStaticRange = !SavepointDynamicRange<T> && requires(T item) { item[0] = std::declval<typename T::value_type>(); };

template<typename T>
concept SavepointFreeVisit = requires(SavepointVisitor visitor, T item) { { SavepointVisit(visitor, item) }; };

template<typename T>
concept SavepointMemberVisit = requires(SavepointVisitor visitor, T item) { { item.Visit(visitor) }; };

template<typename T>
concept SavepointMemcpyable = !SavepointPointer<T> && !SavepointFreeVisit<T> && !SavepointMemberVisit<T>;

template<typename T>
concept SavepointVisitable = !std::is_same_v<T, SavepointVisitor> && !std::is_base_of_v<SavepointBase, T>;

/** @endcond */
