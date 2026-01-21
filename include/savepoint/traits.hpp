#pragma once

#include <savepoint/fwd.hpp>

#include <memory>
#include <ranges>
#include <utility>
#include <type_traits>

/** @cond INTERNAL */

template<typename T>
struct SavepointIsUniquePointerImpl : std::false_type {};

template<typename T, typename Deleter>
struct SavepointIsUniquePointerImpl<std::unique_ptr<T, Deleter>> : std::true_type {};

template<typename T>
concept SavepointIsUniquePointer = SavepointIsUniquePointerImpl<T>::value;

template<typename T>
struct SavepointIsSharedPointerImpl : std::false_type {};

template<typename T>
struct SavepointIsSharedPointerImpl<std::shared_ptr<T>> : std::true_type {};

template<typename T>
concept SavepointIsSharedPointer = SavepointIsSharedPointerImpl<T>::value;

template<typename T>
concept SavepointIsStdPointer = SavepointIsUniquePointer<T> || SavepointIsSharedPointer<T>;

template<typename T>
concept SavepointIsPointer = std::is_pointer_v<T> || SavepointIsStdPointer<T>;

template<typename T, typename Base>
concept SavepointIsPointerBaseOf = SavepointIsStdPointer<T> && std::is_base_of_v<Base, typename T::element_type>;

template<typename T>
struct SavepointIsPairImpl : std::false_type {};

template<typename First, typename Second>
struct SavepointIsPairImpl<std::pair<First, Second>> : std::true_type {};

template<typename T>
concept SavepointIsPair = SavepointIsPairImpl<T>::value;

template<typename T>
concept SavepointIsDynamicRange = requires(T item) { item.insert(std::ranges::end(item), std::declval<typename T::value_type>()); };

template<typename T>
concept SavepointIsStaticRange = !SavepointIsDynamicRange<T> && requires(T item) { item[0] = std::declval<typename T::value_type>(); };

template<typename T>
concept SavepointHasFreeVisit = requires(SavepointVisitor visitor, T item) { { Visit(visitor, item) }; };

template<typename T>
concept SavepointHasMemberVisit = requires(SavepointVisitor visitor, T item) { { item.Visit(visitor) }; };

template<typename T>
concept SavepointCanMemcpy = !SavepointIsPointer<T> && !SavepointHasFreeVisit<T> && !SavepointHasMemberVisit<T> && std::is_trivially_copyable_v<T>;

template<typename T>
concept SavepointCanVisit = !std::is_same_v<T, SavepointVisitor> && !std::is_base_of_v<SavepointBase, T>;

template<typename T>
struct SavepointIsEntityImpl : std::false_type {};

template<typename T>
concept SavepointIsEntity = std::is_base_of_v<SavepointEntity, T> || SavepointIsPointerBaseOf<T, SavepointEntity>;

/** @endcond */
