#pragma once

#include <savepoint/base.hpp>
#include <savepoint/fwd.hpp>
#include <savepoint/log.hpp>
#include <savepoint/traits.hpp>
#include <savepoint/version.hpp>

#include <algorithm>
#include <type_traits>
#include <ranges>
#include <iterator>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <memory>
#include <span>
#include <utility>
#include <vector>

/**
 * @brief Implementation of the Visitor pattern for serialization.
 * 
 * The visitor is used to serialize objects. It uses a simplified version of the
 * [pattern](https://refactoring.guru/design-patterns/visitor) with two operating modes:
 * 1. Reading from the Savepoint.
 * 2. Writing to the Savepoint.
 * 
 * Visitors are a structured blob of binary data. They consist of:
 * 1. A SavepointVersion representing the user's application build version.
 * 2. A SavepointVersion representing the Savepoint build version (reserved for future use).
 * 3. The object data.
 * 
 * When writing, we store the current build versions. When reading, we load the
 * versions used in the previous write. By comparing these versions to the build
 * versions, we can determine what members are safe to deserialize.
 * 
 * @snippet examples/nested_types.cpp nested_types
 */
class SavepointVisitor
{
private:
    static constexpr size_t kHeader = sizeof(SavepointVersion) * 2;

public:
    /**
     * @brief Serialize to/from an items's raw bytes.
     * 
     * Perform a simple memcpy from the visitor to the item's raw bytes (and
     * vice versa) using the size of the item. If the item cannot be deserialized,
     * it will be default initialized using args, assuming args are provided.
     * 
     * @tparam T The type to serialize.
     * @tparam Args The types of the args.
     * @param item The item to serialize.
     * @param version The version required to deserialize.
     * @param args The args for default initialization.
     */
    template<SavepointMemcpyable T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        // For detecting bugs in MSVC concepts
        static_assert(!SavepointPointer<T>);
        static_assert(!std::is_base_of_v<SavepointBase, T>);
        static_assert(!std::ranges::range<T>);
        if (IsReading())
        {
            // Required for write-only containers (e.g. views)
            if constexpr (std::is_const_v<T>)
            {
                SavepointLog("Tried to read into a const");
                return;
            }
            else
            {
                if (Version < version)
                {
                    if constexpr (sizeof...(Args) > 0)
                    {
                        item = T{std::forward<Args>(args)...};
                    }
                    return;
                }
                if (sizeof(T) > GetSize())
                {
                    SavepointLog(std::format("Tried to read past visitor: {} -> {}", Version.GetString(), version.GetString()));
                    if constexpr (sizeof...(Args) > 0)
                    {
                        item = T{std::forward<Args>(args)...};
                    }
                    return;
                }
                std::memcpy(std::addressof(item), Reader.data() + Offset, sizeof(T));
                Offset += sizeof(T);
            }
        }
        else
        {
            Writer.resize(Writer.size() + sizeof(T));
            std::memcpy(Writer.data() + Writer.size() - sizeof(T), std::addressof(item), sizeof(T));
        }
    }

    /**
     * @brief Visit using the implementation from SavepointVisit.
     *
     * If the item cannot be deserialized, it will be default initialized using
     * args, assuming args are provided.
     * 
     * @tparam T The type to serialize.
     * @tparam Args The types of the args.
     * @param item The item to serialize.
     * @param version The version required to deserialize.
     * @param args The args for default initialization.
     */
    template<SavepointFreeVisit T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        if (IsReading())
        {
            if (Version < version || GetSize() == 0)
            {
                if constexpr (sizeof...(Args) > 0)
                {
                    item = T{std::forward<Args>(args)...};
                }
                return;
            }
        }
        SavepointVisit(*this, item);
    }

    /**
     * @brief Visit using the implementation from T::Visit.
     * 
     * If the item cannot be deserialized, it will be default initialized using
     * args, assuming args are provided.
     * 
     * @tparam T The type to serialize.
     * @tparam Args The types of the args.
     * @param item The item to serialize.
     * @param version The version required to deserialize.
     * @param args The args for default initialization.
     */
    template<SavepointMemberVisit T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        if (IsReading())
        {
            if (Version < version || GetSize() == 0)
            {
                if constexpr (sizeof...(Args) > 0)
                {
                    item = T{std::forward<Args>(args)...};
                }
                return;
            }
        }
        item.Visit(*this);
    }

    /**
     * @brief Skip bytes.
     * 
     * @tparam T The type to skip.
     */
    template<SavepointMemcpyable T>
    void Skip()
    {
        if (IsReading())
        {
            if (sizeof(T) > GetSize())
            {
                SavepointLog(std::format("Tried to skip past visitor: {}", Version.GetString()));
                return;
            }
            Offset += sizeof(T);
        }
        else
        {
            Writer.resize(Writer.size() + sizeof(T));
        }
    }

    /**
     * @brief Disable deserialization.
     */
    void Fail()
    {
        if (!IsReading())
        {
            SavepointLog("Tried to fail while writing");
            return;
        }
        Offset = Reader.size();
    }

    /**
     * @brief Check if a visitor is reading.
     * 
     * @return True if the visitor is reading.
     */
    bool IsReading() const
    {
        return !Reader.empty();
    }

    /**
     * @brief Check if a visitor is writing.
     * 
     * @return True if the visitor is writing.
     */
    bool IsWriting() const
    {
        return !IsReading();
    }

    /**
     * @brief Get the application version.
     * 
     * @return The application version.
     */
    SavepointVersion GetVersion() const
    {
        return Version;
    }

    /**
     * @brief Get the number of bytes to write or the remaining to read.
     * 
     * @return The number of bytes.
     */
    size_t GetSize() const
    {
        if (IsReading())
        {
            return Reader.size() - std::min(Offset, Reader.size());
        }
        else
        {
            return Writer.size();
        }
    }

    /** @cond INTERNAL */

    void BeginReading(const void* data, size_t size)
    {
        void* pointer = const_cast<void*>(data);
        Reader = {static_cast<uint8_t*>(pointer), size};
        Offset = 0;
        operator()(Version);
        Skip<SavepointVersion>();
    }

    void BeginWriting(SavepointVersion version)
    {
        Reader = {};
        Writer.resize(kHeader);
        std::memcpy(Writer.data(), &version, sizeof(SavepointVersion));
        std::memcpy(Writer.data() + sizeof(SavepointVersion), &kSavepointVersion, sizeof(SavepointVersion));
        Version = version;
    }

    const void* GetData() const
    {
        return Writer.data();
    }

    /** @endcond */

private:
    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    size_t Offset;
};

/**
 * @brief Visit implementation for serializing std::unique_ptr and std::shared_ptr.
 * 
 * Pointers data and whether they are null are serialized. As such, pointers are
 * allowed to be nullptr and will be handled accordingly. Polymorphics are also
 * supported by storing type information alongside the aforementioned data. When
 * reading, the correct derived type will be instantiated and deserialized.
 * 
 * Raw pointers are unsupported, not because they couldn't be, but because it's
 * not needed and avoids potential pitfalls.
 * 
 * @tparam T The type of the pointer.
 * @param visitor The visitor.
 * @param item The pointer.
 * @see SavepointBase
 * @see SAVEPOINT_DERIVED
 */
template<SavepointStdPointer T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    using ValueT = typename T::element_type;
    if (visitor.IsReading())
    {
        bool hasPointer = false;
        visitor(hasPointer);
        if (!hasPointer)
        {
            if (item)
            {
                SavepointLog("Nulled an allocated pointer since visitor contained a nullptr");
                item.reset();
            }
            return;
        }
        if (!item)
        {
            if constexpr (std::is_base_of_v<SavepointBase, ValueT>)
            {
                item.reset(dynamic_cast<ValueT*>(SavepointReadDerived(visitor)));
                return;
            }
            else if constexpr (std::is_default_constructible_v<ValueT>)
            {
                item.reset(new ValueT());
                if (!item)
                {
                    SavepointLog("Failed to allocate pointer");
                    visitor.Fail();
                    return;
                }
            }
            else
            {
                // Don't static_assert because it'll fail on already instantiated
                // derived classes with abstract parents
                SavepointLog("No method to create pointer");
                visitor.Fail();
                return;
            }
        }
        visitor(*item);
    }
    else
    {
        bool hasPointer = item.get() != nullptr;
        visitor(hasPointer);
        if (hasPointer)
        {
            if constexpr (std::is_base_of_v<SavepointBase, ValueT>)
            {
                SavepointWriteDerived(item.get(), visitor);
            }
            else
            {
                visitor(*item);
            }
        }
    }
}

/**
 * @brief Visit implementation for serializing an std::pair.
 * 
 * @tparam T The type of the pair.
 * @param visitor The visitor.
 * @param item The pair.
 */
template<SavepointPair T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    // Required because maps use const for value_type::first_type
    using First = std::remove_const_t<typename T::first_type>;
    First& first = const_cast<First&>(item.first);
    visitor(first);
    visitor(item.second);
}

/**
 * @brief Visit implementation for serializing containers.
 * 
 * @tparam T The type of the container.
 * @param visitor The visitor.
 * @param item The pointer.
 */
template<std::ranges::range T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    using ValueT = typename T::value_type;
    size_t size = item.size();
    if constexpr (SavepointDynamicRange<T>)
    {
        if (visitor.IsReading() && size)
        {
            SavepointLog("Tried to read into non-empty range");
            item.clear();
        }
    }
    visitor(size);
    if (visitor.IsReading())
    {
        // Can detect when we read garbage and would iterate forever
        if (size > visitor.GetSize())
        {
            SavepointLog("Tried to read past visitor");
            visitor.Fail();
            return;
        }
        if constexpr (SavepointDynamicRange<T>)
        {
            auto inserter = std::inserter(item, std::ranges::end(item));
            for (size_t i = 0; i < size; i++)
            {
                // TODO: mutable iterators
                ValueT element;
                visitor(element);
                *inserter++ = element;
            }
        }
        else if constexpr (SavepointStaticRange<T>)
        {
            size_t maxSize = std::ranges::size(item);
            if (size > maxSize)
            {
                SavepointLog(std::format("Fixed range is too small: {} < {}", maxSize, size));
            }
            if (size < maxSize)
            {
                SavepointLog(std::format("Fixed range will be truncated: {} < {}", maxSize, size));
            }
            maxSize = std::min(size, maxSize);
            for (size_t i = 0; i < maxSize; i++)
            {
                visitor(item[i]);
            }
            for (; maxSize < size; maxSize++)
            {
                visitor.Skip<ValueT>();
            }
        }
        else
        {
            // Required for write-only containers (e.g. views)
            SavepointLog("Unknown range");
        }
    }
    else
    {
        for (auto& element : item)
        {
            visitor(element);
        }
    }
}
