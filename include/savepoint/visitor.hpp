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
#include <savepoint/log.hpp>
#include <savepoint/traits.hpp>
#include <savepoint/version.hpp>

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

/*
 * Modified implementation of the visitor pattern for serializing/deserializing objects
 *
 * Each visitor consists of:
 * 1. The savepoint version (unused, reserved for future use)
 * 2. The application version (used for versioning user data)
 * 3. The data (binary blob of user data)
 * 
 * Visitors can be in 2 modes:
 * 1. Writing (setting the current versions and data)
 * 2. Reading (carefully checking versions and data)
 *
 * For non-trivial objects that use pointers or require versioning, users
 * should implement 1 of the following functions:
 * 
 * struct Entity
 * {
 *     int X;
 *     int Y;
 *
 *     // Option 1
 *     void Visit(SavepointVisitor& visitor)
 *     {
 *         visitor(X);
 *         visitor(Y);
 *     }
 * };
 * 
 * // Option 2
 * void SavepointVisit(SavepointVisitor& visitor, Entity& entity)
 * {
 *     visitor(entity.X);
 *     visitor(entity.Y);
 * }
 * 
 * For more information, check the examples
 */
class SavepointVisitor
{
private:
    // Header consists of savepoint and application version
    static constexpr size_t kHeader = sizeof(SavepointVersion) * 2;

public:
    // Simplest method for visiting. Copies the object directly to/from the blob
    template<SavepointMemcpyable T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
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

    // Visit using the free visit function
    template<SavepointFreeVisit T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        if (IsReading())
        {
            if (Version < version)
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

    // Visit using the member visit function
    template<SavepointMemberVisit T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        if (IsReading())
        {
            if (Version < version)
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

    // Skip sizeof(T) * size bytes on the reader
    template<SavepointMemcpyable T>
    void Skip(size_t size = 1)
    {
        if (IsWriting())
        {
            SavepointLog("Tried to skip on a writer");
            return;
        }
        if (sizeof(T) * size > GetSize())
        {
            SavepointLog(std::format("Tried to skip past visitor: {}", Version.GetString()));
            // We don't return since we can use it to completely skip bad visitors
            // return;
        }
        Offset += sizeof(T) * size;
    }

    // TODO: hide from user
    void Reset(SavepointVersion application, SavepointVersion savepoint)
    {
        Reader = {};
        Writer.resize(kHeader);
        std::memcpy(Writer.data(), &application, sizeof(SavepointVersion));
        // Reserved for future use
        std::memcpy(Writer.data() + sizeof(SavepointVersion), &savepoint, sizeof(SavepointVersion));
    }

    // TODO: hide from user
    void Reset(void* data, size_t size)
    {
        Reader = {static_cast<uint8_t*>(data), size};
        Offset = 0;
        operator()(Version);
        // Reserved for future use
        Skip<SavepointVersion>();
    }

    bool IsReading() const
    {
        return !Reader.empty();
    }

    bool IsWriting() const
    {
        return !IsReading();
    }

    // TODO: hide from user
    bool IsEmpty() const
    {
        return Writer.size() == kHeader;
    }

    // TODO: hide from user
    const void* GetData() const
    {
        return Writer.data();
    }

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

private:
    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    size_t Offset;
};

// Visit implementation for pointers (unique and shared only)
template<SavepointStdPointer T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    using E = typename T::element_type;
    if (visitor.IsReading())
    {
        if (!item)
        {
            if constexpr (SavepointUniquePointer<T>)
            {
                item = std::make_unique<E>();
            }
            else if constexpr (SavepointSharedPointer<T>)
            {
                item = std::make_shared<E>();
            }
            else
            {
                static_assert(false, "Unknown pointer");
            }
        }
        visitor(*item);
    }
    else
    {
        if (item)
        {
            visitor(*item);
        }
        else
        {
            SavepointLog("Tried to write null pointer");
        }
    }
}

// TODO: convert for tuple
template<SavepointPair T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    // Required because maps use const for value_type::first_type
    using First = std::remove_const_t<typename T::first_type>;
    First& first = const_cast<First&>(item.first);
    visitor(first);
    visitor(item.second);
}

// Visit implementation for all iterable containers
template<SavepointRange T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    using E = typename T::value_type;
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
        // Inaccurate but can detect when we read garbage and would iterate forever
        if (size > visitor.GetSize())
        {
            SavepointLog("Tried to read past visitor");
            visitor.Skip<uint8_t>(size);
            return;
        }
        if constexpr (SavepointDynamicRange<T>)
        {
            auto inserter = std::inserter(item, std::ranges::end(item));
            for (size_t i = 0; i < size; i++)
            {
                // TODO: mutable iterators
                E element;
                visitor(element);
                *inserter++ = element;
            }
        }
        else if constexpr (SavepointStaticRange<T>)
        {
            size_t maxSize = std::ranges::size(item);
            if (size > maxSize)
            {
                SavepointLog(std::format("Fixed range is too small: %d < %d", maxSize, size));
            }
            if (size < maxSize)
            {
                SavepointLog(std::format("Fixed range will be truncated: %d < %d", maxSize, size));
            }
            maxSize = std::min(size, maxSize);
            for (size_t i = 0; i < maxSize; i++)
            {
                visitor(item[i]);
            }
            // Skip excess data when truncated
            for (; maxSize < size; maxSize++)
            {
                visitor.Skip<E>();
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
