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

class SavepointVisitor
{
private:
    static constexpr size_t kHeader = sizeof(SavepointVersion) * 2;

public:
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

    template<SavepointMemcpyable T>
    void operator()(T* data, size_t maxSize, size_t size)
    {
        if (IsReading())
        {
            // Required for write-only containers (e.g. views)
            if constexpr (std::is_const_v<T>)
            {
                SavepointLog("Tried to read into a const pointer");
                return;
            }
            else
            {
                if (maxSize < size)
                {
                    SavepointLog(std::format("Truncating buffer: {}, {} -> {}", Version.GetString(), size, maxSize));
                    size = maxSize;
                }
                if (size > GetSize())
                {
                    SavepointLog(std::format("Tried to read past visitor: {}", Version.GetString()));
                    return;
                }
                std::memcpy(data, Reader.data() + Offset, size);
                Offset += size;
            }
        }
        else
        {
            if (maxSize != size)
            {
                SavepointLog(std::format("Sizes don't match: {}, {} != {}", Version.GetString(), maxSize, size));
                size = maxSize;
            }
            Writer.resize(Writer.size() + size);
            std::memcpy(Writer.data() + Writer.size() - size, data, size);
        }
    }

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
            // We don't return since we can use it to completely skip bad visitors
            SavepointLog(std::format("Tried to skip past visitor: {}", Version.GetString()));
        }
        Offset += sizeof(T) * size;
    }

    void Reset(void* data, size_t size)
    {
        Reader = {static_cast<uint8_t*>(data), size};
        Offset = 0;
        operator()(Version);
        // Reserved for future use
        Skip<SavepointVersion>();
    }

    void Reset(SavepointVersion application, SavepointVersion savepoint)
    {
        Reader = {};
        Writer.resize(kHeader);
        std::memcpy(Writer.data(), &application, sizeof(SavepointVersion));
        // Reserved for future use
        std::memcpy(Writer.data() + sizeof(SavepointVersion), &savepoint, sizeof(SavepointVersion));
    }

    bool IsReading() const
    {
        return !Reader.empty();
    }

    bool IsWriting() const
    {
        return !IsReading();
    }

    bool IsEmpty() const
    {
        return Writer.size() == kHeader;
    }

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

template<SavepointPair T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    // Required because maps use const for value_type::first_type
    using First = std::remove_const_t<typename T::first_type>;
    First& first = const_cast<First&>(item.first);
    visitor(first);
    visitor(item.second);
}

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
            maxSize = std::min(size, maxSize);
            for (size_t i = 0; i < maxSize; i++)
            {
                E element;
                visitor(element);
                item[i] = element;
            }
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
