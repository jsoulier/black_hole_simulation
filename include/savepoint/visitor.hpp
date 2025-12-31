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

#include <savepoint/base.hpp>
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

/**
 * @brief 
 * 
 */
class SavepointVisitor
{
private:
    static constexpr size_t kHeader = sizeof(SavepointVersion) * 2;

public:
    /**
     * @brief 
     * 
     * @tparam T 
     * @tparam Args 
     * @param item 
     * @param version 
     * @param args 
     */
    template<SavepointMemcpyable T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        // For detecting bugs in MSVC concepts
        static_assert(!SavepointRange<T>);
        static_assert(!SavepointPointer<T>);
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
     * @brief 
     * 
     * @tparam T 
     * @tparam Args 
     * @param item 
     * @param version 
     * @param args 
     */
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

    /**
     * @brief 
     * 
     * @tparam T 
     * @tparam Args 
     * @param item 
     * @param version 
     * @param args 
     */
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

    /**
     * @brief 
     * 
     * @tparam T 
     * @param size 
     */
    template<SavepointMemcpyable T>
    void Skip(size_t size = 1)
    {
        if (IsWriting())
        {
            SavepointLog("Tried to skip on a writer");
            return;
        }
        if (!size)
        {
            SavepointLog("Tried to skip nothing");
            return;
        }
        if (sizeof(T) * size > GetSize())
        {
            SavepointLog(std::format("Tried to skip past visitor: {}", Version.GetString()));
            return;
        }
        Offset += sizeof(T) * size;
    }

    /**
     * @brief 
     * 
     */
    void Fail()
    {
        Offset = Reader.size();
    }

    /**
     * @brief 
     * 
     * @return 
     */
    bool IsReading() const
    {
        return !Reader.empty();
    }

    /**
     * @brief 
     * 
     * @return 
     */
    bool IsWriting() const
    {
        return !IsReading();
    }

    /**
     * @brief 
     * 
     * @return 
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

    void Reset(SavepointVersion application, SavepointVersion savepoint)
    {
        Reader = {};
        Writer.resize(kHeader);
        std::memcpy(Writer.data(), &application, sizeof(SavepointVersion));
        // Reserved for future use
        std::memcpy(Writer.data() + sizeof(SavepointVersion), &savepoint, sizeof(SavepointVersion));
    }

    void Reset(void* data, size_t size)
    {
        Reader = {static_cast<uint8_t*>(data), size};
        Offset = 0;
        operator()(Version);
        // Reserved for future use
        Skip<SavepointVersion>();
    }

    bool IsEmpty() const
    {
        return Writer.size() == kHeader;
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
 * @brief 
 * 
 * @tparam T 
 * @param visitor 
 * @param item 
 */
template<SavepointStdPointer T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    using E = typename T::element_type;
    if (visitor.IsReading())
    {
        bool hasPointer = false;
        visitor(hasPointer);
        if (!hasPointer)
        {
            return;
        }
        if (!item)
        {
            if constexpr (std::is_base_of_v<SavepointBase, E>)
            {
                item.reset(dynamic_cast<E*>(SavepointReadDerived(visitor)));
                return;
            }
            else if constexpr (std::is_default_constructible_v<E>)
            {
                item.reset(new E());
                if (!item)
                {
                    SavepointLog("Failed to allocate pointer");
                    visitor.Fail();
                    return;
                }
            }
            else
            {
                // Don't static_assert because it'll fail on already instanciated
                // derived classes with abstract parents
                SavepointLog("No method to create pointer");
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
            if constexpr (std::is_base_of_v<SavepointBase, E>)
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
 * @brief 
 * 
 * @tparam T 
 * @param visitor 
 * @param item 
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
 * @brief 
 * 
 * @tparam T 
 * @param visitor 
 * @param item 
 */
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
