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

#include <savepoint_fwd.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using SavepointLogFunction = std::function<void(const std::string_view& string)>;

void SavepointSetLogFunction(const SavepointLogFunction& function);
void SavepointLog(const std::string_view& string);

class SavepointVersion
{
private:
    friend class Savepoint;

public:
    constexpr SavepointVersion()
        : Value{0}
    {
    }

    constexpr SavepointVersion(uint32_t major, uint32_t minor, uint32_t patch)
        : Value{major << 24 | minor << 16 | patch}
    {
    }

    constexpr uint32_t GetMajor() const
    {
        return (Value >> 24) & 0xFF;
    }

    constexpr uint32_t GetMinor() const
    {
        return (Value >> 16) & 0xFF;
    }

    constexpr uint32_t GetPatch() const
    {
        return Value & 0xFFFF;
    }

    std::string GetString() const
    {
        return std::format("{}.{}.{}", GetMajor(), GetMinor(), GetPatch());
    }

    constexpr bool operator==(const SavepointVersion other) const
    {
        return Value == other.Value;
    }

    constexpr bool operator!=(const SavepointVersion other) const
    {
        return Value != other.Value;
    }

    constexpr bool operator<(const SavepointVersion other) const
    {
        return Value < other.Value;
    }

    constexpr bool operator>(const SavepointVersion other) const
    {
        return Value > other.Value;
    }

    constexpr bool operator<=(const SavepointVersion other) const
    {
        return Value <= other.Value;
    }

    constexpr bool operator>=(const SavepointVersion other) const
    {
        return Value >= other.Value;
    }

private:
    uint32_t Value;
};

class SavepointID
{
private:
    friend class Savepoint;

public:
    constexpr SavepointID()
        : Value{std::numeric_limits<uint32_t>::max()}
    {
    }

    constexpr bool operator==(const SavepointID other) const
    {
        return Value == other.Value;
    }

    constexpr bool operator!=(const SavepointID other) const
    {
        return Value != other.Value;
    }

    constexpr operator bool() const
    {
        return Value != SavepointID{}.Value;
    }

private:
    uint32_t Value;
};

class SavepointBase
{
private:
    friend class Savepoint;

public:
    virtual void Visit(SavepointVisitor& visitor) = 0;

private:
    virtual std::string_view SavepointDerivedGetString() const = 0;
};

using SavepointDerivedFunction = SavepointBase*(*)();

void SavepointAddDerivedFunction(const std::string_view& string, const SavepointDerivedFunction& function);

#define SAVEPOINT_DERIVED(T) \
    private: \
        struct SavepointDerivedFunctionRegistrar \
        { \
            static SavepointBase* Function() \
            { \
                return new T(); \
            } \
            SavepointDerivedFunctionRegistrar() \
            { \
                SavepointAddDerivedFunction(#T, Function); \
            } \
        }; \
        static inline SavepointDerivedFunctionRegistrar SavepointDerivedFunctionRegistrar; \
        std::string_view SavepointDerivedGetString() const override \
        { \
            return #T;\
        } \
    public: \

template<typename T>
struct SavepointRawPointerImpl : std::is_pointer<T> {};

template<typename T>
concept SavepointRawPointer = SavepointRawPointerImpl<T>::value;

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
concept SavepointFreeVisit = requires(SavepointVisitor visitor, T item) { { SavepointVisit(visitor, item) }; };

template<typename T>
concept SavepointMemberVisit = requires(SavepointVisitor visitor, T item) { { item.Visit(visitor) }; };

template<typename T>
concept SavepointPrimitive = !SavepointPointer<T> && !SavepointFreeVisit<T> && !SavepointMemberVisit<T>;

class SavepointVisitor
{
private:
    friend class Savepoint;

    static constexpr size_t kHeader = sizeof(SavepointVersion) * 2;

public:
    template<SavepointPrimitive T, typename... Args>
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

    template<SavepointPrimitive T>
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

    template<typename T>
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

    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    size_t Offset;
};

template<typename T>
struct SavepointPairImpl : std::false_type {};

template<typename First, typename Second>
struct SavepointPairImpl<std::pair<First, Second>> : std::true_type {};

template<typename T>
concept SavepointPair = SavepointPairImpl<T>::value;

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

template<typename T>
concept SavepointDynamicRange = requires(T item)
{
    item.insert(std::ranges::end(item), std::declval<typename T::value_type>());
};

template<typename T>
concept SavepointStaticRange = !SavepointDynamicRange<T> && requires(T item)
{
    item[0] = std::declval<typename T::value_type>();
};

template<typename T>
concept SavepointRange = std::ranges::range<T>;

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

template<typename T> using SavepointReadFunction = std::function<void(T& item)>;
template<typename T> using SavepointReadEntityFunction = std::function<void(T& item, SavepointID id)>;
template<typename T> using SavepointReadTile2DFunction = std::function<void(T& item, int x, int y)>;
template<typename T> using SavepointReadTile3DFunction = std::function<void(T& item, int x, int y, int z)>;
using SavepointReadBaseFunction = std::function<void(SavepointBase* base)>;
using SavepointReadBaseEntityFunction = std::function<void(SavepointBase* base, SavepointID id)>;
using SavepointReadBaseTile2DFunction = std::function<void(SavepointBase* base, int x, int y)>;
using SavepointReadBaseTile3DFunction = std::function<void(SavepointBase* base, int x, int y, int z)>;

template<typename T>
concept SavepointBasicReadWrite = !std::is_same_v<T, SavepointVisitor> && !std::is_base_of_v<SavepointBase, std::decay_t<T>>;

enum class SavepointStatus
{
    Failed,
    Existing,
    New,
};

class Savepoint
{
public:
    Savepoint();
    ~Savepoint();
    Savepoint(const Savepoint& other) = delete;
    Savepoint& operator=(const Savepoint& other) = delete;
    Savepoint(Savepoint&& other) = delete;
    Savepoint& operator=(Savepoint&& other) = delete;
    SavepointStatus Open(const std::string_view& path, SavepointVersion version);
    bool IsOpen() const;
    void Close();
    void Save();

    template<SavepointBasicReadWrite T>
    void Write(T& item)
    {
        Visitor.Reset(ApplicationVersion, InternalVersion);
        Visitor(item);
        Write(Visitor);
    }

    template<SavepointBasicReadWrite T>
    void Write(T& item, SavepointID& id, int level)
    {
        Visitor.Reset(ApplicationVersion, InternalVersion);
        Visitor(item);
        Write(Visitor, id, level);
    }

    template<SavepointBasicReadWrite T>
    void Write(T& item, int x, int y, int level)
    {
        Visitor.Reset(ApplicationVersion, InternalVersion);
        Visitor(item);
        Write(Visitor, x, y, level);
    }

    template<SavepointBasicReadWrite T>
    void Write(T& item, int x, int y, int z, int level)
    {
        Visitor.Reset(ApplicationVersion, InternalVersion);
        Visitor(item);
        Write(Visitor, x, y, z, level);
    }

    template<SavepointBasicReadWrite T>
    void Read(const SavepointReadFunction<T>& function)
    {
        Read([&function](SavepointVisitor& visitor)
        {
            T item;
            visitor(item);
            function(item);
        });
    }

    template<SavepointBasicReadWrite T>
    void Read(const SavepointReadEntityFunction<T>& function, int level)
    {
        Read([&function](SavepointVisitor& visitor, SavepointID id)
        {
            T item;
            visitor(item);
            function(item, id);
        }, level);
    }

    template<SavepointBasicReadWrite T>
    void Read(const SavepointReadTile2DFunction<T>& function, int level)
    {
        Read([&function](SavepointVisitor& visitor, int x, int y)
        {
            T item;
            visitor(item);
            function(item, x, y);
        }, level);
    }

    template<SavepointBasicReadWrite T>
    void Read(const SavepointReadTile3DFunction<T>& function, int level)
    {
        Read([&function](SavepointVisitor& visitor, int x, int y, int z)
        {
            T item;
            visitor(item);
            function(item, x, y, z);
        }, level);
    }

    void Write(SavepointBase* base);
    void Write(SavepointBase* base, SavepointID& id, int level);
    void Write(SavepointBase* base, int x, int y, int level);
    void Write(SavepointBase* base, int x, int y, int z, int level);
    void Read(const SavepointReadBaseFunction& function);
    void Read(const SavepointReadBaseEntityFunction& function, int level);
    void Read(const SavepointReadBaseTile2DFunction& function, int level);
    void Read(const SavepointReadBaseTile3DFunction& function, int level);
    void Delete(const SavepointID id);
    void Clear();

private:
    using ReadFunction = std::function<void(SavepointVisitor& visitor)>;
    using ReadEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;
    using ReadTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;
    using ReadTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;

    bool SetBase(SavepointBase* base);
    SavepointBase* GetBase(SavepointVisitor& visitor);
    void Write(SavepointVisitor& visitor);
    void Write(SavepointVisitor& visitor, SavepointID& id, int level);
    void Write(SavepointVisitor& visitor, int x, int y, int level);
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level);
    void Read(const ReadFunction& function);
    void Read(const ReadEntityFunction& function, int level);
    void Read(const ReadTile2DFunction& function, int level);
    void Read(const ReadTile3DFunction& function, int level);

    typedef struct sqlite3 sqlite;
    typedef struct sqlite3_stmt sqlite_stmt;
    SavepointVersion ApplicationVersion;
    SavepointVersion InternalVersion;
    SavepointVisitor Visitor;
    sqlite3* Handle;
    sqlite3_stmt* WriteStatusStmt;
    sqlite3_stmt* WriteStmt;
    sqlite3_stmt* InsertEntityStmt;
    sqlite3_stmt* UpdateEntityStmt;
    sqlite3_stmt* WriteTile2DStmt;
    sqlite3_stmt* WriteTile3DStmt;
    sqlite3_stmt* ReadStatusStmt;
    sqlite3_stmt* ReadStmt;
    sqlite3_stmt* ReadEntitiesStmt;
    sqlite3_stmt* ReadTiles2DStmt;
    sqlite3_stmt* ReadTiles3DStmt;
    sqlite3_stmt* DeleteEntityStmt;
    sqlite3_stmt* ClearEntitiesStmt;
    sqlite3_stmt* ClearTiles2DStmt;
    sqlite3_stmt* ClearTiles3DStmt;
};