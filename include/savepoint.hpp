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
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

using SavepointLogFunction = std::function<void(const std::string_view& string)>;

void SavepointSetLogFunction(const SavepointLogFunction& function);
void SavepointDefaultLogFunction(const std::string_view& string);
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

template<typename T>
struct SavepointPointerImpl : std::is_pointer<T> {};

template<typename T>
struct SavepointPointerImpl<std::shared_ptr<T>> : std::true_type {};

template<typename T, typename Deleter>
struct SavepointPointerImpl<std::unique_ptr<T, Deleter>> : std::true_type {};

template<typename T>
concept SavepointPointer = SavepointPointerImpl<T>::value;

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

    SavepointVisitor(const SavepointVisitor& other) = delete;
    SavepointVisitor& operator=(const SavepointVisitor& other) = delete;

public:
    SavepointVisitor()
        : Version{}
        , Writer{}
        , Reader{}
        , Offset{0}
    {
        Reset();
    }

    template<SavepointPrimitive T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        if (IsReader())
        {
            if (Version < version)
            {
                if constexpr (sizeof...(Args) > 0)
                {
                    item = T{std::forward<Args>(args)...};
                }
                return;
            }
            if (Offset + sizeof(T) > Reader.size())
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
        else
        {
            Writer.resize(Writer.size() + sizeof(T));
            std::memcpy(Writer.data() + Writer.size() - sizeof(T), std::addressof(item), sizeof(T));
        }
    }

    template<SavepointFreeVisit T, typename... Args>
    void operator()(T& item, SavepointVersion version = {}, Args&&... args)
    {
        if (!Reader.empty())
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
        if (!Reader.empty())
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

    template<typename T>
    void operator()(T* data, size_t maxSize, size_t size)
    {
        if (IsReader())
        {
            if (maxSize < size)
            {
                SavepointLog(std::format("Truncating buffer: {}, {} -> {}", Version.GetString(), size, maxSize));
                size = maxSize;
            }
            if (Offset + size > Reader.size())
            {
                SavepointLog(std::format("Tried to read past visitor: {}", Version.GetString()));
                return;
            }
            std::memcpy(data, Reader.data() + Offset, size);
            Offset += size;
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

    void Reset()
    {
        Writer.resize(sizeof(SavepointVersion));
    }

    bool IsReader() const
    {
        return !Reader.empty();
    }

    bool IsWriter() const
    {
        return !IsReader();
    }

private:
    bool Empty() const
    {
        return Writer.size() == sizeof(SavepointVersion);
    }

    void SetVersion(SavepointVersion version)
    {
        std::memcpy(Writer.data(), &version, sizeof(version));
    }

    void Reset(void* data, uint32_t size)
    {
        Reader = {static_cast<uint8_t*>(data), size};
        Offset = 0;
        operator()(Version);
    }

    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    uint32_t Offset;
};

using SavepointFunction = std::function<void(SavepointVisitor& visitor)>;
using SavepointEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;
using SavepointTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;
using SavepointTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;

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
    Savepoint(const Savepoint& other) = delete;
    Savepoint& operator=(const Savepoint& other) = delete;
    Savepoint(Savepoint&& other) = delete;
    Savepoint& operator=(Savepoint&& other) = delete;
    SavepointStatus Open(const std::string_view& path, SavepointVersion version);
    void Close();
    void Save();
    void Write(SavepointVisitor& visitor);
    void Write(SavepointVisitor& visitor, SavepointID& id, int level);
    void Write(SavepointVisitor& visitor, int x, int y, int level);
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level);
    void Read(const SavepointFunction& function);
    void Read(const SavepointEntityFunction& function, int level);
    void Read(const SavepointTile2DFunction& function, int level);
    void Read(const SavepointTile3DFunction& function, int level);
    void Delete(const SavepointID id);
    void Clear();

private:
    typedef struct sqlite3 sqlite;
    typedef struct sqlite3_stmt sqlite_stmt;
    SavepointVersion Version;
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