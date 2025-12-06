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
#include <array>
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
void SavepointLog(const std::string_view& string);

class SavepointVersion
{
private:
    friend class SavepointDatabase;

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
    friend class SavepointDatabase;

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
    friend class SavepointDatabase;

public:
    virtual void Visit(SavepointVisitor& visitor) = 0;

private:
    virtual const std::string_view SavepointDerivedGetString() const = 0;
};

using SavepointDerivedFunction = std::function<SavepointBase*()>;

void SavepointAddDerivedFunction(const std::string_view& string, const SavepointDerivedFunction& function);

#define SAVEPOINT_DERIVED(T) \
    private: \
        struct SavepointDerivedAddFactory \
        { \
            static SavepointBase* Factory() \
            { \
                return new T(); \
            } \
            SavepointDerivedAddFactory() \
            { \
                SavepointAddDerivedFunction(#T, Factory); \
            } \
        }; \
        static inline SavepointDerivedAddFactory SavepointFactory; \
        const std::string_view SavepointDerivedGetString() const override \
        { \
            return #T;\
        } \
    public: \

template<typename T>
struct SavepointPointerImpl : std::is_pointer<T> {};

template<typename T>
struct SavepointPointerImpl<std::shared_ptr<T>> : std::true_type {};

template<typename T, typename Deleter>
struct SavepointPointerImpl<std::unique_ptr<T, Deleter>> : std::true_type {};

template<typename T>
concept SavepointPointer = SavepointPointerImpl<T>::value;

template<typename T>
struct SavepointVectorImpl : std::false_type {};

template<typename T, typename Allocator>
struct SavepointVectorImpl<std::vector<T, Allocator>> : std::true_type {};

template<typename T>
concept SavepointVector = SavepointVectorImpl<T>::value;

template<typename T>
struct SavepointArrayImpl : std::false_type {};

template<typename T, size_t N>
struct SavepointArrayImpl<std::array<T, N>> : std::true_type
{
    static constexpr size_t kSize = N;
};

template<typename T>
concept SavepointArray = SavepointArrayImpl<T>::value;

template<typename T>
inline constexpr size_t SavepointArraySize = SavepointArrayImpl<T>::kSize;

template<typename T>
struct SavepointStringImpl : std::false_type {};

template<typename T, typename Traits, typename Allocator>
struct SavepointStringImpl<std::basic_string<T, Traits, Allocator>> : std::true_type {};

template<typename T>
concept SavepointString = SavepointStringImpl<T>::value;

template<typename T>
concept SavepointFreeVisit = requires(SavepointVisitor visitor, T item) { { SavepointVisit(visitor, item) }; };

template<typename T>
concept SavepointMemberVisit = requires(SavepointVisitor visitor, T item) { { item.Visit(visitor) }; };

template<typename T>
concept SavepointPrimitive = !SavepointPointer<T> && !SavepointFreeVisit<T> && !SavepointMemberVisit<T>;

class SavepointVisitor
{
private:
    friend class SavepointDatabase;

    static constexpr size_t kHeaderSize = sizeof(SavepointVersion) * 2;

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
        }
        SavepointVisit(*this, item);
    }

    template<SavepointMemberVisit T, typename... Args>
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
        }
        item.Visit(*this);
    }

    template<SavepointPrimitive T>
    void operator()(T* data, size_t maxSize, size_t size)
    {
        if (IsReader())
        {
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
                if (Offset + size > Reader.size())
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

    bool IsReader() const
    {
        return !Reader.empty();
    }

    bool IsWriter() const
    {
        return !IsReader();
    }

    void Reset()
    {
        Reader = {};
        Writer.resize(kHeaderSize);
    }

private:
    bool Empty() const
    {
        return Writer.size() == kHeaderSize;
    }

    void SetApplicationVersion(SavepointVersion version)
    {
        std::memcpy(Writer.data(), &version, sizeof(SavepointVersion));
    }

    void SetSavepointVersion(SavepointVersion version)
    {
        std::memcpy(Writer.data() + sizeof(SavepointVersion), &version, sizeof(SavepointVersion));
    }

    void Reset(void* data, size_t size)
    {
        Reader = {static_cast<uint8_t*>(data), size};
        Offset = 0;
        operator()(Version);
        Offset += sizeof(SavepointVersion);
    }

    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    size_t Offset;
};

template<SavepointVector T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    size_t size = item.size() * sizeof(T::value_type);
    visitor(size);
    item.resize(size);
    visitor(item.data(), size, size);
}

template<SavepointArray T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    static constexpr size_t kCapacity = SavepointArraySize<T> * sizeof(T::value_type);
    size_t size = kCapacity;
    visitor(size);
    visitor(item.data(), kCapacity, size);
}

template<SavepointString T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    size_t size = item.size() * sizeof(T::value_type);
    visitor(size);
    item.resize(size);
    visitor(item.data(), size, size);
}

using SavepointReadFunction = std::function<void(SavepointVisitor& visitor)>;
using SavepointReadEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;
using SavepointReadTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;
using SavepointReadTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;
using SavepointReadBaseFunction = std::function<void(SavepointBase* base)>;
using SavepointReadBaseEntityFunction = std::function<void(SavepointBase* base, SavepointID id)>;
using SavepointReadBaseTile2DFunction = std::function<void(SavepointBase* base, int x, int y)>;
using SavepointReadBaseTile3DFunction = std::function<void(SavepointBase* base, int x, int y, int z)>;

enum class SavepointStatus
{
    Failed,
    Existing,
    New,
};

class SavepointDatabase
{
public:
    SavepointDatabase();
    ~SavepointDatabase();

    SavepointDatabase(const SavepointDatabase& other) = delete;
    SavepointDatabase& operator=(const SavepointDatabase& other) = delete;
    SavepointDatabase(SavepointDatabase&& other) = delete;
    SavepointDatabase& operator=(SavepointDatabase&& other) = delete;
    SavepointStatus Open(const std::string_view& path, SavepointVersion version);
    bool IsOpen() const;
    void Close();
    void Save();
    void Write(SavepointVisitor& visitor);
    void Write(SavepointVisitor& visitor, SavepointID& id, int level);
    void Write(SavepointVisitor& visitor, int x, int y, int level);
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level);
    void Write(SavepointBase* base);
    void Write(SavepointBase* base, SavepointID& id, int level);
    void Write(SavepointBase* base, int x, int y, int level);
    void Write(SavepointBase* base, int x, int y, int z, int level);
    void Read(const SavepointReadFunction& function);
    void Read(const SavepointReadEntityFunction& function, int level);
    void Read(const SavepointReadTile2DFunction& function, int level);
    void Read(const SavepointReadTile3DFunction& function, int level);
    void Read(const SavepointReadBaseFunction& function);
    void Read(const SavepointReadBaseEntityFunction& function, int level);
    void Read(const SavepointReadBaseTile2DFunction& function, int level);
    void Read(const SavepointReadBaseTile3DFunction& function, int level);
    void Delete(const SavepointID id);
    void Clear();

private:
    bool SetBase(SavepointBase* base);
    SavepointBase* GetBase(SavepointVisitor& visitor);

    typedef struct sqlite3 sqlite;
    typedef struct sqlite3_stmt sqlite_stmt;
    SavepointVersion Version;
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