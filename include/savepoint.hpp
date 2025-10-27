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

/* Set the log function used internally */
void SavepointSetLogFunction(const SavepointLogFunction& function);

/* Internal */
void SavepointLog(const std::string_view& string);

/* A major.minor.minor version number as a uint32 for quick comparisons */
class SavepointVersion
{
private:
    friend class SavepointStorage;

public:
    /* Create the lowest version */
    constexpr SavepointVersion()
        : Value{0}
    {
    }

    /* Create a version from major, minor, patch */
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

    /* Get the version as a string in the format major.minor.patch */
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

/* A unique ID (per savepoint) for referencing entities */
class SavepointID
{
private:
    friend class SavepointStorage;

public:
    /* Create an invalid ID */
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

    /* Check if an ID is valid */
    constexpr operator bool() const
    {
        return Value != SavepointID{}.Value;
    }

private:
    uint32_t Value;
};

/*
 * Base class for the user base class
 * 
 * class Entity : public SavepointBase
 * {
 * };
 * 
 * class Player : public Entity
 * {
 *     SAVEPOINT_DERIVED(Player)
 * 
 *     void Visit(SavepointVisitor& visitor) override
 *     {
 *         // ...
 *     }
 * };
 */
class SavepointBase
{
private:
    friend class SavepointStorage;

public:
    /* Visit function users must implement */
    virtual void Visit(SavepointVisitor& visitor) = 0;

private:
    /* Get the class name of a derived type (internal) */
    virtual const std::string_view SavepointDerivedGetString() const = 0;
};

/* Internal */
using SavepointDerivedFunction = std::function<SavepointBase*()>;

/* Internal */
void SavepointAddDerivedFunction(const std::string_view& string, const SavepointDerivedFunction& function);

/* Register a derived type as an base (see SavepointBase) */
#define SAVEPOINT_DERIVED(T) \
    private: \
        struct SavepointDerivedRegistrar##T \
        { \
            SavepointDerivedRegistrar##T() \
            { \
                SavepointAddDerivedFunction(#T, []() { return new T(); }); \
            } \
        }; \
        static inline SavepointDerivedRegistrar##T SavepointDerivedRegistrar; \
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

/*
 * Byte buffer for reading/writing using the visitor pattern
 * 
 * class Entity
 * {
 *     int X;
 *     int Z;
 * 
 *     void Visit(SavepointVisitor& visitor)
 *     {
 *         visitor(X);
 *         visitor(Z);
 *     }
 * };
 * 
 * class Player : public Entity
 * {
 *     int Health;
 * 
 *     void Visit(SavepointVisitor& visitor)
 *     {
 *         Entity::Visit(visitor);
 *         visitor(Health);
 *     }
 * };
 * 
 * For a new version, avoid corrupting old saves by versioning
 * 
 * class EntityV2
 * {
 *     int X;
 *     int Y; // new
 *     int Z;
 *     int W = 1; // new
 * 
 *     void Visit(SavepointVisitor& visitor)
 *     {
 *         visitor(X);
 *         visitor(Y, {0, 2, 1}, 0); // Added in 0.2.1. Default to 0
 *         visitor(Z);
 *         visitor(W, {0, 2, 1}); // Added in 0.2.1. Don't change value
 *     }
 * };
 * 
 * For external types, use the free function
 * 
 * struct ExternalEntity
 * {
 *     std::vector<int> Data;
 * };
 * 
 * void SavepointVisit(SavepointVisitor& visitor, ExternalEntity& entity)
 * {
 *     visitor(entity.Data);
 * }
 */
class SavepointVisitor
{
private:
    friend class SavepointStorage;

    /* 
     * Header is composed of 2 versions:
     * 1. The application version
     * 2. The savepoint version (reserved for future use)
     */
    static constexpr size_t kHeaderSize = sizeof(SavepointVersion) * 2;

    SavepointVisitor(const SavepointVisitor& other) = delete;
    SavepointVisitor& operator=(const SavepointVisitor& other) = delete;

public:
    /* Create a visitor for writing */
    SavepointVisitor()
        : Version{}
        , Writer{}
        , Reader{}
        , Offset{0}
    {
        Reset();
    }

    /* Visit a primitive (e.g. float, uint32_t). If reading, checks that the version is satisfied */
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

    /* Visit an base with a SavepointVisit free function defined. If reading, checks that the version is satisfied */
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

    /* Visit an base with a Visit member function defined. If reading, checks that the version is satisfied */
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

    /* Visit an allocated buffer. Copies up to maxSize but expects size in the visitor. Try to avoid using */
    template<SavepointPrimitive T>
    void operator()(T* data, size_t maxSize, size_t size)
    {
        if (IsReader())
        {
            /* Should never happen. Used to avoid compile error when using memcpy on a const pointer */
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

    /* Check if visitor is reading. Try to avoid using */
    bool IsReader() const
    {
        return !Reader.empty();
    }

    /* Check if visitor is reading. Try to avoid using */
    bool IsWriter() const
    {
        return !IsReader();
    }

    /* Reset a visitor for writing */
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

    /* Reset a visitor for reading */
    void Reset(void* data, size_t size)
    {
        Reader = {static_cast<uint8_t*>(data), size};
        Offset = 0;
        operator()(Version);
        /* Skip savepoint version. Reserved for future use */
        Offset += sizeof(SavepointVersion);
    }

    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    size_t Offset;
};

/* Visit a vector */
template<SavepointVector T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    size_t size = item.size() * sizeof(T::value_type);
    visitor(size);
    item.resize(size);
    visitor(item.data(), size, size);
}

/* Visit an array. Truncates if not large enough */
template<SavepointArray T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    static constexpr size_t kCapacity = SavepointArraySize<T> * sizeof(T::value_type);
    size_t size = kCapacity;
    visitor(size);
    visitor(item.data(), kCapacity, size);
}

/* Visit a string */
template<SavepointString T>
void SavepointVisit(SavepointVisitor& visitor, T& item)
{
    size_t size = item.size() * sizeof(T::value_type);
    visitor(size);
    item.resize(size);
    visitor(item.data(), size, size);
}

/* Read callbacks */
using SavepointReadFunction = std::function<void(SavepointVisitor& visitor)>;
using SavepointReadEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;
using SavepointReadTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;
using SavepointReadTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;

/* Base read callbacks */
using SavepointReadBaseFunction = std::function<void(SavepointBase* base)>;
using SavepointReadBaseEntityFunction = std::function<void(SavepointBase* base, SavepointID id)>;
using SavepointReadBaseTile2DFunction = std::function<void(SavepointBase* base, int x, int y)>;
using SavepointReadBaseTile3DFunction = std::function<void(SavepointBase* base, int x, int y, int z)>;

enum class SavepointStatus
{
    /* Failed to open savepoint */
    Failed,

    /* Opened an existing savepoint */
    Existing,

    /* Opened a new savepoint */
    New,
};

/*
 * Database connection handle
 *
 * int main()
 * {
 *     SavepointStorage savepoint;
 *     switch (savepoint.Open("<path>", {1, 1, 1}))
 *     {
 *
 *     // Failed to open
 *     case SavepointStatus::Failed:
 *         return 1;
 *
 *     // Read entities and tiles
 *     case SavepointStatus::Existing:
 *         break;
 *
 *     // Generate new world
 *     case SavepointStatus::New:
 *         break;
 *
 *     }
 *     savepoint.Save();
 *     savepoint.Close();
 *     return 0;
 * }
 */
class SavepointStorage
{
public:
    /* Default initialize */
    SavepointStorage();
    
    /* Does not close the database */
    ~SavepointStorage();

    SavepointStorage(const SavepointStorage& other) = delete;
    SavepointStorage& operator=(const SavepointStorage& other) = delete;
    SavepointStorage(SavepointStorage&& other) = delete;
    SavepointStorage& operator=(SavepointStorage&& other) = delete;
    
    /* Open a database connection. Version should be your application version */
    SavepointStatus Open(const std::string_view& path, SavepointVersion version);
    
    /* Close the database connection */
    void Close();

    /* Commit changes and change status to Existing */
    void Save();

    /* Write a single instance to the database */
    void Write(SavepointVisitor& visitor);

    /* Write an entity to a level. If ID is invalid, sets the ID. Otherwise updates entity (possibly changes its level) */
    void Write(SavepointVisitor& visitor, SavepointID& id, int level);

    /* Write a tile to an xy coordinate and level */
    void Write(SavepointVisitor& visitor, int x, int y, int level);

    /* Write a tile to an xyz coordinate and level */
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level);

    /* Write a single instance to the database */
    void Write(SavepointBase* base);

    /* Write an entity to a level */
    void Write(SavepointBase* base, SavepointID& id, int level);

    /* Write a tile to an xy coordinate and level */
    void Write(SavepointBase* base, int x, int y, int level);

    /* Write a tile to an xyz coordinate and level */
    void Write(SavepointBase* base, int x, int y, int z, int level);
    
    /* Read a single instance from the database */
    void Read(const SavepointReadFunction& function);

    /* Read all entities from level */
    void Read(const SavepointReadEntityFunction& function, int level);

    /* Read all xy tiles from level */
    void Read(const SavepointReadTile2DFunction& function, int level);

    /* Read all xyz tiles from level */
    void Read(const SavepointReadTile3DFunction& function, int level);

    /* Read a single instance from the database */
    void Read(const SavepointReadBaseFunction& function);

    /* Read all entities from level */
    void Read(const SavepointReadBaseEntityFunction& function, int level);

    /* Read all xy tiles from level */
    void Read(const SavepointReadBaseTile2DFunction& function, int level);

    /* Read all xyz tiles from level */
    void Read(const SavepointReadBaseTile3DFunction& function, int level);
    
    /* Delete an entity from the database */
    void Delete(const SavepointID id);
    
    /* Delete all entities and tiles */
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