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

/**
 * @brief The signature of the log function
 * 
 * @param string The message
 */
using SavepointLogFunction = std::function<void(const std::string_view& string)>;

/**
 * @brief Replace the default log function
 * 
 * @param function 
 */
void SavepointSetLogFunction(const SavepointLogFunction& function);

/**
 * @brief Log to stderr
 * 
 * @param string 
 */
void SavepointDefaultLogFunction(const std::string_view& string);

/**
 * @brief Forward a message to the set log function
 * 
 * @param string 
 */
void SavepointLog(const std::string_view& string);

/**
 * @brief Representation of a major.minor.patch version
 */
class SavepointVersion
{
private:
    friend class Savepoint;

public:
    /**
     * @brief Create a version with the smallest value
     */
    constexpr SavepointVersion()
        : Value{0}
    {
    }

    /**
     * @brief Create a version
     * 
     * @param major The major version
     * @param minor The minor version
     * @param patch The patch version
     */
    constexpr SavepointVersion(uint32_t major, uint32_t minor, uint32_t patch)
        : Value{major << 24 | minor << 16 | patch}
    {
    }

    /**
     * @brief Get the major version
     * 
     * @return The major version
     */
    constexpr uint32_t GetMajor() const
    {
        return (Value >> 24) & 0xFF;
    }

    /**
     * @brief Get the minor version
     * 
     * @return The minor version
     */
    constexpr uint32_t GetMinor() const
    {
        return (Value >> 16) & 0xFF;
    }

    /**
     * @brief Get the patch version
     * 
     * @return The patch version
     */
    constexpr uint32_t GetPatch() const
    {
        return Value & 0xFFFF;
    }

    /**
     * @brief Get the version as a string with the format major.minor.patch
     * 
     * @return The version string
     */
    std::string GetString() const
    {
        return std::format("{}.{}.{}", GetMajor(), GetMinor(), GetPatch());
    }

    /**
     * @brief Check if the version is equal to the other
     * 
     * @param other The other version
     * @return If the version is equal
     */
    constexpr bool operator==(const SavepointVersion other) const
    {
        return Value == other.Value;
    }

    /**
     * @brief Check if the version is not equal to the other
     * 
     * @param other The other version
     * @return If the version is not equal
     */
    constexpr bool operator!=(const SavepointVersion other) const
    {
        return Value != other.Value;
    }

    /**
     * @brief Check if the version is less than to the other
     * 
     * @param other The other version
     * @return If the version is less than
     */
    constexpr bool operator<(const SavepointVersion other) const
    {
        return Value < other.Value;
    }

    /**
     * @brief Check if the version is greater than to the other
     * 
     * @param other The other version
     * @return If the version is greater than
     */
    constexpr bool operator>(const SavepointVersion other) const
    {
        return Value > other.Value;
    }

    /**
     * @brief Check if the version is less than or equal to the other
     * 
     * @param other The other version
     * @return If the version is less than or equal
     */
    constexpr bool operator<=(const SavepointVersion other) const
    {
        return Value <= other.Value;
    }

    /**
     * @brief Check if the version is greater than or equal to the other
     * 
     * @param other The other version
     * @return If the version is greater than or equal
     */
    constexpr bool operator>=(const SavepointVersion other) const
    {
        return Value >= other.Value;
    }

private:
    uint32_t Value;
};

/**
 * Unique ID for referencing entities (not unique across different savepoints)
 */
class SavepointID
{
private:
    friend class Savepoint;

public:
    /**
     * @brief Create an invalid ID
     */
    constexpr SavepointID()
        : Value{std::numeric_limits<uint32_t>::max()}
    {
    }

    /**
     * @brief Check if the ID is equal to the other
     * 
     * @param other The other ID
     * 
     * @return If the ID is equal
     */
    constexpr bool operator==(const SavepointID other) const
    {
        return Value == other.Value;
    }

    /**
     * @brief Check if the ID is not equal to the other
     * 
     * @param other The other ID
     * 
     * @return If the ID is not equal
     */
    constexpr bool operator!=(const SavepointID other) const
    {
        return Value != other.Value;
    }

    /**
     * @brief Check if the ID is valid
     * 
     * @return If the ID is valid
     */
    constexpr operator bool() const
    {
        return Value != SavepointID{}.Value;
    }

private:
    uint32_t Value;
};

/**
 * @brief Check if a type is a pointer
 * 
 * @tparam T The type
 */
template<typename T>
struct SavepointPointerImpl : std::is_pointer<T> {};

/**
 * @copydoc SavepointPointerImpl
 */
template<typename T>
struct SavepointPointerImpl<std::shared_ptr<T>> : std::true_type {};

/**
 * @copydoc SavepointPointerImpl
 */
template<typename T, typename Deleter>
struct SavepointPointerImpl<std::unique_ptr<T, Deleter>> : std::true_type {};

/**
 * @copydoc SavepointPointerImpl
 */
template<typename T>
concept SavepointPointer = SavepointPointerImpl<T>::value;

/**
 * @brief Check if a type has a free visit function
 * 
 * @tparam T The type
 */
template<typename T>
concept SavepointFreeVisit = requires(SavepointVisitor visitor, T item) { { SavepointVisit(visitor, item) }; };

/**
 * @brief Check if a type has a member visit function
 * 
 * @tparam T The type
 */
template<typename T>
concept SavepointMemberVisit = requires(SavepointVisitor visitor, T item) { { item.Visit(visitor) }; };

/**
 * @brief Check if a type is a primitive (can be copied)
 * 
 * @tparam T The type
 */
template<typename T>
concept SavepointPrimitive = !SavepointPointer<T> && !SavepointFreeVisit<T> && !SavepointMemberVisit<T>;

/**
 * Visitor for serializing to/from a byte stream
 */
class SavepointVisitor
{
private:
    friend class Savepoint;

    SavepointVisitor() = default;
    SavepointVisitor(const SavepointVisitor& other) = delete;
    SavepointVisitor& operator=(const SavepointVisitor& other) = delete;

public:
    /**
     * @brief Create a new visitor
     * 
     * @param version The version (should be the version of your application)
     */
    SavepointVisitor(SavepointVersion version)
        : Version{version}
        , Writer{}
        , Reader{}
        , Offset{0}
    {
        operator()(version);
    }

    /**
     * @brief Serialize to/from the byte stream
     * 
     * @tparam T The type to serialize
     * @tparam Args The types of the arguments for construction
     * @param item The item to serialize
     * @param version The visitor version required to deserialize
     * @param args The args to forward for construction if deserialization requirements aren't met
     */
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

    /**
     * @copydoc operator()
     */
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

    /**
     * @copydoc operator()
     */
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

    /**
     * @brief Serialize a buffer to/from a byte stream
     * 
     * @tparam T The pointer type
     * @param data The data to serialize
     * @param maxSize The size in bytes of the memory referenced by data
     * @param size The size in bytes of the serialized memory
     */
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

    /**
     * @brief Reset the visitor
     */
    void Reset()
    {
        Writer.resize(sizeof(Version));
    }

    /**
     * @brief Check if the visitor is reading
     * 
     * @return If visitor is reading
     */
    bool IsReader() const
    {
        return !Reader.empty();
    }

    /**
     * @brief Check if the visitor is writing
     * 
     * @return If visitor is writing
     */
    bool IsWriter() const
    {
        return !IsReader();
    }

private:
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

/**
 * @brief The signature of the reader callback
 */
using SavepointFunction = std::function<void(SavepointVisitor& visitor)>;

/**
 * @brief The signature of the entity reader callback
 */
using SavepointEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;

/**
 * @brief The signature of the 2D tile reader callback
 */
using SavepointTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;

/**
 * @brief The signature of the 3D tile reader callback
 */
using SavepointTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;

/**
 * @brief Return codes
 */
enum class SavepointStatus
{
    Failed,   /**< A failure occured */
    Existing, /**< An existing savepoint was opened */
    New,      /**< A new savepoint was opened */
};

/**
 * Database connection handle
 */
class Savepoint
{
public:
    /**
     * Create a new savepoint
     */
    Savepoint();
    Savepoint(const Savepoint& other) = delete;
    Savepoint& operator=(const Savepoint& other) = delete;
    Savepoint(Savepoint&& other) = delete;
    Savepoint& operator=(Savepoint&& other) = delete;
    
    /**
     * @brief Open the savepoint
     * 
     * @param path The path to the file
     * @return The savepoint status
     */
    SavepointStatus Open(const std::string_view& path);

    /**
     * @brief Close the savepoint (does NOT save!)
     */
    void Close();
    
    /**
     * @brief Save pending changes
     */
    void Save();

    /**
     * @brief Write a global visitor (not shared between savepoints)
     * 
     * @param visitor The visitor to write
     */
    void Write(const SavepointVisitor& visitor);

    /**
     * @brief Write a visitor referenced by an ID
     * 
     * @param visitor The visitor to write
     * @param id The ID to use (not unique between levels). If invalid, creates a new ID
     * @param level The level to write. If id exists, moves to the new level
     */
    void Write(const SavepointVisitor& visitor, SavepointID& id, int level);

    /**
     * @brief Write a visitor to a 2D tile location
     * 
     * @param visitor The visitor to write
     * @param x The x coordinate
     * @param y The y coordinate
     * @param level The level to write
     */
    void Write(const SavepointVisitor& visitor, int x, int y, int level);

    /**
     * @brief Write a visitor to a 2D tile location
     * 
     * @param visitor The visitor to write
     * @param x The x coordinate
     * @param y The y coordinate
     * @param z The z coordinate
     * @param level The level to write
     */
    void Write(const SavepointVisitor& visitor, int x, int y, int z, int level);

    /**
     * @brief Read a global visitor (not shared between savepoints)
     * 
     * @param function The callback to use
     */
    void Read(const SavepointFunction& function);

    /**
     * @brief Read visitors referenced by an ID
     * 
     * @param function The callback to use
     * @param level The level to read
     */
    void Read(const SavepointEntityFunction& function, int level);

    /**
     * @brief Read visitors referenced by 2D coordinates
     * 
     * @param function The callback to use
     * @param level The level to read
     */
    void Read(const SavepointTile2DFunction& function, int level);

    /**
     * @brief Read visitors referenced by 3D coordinates
     * 
     * @param function The callback to use
     * @param level The level to read
     */
    void Read(const SavepointTile3DFunction& function, int level);

    /**
     * @brief Delete an ID
     * 
     * @param id The ID to delete
     */
    void Delete(const SavepointID id);

    /**
     * @brief Delete all entities and tiles
     */
    void Clear();

private:
    typedef struct sqlite3 sqlite;
    typedef struct sqlite3_stmt sqlite_stmt;
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