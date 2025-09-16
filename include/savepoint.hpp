#pragma once

#include <cstdint>
#include <cstring>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

class Savepoint;
class SavepointID;
class SavepointArchive;
class SavepointVersion;

using SavepointLogFunction = std::function<void(const std::string& string)>;

/**
 * @brief Set the log output function
 * 
 * @param function 
 */
void SavepointSetLogFunction(const SavepointLogFunction& function);

/**
 * @brief Call the default log output function
 * 
 * @param string 
 */
void SavepointDefaultLogFunction(const std::string& string);

/**
 * @brief Call the current log output function
 * 
 * @param string 
 */
void SavepointLog(const std::string& string);

/**
 * @brief 
 * 
 */
class SavepointVersion
{
private:
    friend class Savepoint;

public:
    /**
     * @brief Create the lowest version
     * 
     */
    constexpr SavepointVersion()
        : Value{0}
    {
    }

    /**
     * @brief Create a specific version
     * 
     * @param major
     * @param minor
     * @param patch
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
     * @brief Get the version as a string with the format "major.minor.patch"
     * 
     * @return std::string 
     */
    std::string GetString() const
    {
        return std::format("{}.{}.{}", GetMajor(), GetMinor(), GetPatch());
    }

    /**
     * @brief 
     * 
     * @param other 
     * @return
     */
    constexpr bool operator==(const SavepointVersion other) const
    {
        return Value == other.Value;
    }

    /**
     * @brief 
     * 
     * @param other 
     * @return
     */
    constexpr bool operator!=(const SavepointVersion other) const
    {
        return Value != other.Value;
    }

    /**
     * @brief 
     * 
     * @param other 
     * @return
     */
    constexpr bool operator<(const SavepointVersion other) const
    {
        return Value < other.Value;
    }

    /**
     * @brief 
     * 
     * @param other 
     * @return
     */
    constexpr bool operator>(const SavepointVersion other) const
    {
        return Value > other.Value;
    }

    /**
     * @brief 
     * 
     * @param other 
     * @return
     */
    constexpr bool operator<=(const SavepointVersion other) const
    {
        return Value <= other.Value;
    }

    /**
     * @brief 
     * 
     * @param other 
     * @return
     */
    constexpr bool operator>=(const SavepointVersion other) const
    {
        return Value >= other.Value;
    }

private:
    uint32_t Value;
};

/**
 * @brief A unique ID for representing entities
 * 
 */
class SavepointID
{
private:
    friend class Savepoint;

public:
    /**
     * @brief Create an invalid ID
     * 
     */
    constexpr SavepointID()
        : Value{std::numeric_limits<uint32_t>::max()}
    {
    }

    /**
     * @brief 
     * 
     * @param other 
     * @return
     */
    constexpr bool operator==(const SavepointID other) const
    {
        return Value == other.Value;
    }

    /**
     * @brief 
     * 
     * @param other 
     * @return
     */
    constexpr bool operator!=(const SavepointID other) const
    {
        return Value != other.Value;
    }

    /**
     * @brief Check if the ID is valid
     * 
     * @return True if the ID is valid
     */
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
inline constexpr bool SavepointPointer = SavepointPointerImpl<T>::value;

template<typename T>
concept SavepointVisitable = requires { { &T::Visit }; };

template<typename T>
concept SavepointPrimitive = !SavepointPointer<T> && !SavepointVisitable<T>;

/**
 * @brief Byte buffer for serializing to/from the savepoint
 * 
 */
class SavepointArchive
{
private:
    friend class Savepoint;

    SavepointArchive()
        : Version{}
        , Writer{}
        , Reader{}
        , Offset{0}
    {
    }

public:
    SavepointArchive(const SavepointArchive& other) = default;
    SavepointArchive& operator=(const SavepointArchive& other) = default;

    /**
     * @brief Create a new versioned archive
     * 
     * @param version 
     */
    SavepointArchive(SavepointVersion version)
        : Version{version}
        , Writer{}
        , Reader{}
        , Offset{0}
    {
        operator()(version);
    }

    /**
     * @brief Move an archive
     * 
     * @param other 
     * @return
     */
    SavepointArchive& operator=(SavepointArchive&& other)
    {
        Version = other.Version;
        Writer = std::move(other.Writer);
        Reader = other.Reader;
        Offset = other.Offset;
        return *this;
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
    template<SavepointPrimitive T, typename... Args>
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
            if (Offset + sizeof(T) > Reader.size())
            {
                SavepointLog(std::format("Tried to read past the end of an archive: {} -> {}", Version.GetString(), version.GetString()));
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
     * @brief 
     * 
     * @tparam T 
     * @tparam Args 
     * @param item 
     * @param version 
     * @param args 
     */
    template<SavepointVisitable T, typename... Args>
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
     * @brief 
     * 
     */
    void Reset()
    {
        Writer.clear();
        operator()(Version);
    }

private:
    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    uint32_t Offset;
};

using SavepointEntityFunction = std::function<void(SavepointArchive& archive, SavepointID id)>;
using SavepointTile2DFunction = std::function<void(SavepointArchive& archive, int x, int y)>;
using SavepointTile3DFunction = std::function<void(SavepointArchive& archive, int x, int y, int z)>;

/**
 * @brief A database connection
 * 
 */
class Savepoint
{
public:
    Savepoint() = default;
    Savepoint(const Savepoint& other) = delete;
    Savepoint& operator=(const Savepoint& other) = delete;
    Savepoint(Savepoint&& other) = delete;
    Savepoint& operator=(Savepoint&& other) = delete;

    /**
     * @brief Open a new or existing database
     * 
     * @param path 
     * @return True if the database was successfully created/opened
     */
    bool Open(const std::string& path);
    
    /**
     * @brief 
     * 
     * @param save If true, saves before closing
     */
    void Close(bool save = true);

    /**
     * @brief Save all pending changes. Ends the current transaction and begins a new one
     * 
     */
    void Save();
    
    /**
     * @brief Write a single instance archive. Useful for metadata (e.g. time of day)
     * 
     * @param archive 
     */
    void Write(const SavepointArchive& archive);
    
    /**
     * @brief Write or move an entity to a level
     * 
     * @param archive 
     * @param id The in/out ID
     * @param level 
     */
    void Write(const SavepointArchive& archive, SavepointID& id, int level);

    /**
     * @brief Write a tile to a position and level
     * 
     * @param archive 
     * @param x 
     * @param y 
     * @param level 
     */
    void Write(const SavepointArchive& archive, int x, int y, int level);

    /**
     * @brief Write a tile to a position and level
     * 
     * @param archive 
     * @param x 
     * @param y 
     * @param z 
     * @param level 
     */
    void Write(const SavepointArchive& archive, int x, int y, int z, int level);

    /**
     * @brief Read a single instance archive
     * 
     * @return SavepointArchive 
     */
    SavepointArchive Read();
    
    /**
     * @brief Read entities from a level
     * 
     * @param function 
     * @param level 
     */
    void Read(const SavepointEntityFunction& function, int level);

    /**
     * @brief Read tiles from a level
     * 
     * @param function 
     * @param level 
     */
    void Read(const SavepointTile2DFunction& function, int level);

    /**
     * @brief Read tiles from a level
     * 
     * @param function 
     * @param level 
     */
    void Read(const SavepointTile3DFunction& function, int level);

    /**
     * @brief Delete an entity
     * 
     * @param id 
     */
    void Delete(const SavepointID id);

    /**
     * @brief Delete all entities and tiles
     * 
     */
    void Clear();

private:
    typedef struct sqlite3 sqlite;
    typedef struct sqlite3_stmt sqlite_stmt;
    sqlite3* Handle;
    sqlite3_stmt* WriteStmt;
    sqlite3_stmt* InsertEntityStmt;
    sqlite3_stmt* UpdateEntityStmt;
    sqlite3_stmt* WriteTile2DStmt;
    sqlite3_stmt* WriteTile3DStmt;
    sqlite3_stmt* ReadStmt;
    sqlite3_stmt* ReadEntitiesStmt;
    sqlite3_stmt* ReadTiles2DStmt;
    sqlite3_stmt* ReadTiles3DStmt;
    sqlite3_stmt* DeleteEntityStmt;
    sqlite3_stmt* ClearEntitiesStmt;
    sqlite3_stmt* ClearTiles2DStmt;
    sqlite3_stmt* ClearTiles3DStmt;
};