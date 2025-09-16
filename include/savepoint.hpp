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

void SavepointSetLogFunction(const SavepointLogFunction& function);
void SavepointDefaultLogFunction(const std::string& string);
void SavepointLog(const std::string& string);

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
concept SavepointVisitable = requires(T t, SavepointArchive& archive)
{
    { t.Visit(archive) };
};

template<typename T>
concept SavepointPrimitive = requires()
{
    requires (!std::is_pointer_v<T>);
    requires (!SavepointVisitable<T>);
};

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
    SavepointArchive(SavepointVersion version)
        : Version{version}
        , Writer{}
        , Reader{}
        , Offset{0}
    {
        operator()(version);
    }

    SavepointArchive& operator=(SavepointArchive&& other)
    {
        Version = other.Version;
        Writer = std::move(other.Writer);
        Reader = other.Reader;
        Offset = other.Offset;
        return *this;
    }

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

    void Skip(uint32_t size)
    {
        if (!Reader.empty())
        {
            if (Offset + size > Reader.size())
            {
                SavepointLog(std::format("Tried to skip past the end of an archive: {}", Version.GetString()));
                return;
            }
            Offset += size;
        }
        else
        {
            Writer.resize(Writer.size() + size);
        }
    }

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

class Savepoint
{
public:
    Savepoint() = default;
    Savepoint(const Savepoint& other) = delete;
    Savepoint& operator=(const Savepoint& other) = delete;
    Savepoint(Savepoint&& other) = delete;
    Savepoint& operator=(Savepoint&& other) = delete;
    bool Open(const std::string& path);
    void Close(bool save = true);
    void Save();
    void Write(const SavepointArchive& archive);
    void Write(const SavepointArchive& archive, SavepointID& id, int level = 0);
    void Write(const SavepointArchive& archive, int x, int y, int level = 0);
    void Write(const SavepointArchive& archive, int x, int y, int z, int level = 0);
    SavepointArchive Read();
    void Read(const SavepointEntityFunction& function, int level = 0);
    void Read(const SavepointTile2DFunction& function, int level = 0);
    void Read(const SavepointTile3DFunction& function, int level = 0);
    void Delete(const SavepointID id);
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