#pragma once

#include <savepoint_fwd.hpp>

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
inline constexpr bool SavepointPointer = SavepointPointerImpl<T>::value;

template<typename T>
concept SavepointVisitable = requires { { &T::Visit }; };

template<typename T>
concept SavepointPrimitive = !SavepointPointer<T> && !SavepointVisitable<T>;

class SavepointVisitor
{
private:
    friend class Savepoint;

    SavepointVisitor() = default;
    SavepointVisitor(const SavepointVisitor& other) = delete;
    SavepointVisitor& operator=(const SavepointVisitor& other) = delete;

public:
    SavepointVisitor(SavepointVersion version)
        : Version{version}
        , Writer{}
        , Reader{}
        , Offset{0}
    {
        operator()(version);
    }

    SavepointVisitor& operator=(SavepointVisitor&& other)
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
                SavepointLog(std::format("Tried to read past the end of an visitor: {} -> {}", Version.GetString(), version.GetString()));
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

    void Reset()
    {
        Reader = {};
        Writer.clear();
        operator()(Version);
    }

private:
    void Reset(void* data, uint32_t size)
    {
        Reader = {static_cast<uint8_t*>(data), size};
        Offset = 0;
        operator()(Version);
    }

private:
    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    uint32_t Offset;
};

using SavepointFunction = std::function<void(SavepointVisitor& visitor)>;
using SavepointEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;
using SavepointTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;
using SavepointTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;

class Savepoint
{
public:
    Savepoint();
    Savepoint(const Savepoint& other) = delete;
    Savepoint& operator=(const Savepoint& other) = delete;
    Savepoint(Savepoint&& other) = delete;
    Savepoint& operator=(Savepoint&& other) = delete;
    bool Open(const std::string_view& path);
    void Close();
    void Save();
    void Write(const SavepointVisitor& visitor);
    void Write(const SavepointVisitor& visitor, SavepointID& id, int level);
    void Write(const SavepointVisitor& visitor, int x, int y, int level);
    void Write(const SavepointVisitor& visitor, int x, int y, int z, int level);
    void Read(const SavepointFunction& function);
    void Read(const SavepointEntityFunction& function, int level);
    void Read(const SavepointTile2DFunction& function, int level);
    void Read(const SavepointTile3DFunction& function, int level);
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