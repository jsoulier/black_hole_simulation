#pragma once

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

class Savepoint;
class SavepointID;
class SavepointSerializer;
class SavepointVersion;

using SavepointLogFunction = std::function<void(const std::string& string)>;

void SavepointSetLogFunction(const SavepointLogFunction& function);
void SavepointDefaultLogFunction(const std::string& string);
void SavepointLog(const std::string& string);

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

class SavepointSerializer
{
private:
    friend class Savepoint;

public:
    SavepointSerializer()
        : Version{}
        , Writer{}
        , Reader{}
        , Offset{}
    {
    }

    template<typename T, typename... Args>
    void Visit(T& item, SavepointVersion version, Args&&... args)
    {
        if (!Reader.empty())
        {
            if (Version < version)
            {
                item = T{std::forward<Args>(args)...};
                return;
            }
            if (Offset + sizeof(T) > Reader.size())
            {
                SavepointLog(std::format("Failed to read from serializer: {} -> {}", Version.GetString(), version.GetString()));
                item = T{std::forward<Args>(args)...};
                return;
            }
            std::memcpy(std::addressof(item), Reader.size() + Offset, sizeof(T));
            Offset += sizeof(T);
        }
        else
        {
            Writer.resize(Writer.size() + sizeof(T));
            std::memcpy(Writer.data() + Writer.size() - sizeof(T), std::addressof(item), sizeof(T));
        }
    }

private:
    SavepointVersion Version;
    std::vector<uint8_t> Writer;
    std::span<uint8_t> Reader;
    uint32_t Offset;
};

using SavepointEntityFunction = std::function<void(SavepointSerializer& serializer, SavepointID id)>;
using SavepointTile2DFunction = std::function<void(SavepointSerializer& serializer, int x, int y)>;
using SavepointTile3DFunction = std::function<void(SavepointSerializer& serializer, int x, int y, int z)>;

class Savepoint
{
public:
    Savepoint();
    Savepoint(const Savepoint& other) = delete;
    Savepoint& operator=(const Savepoint& other) = delete;
    Savepoint(Savepoint&& other) = delete;
    Savepoint& operator=(Savepoint&& other) = delete;
    bool Open(const std::filesystem::path& path);
    void Close(bool save = true);
    void Save();
    void Write(const SavepointSerializer& serializer);
    void Write(const SavepointSerializer& serializer, SavepointID& id, int level = 0);
    void Write(const SavepointSerializer& serializer, int x, int y, int level = 0);
    void Write(const SavepointSerializer& serializer, int x, int y, int z, int level = 0);
    SavepointSerializer Read();
    void Read(const SavepointEntityFunction& function, int level = 0);
    void Read(const SavepointTile2DFunction& function, int level = 0);
    void Read(const SavepointTile3DFunction& function, int level = 0);
    void Delete(const SavepointID id);

private:
    typedef struct sqlite3 sqlite;
    typedef struct sqlite3_stmt sqlite_stmt;
    sqlite3* Handle;
};