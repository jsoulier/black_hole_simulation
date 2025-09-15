#include <cstdio>
#include <filesystem>
#include <format>
#include <savepoint.hpp>
#include <string>
#include "sqlite3.h"

static SavepointLogFunction logFunction = SavepointDefaultLogFunction;

void SavepointSetLogFunction(const SavepointLogFunction& function)
{
    logFunction = function;
}

void SavepointDefaultLogFunction(const std::string& string)
{
    std::fprintf(stderr, "%s\n", string.data());
}

void SavepointLog(const std::string& string)
{
    logFunction(string);
}

Savepoint::Savepoint()
    : Handle{nullptr}
{
}

bool Savepoint::Open(const std::filesystem::path& path)
{
    if (sqlite3_open(path.string().data(), &Handle) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to open database: {}, {}", path.string(), sqlite3_errmsg(Handle)));
        return false;
    }
    static constexpr const char* kString =
        "CREATE TABLE IF NOT EXISTS header ("
        "    data BLOB NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS entities ("
        "    id INT PRIMARY KEY,"
        "    level INT NOT NULL,"
        "    data BLOB NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS tiles_2d ("
        "    x INT NOT NULL,"
        "    y INT NOT NULL,"
        "    level INT NOT NULL,"
        "    data BLOB NOT NULL,"
        "    PRIMARY KEY (x, y)"
        ");"
        "CREATE TABLE IF NOT EXISTS tiles_3d ("
        "    x INT NOT NULL,"
        "    y INT NOT NULL,"
        "    z INT NOT NULL,"
        "    level INT NOT NULL,"
        "    data BLOB NOT NULL,"
        "    PRIMARY KEY (x, y, z)"
        ");"
        "CREATE INDEX IF NOT EXISTS entities_index ON entities (level);"
        "CREATE INDEX IF NOT EXISTS tiles_2d_index ON tiles_2d (level);"
        "CREATE INDEX IF NOT EXISTS tiles_3d_index ON tiles_3d (level);"
        " ";
    static constexpr const char* kWriteHeaderString =
        "INSERT OR REPLACE INTO header (data) VALUES (?);";
    static constexpr const char* kInsertEntityString =
        "INSERT INTO entities (id, level, data) VALUES (?, ?, ?);";
    static constexpr const char* kUpdateEntityString =
        "UPDATE entities SET level = ?, data = ? WHERE id = ?;";
    static constexpr const char* kWriteTile2DString =
        "INSERT OR REPLACE INTO tiles_2d (x, y, level, data) VALUES (?, ?, ?, ?);";
    static constexpr const char* kWriteTile3DString =
        "INSERT OR REPLACE INTO tiles_3d (x, y, z, level, data) VALUES (?, ?, ?, ?, ?);";
    static constexpr const char* kReadHeaderString =
        "SELECT data FROM header LIMIT 1;";
    static constexpr const char* kReadEntitiesString =
        "SELECT id, data FROM entities WHERE level = ?;";
    static constexpr const char* kReadTiles2DString =
        "SELECT x, y, data FROM tiles_2d WHERE level = ?;";
    static constexpr const char* kReadTiles3DString =
        "SELECT x, y, z, data FROM tiles_3d WHERE level = ?;";
    static constexpr const char* kDeleteEntityString =
        "DELETE FROM entities WHERE id = ?;";
    if (sqlite3_exec(Handle, kString, nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to execute kString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kWriteHeaderString, -1, &WriteStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteHeaderString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kInsertEntityString, -1, &InsertEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kInsertEntityString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kUpdateEntityString, -1, &UpdateEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kUpdateEntityString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kWriteTile2DString, -1, &WriteTile2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteTile2DString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kWriteTile3DString, -1, &WriteTile3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteTile3DString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kReadHeaderString, -1, &ReadStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadHeaderString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kReadEntitiesString, -1, &ReadEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadEntitiesString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kReadTiles2DString, -1, &ReadTile2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadTiles2DString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kReadTiles3DString, -1, &ReadTile3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadTiles3DString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kDeleteEntityString, -1, &DeleteEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kDeleteEntityString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_exec(Handle, "BEGIN;", nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to begin transaction: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    return true;
}

void Savepoint::Close()
{
    sqlite3_finalize(WriteStmt);
    sqlite3_finalize(InsertEntityStmt);
    sqlite3_finalize(UpdateEntityStmt);
    sqlite3_finalize(WriteTile2DStmt);
    sqlite3_finalize(WriteTile3DStmt);
    sqlite3_finalize(ReadStmt);
    sqlite3_finalize(ReadEntityStmt);
    sqlite3_finalize(ReadTile2DStmt);
    sqlite3_finalize(ReadTile3DStmt);
    sqlite3_finalize(DeleteEntityStmt);
    sqlite3_close(Handle);
}

void Savepoint::Save()
{
    if (sqlite3_exec(Handle, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to end transaction: {}", sqlite3_errmsg(Handle)));
    }
    if (sqlite3_exec(Handle, "BEGIN;", nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to begin transaction: {}", sqlite3_errmsg(Handle)));
    }
}

void Savepoint::Write(const SavepointArchive& archive)
{
}

void Savepoint::Write(const SavepointArchive& archive, SavepointID& id, int level)
{
}

void Savepoint::Write(const SavepointArchive& archive, int x, int y, int level)
{
}

void Savepoint::Write(const SavepointArchive& archive, int x, int y, int z, int level)
{
}

SavepointArchive Savepoint::Read()
{
    return {};
}

void Savepoint::Read(const SavepointEntityFunction& function, int level)
{
}

void Savepoint::Read(const SavepointTile2DFunction& function, int level)
{
}

void Savepoint::Read(const SavepointTile3DFunction& function, int level)
{
}

void Savepoint::Delete(const SavepointID id)
{
}