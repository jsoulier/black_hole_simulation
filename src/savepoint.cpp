#include <cstdio>
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

bool Savepoint::Open(const std::string& path)
{
    if (sqlite3_open(path.data(), &Handle) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to open database: {}, {}", path, sqlite3_errmsg(Handle)));
        return false;
    }
    static constexpr const char* kString =
        "CREATE TABLE IF NOT EXISTS header ("
        "    data BLOB NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS entities ("
        "    id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "    level INTEGER NOT NULL,"
        "    data BLOB NOT NULL"
        ");"
        "CREATE TABLE IF NOT EXISTS tiles_2d ("
        "    x INTEGER NOT NULL,"
        "    y INTEGER NOT NULL,"
        "    level INTEGER NOT NULL,"
        "    data BLOB NOT NULL,"
        "    PRIMARY KEY (x, y)"
        ");"
        "CREATE TABLE IF NOT EXISTS tiles_3d ("
        "    x INTEGER NOT NULL,"
        "    y INTEGER NOT NULL,"
        "    z INTEGER NOT NULL,"
        "    level INTEGER NOT NULL,"
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
        "INSERT INTO entities (level, data) VALUES (?, ?);";
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
    static constexpr const char* kClearEntitiesString =
        "DELETE FROM entities;";
    static constexpr const char* kClearTiles2DString =
        "DELETE FROM tiles_2d;";
    static constexpr const char* kClearTiles3DString =
        "DELETE FROM tiles_3d;";
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
    if (sqlite3_prepare_v2(Handle, kReadEntitiesString, -1, &ReadEntitiesStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadEntitiesString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kReadTiles2DString, -1, &ReadTiles2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadTiles2DString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kReadTiles3DString, -1, &ReadTiles3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadTiles3DString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kDeleteEntityString, -1, &DeleteEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kDeleteEntityString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kClearEntitiesString, -1, &ClearEntitiesStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearEntitiesString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kClearTiles2DString, -1, &ClearTiles2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearTiles2DString: {}", sqlite3_errmsg(Handle)));
        return false;
    }
    if (sqlite3_prepare_v2(Handle, kClearTiles3DString, -1, &ClearTiles3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearTiles3DString: {}", sqlite3_errmsg(Handle)));
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
    sqlite3_finalize(ReadEntitiesStmt);
    sqlite3_finalize(ReadTiles2DStmt);
    sqlite3_finalize(ReadTiles3DStmt);
    sqlite3_finalize(DeleteEntityStmt);
    sqlite3_finalize(ClearEntitiesStmt);
    sqlite3_finalize(ClearTiles2DStmt);
    sqlite3_finalize(ClearTiles3DStmt);
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
    if (archive.Writer.empty())
    {
        SavepointLog("Tried to write an empty archive");
        return;
    }
    const void* data = archive.Writer.data();
    uint32_t size = archive.Writer.size();
    if (!id)
    {
        sqlite3_bind_int(InsertEntityStmt, 1, level);
        sqlite3_bind_blob(InsertEntityStmt, 2, data, size, SQLITE_TRANSIENT);
        if (sqlite3_step(InsertEntityStmt) != SQLITE_DONE)
        {
            SavepointLog(std::format("Failed to insert new entity: {}", sqlite3_errmsg(Handle)));
        }
        else
        {
            id.Value = sqlite3_last_insert_rowid(Handle);
        }
        sqlite3_reset(InsertEntityStmt);
    }
    else
    {
        sqlite3_bind_int(UpdateEntityStmt, 1, level);
        sqlite3_bind_blob(UpdateEntityStmt, 2, data, size, SQLITE_TRANSIENT);
        sqlite3_bind_int(UpdateEntityStmt, 3, id.Value);
        if (sqlite3_step(UpdateEntityStmt) != SQLITE_DONE)
        {
            SavepointLog(std::format("Failed to update existing entity: {}", sqlite3_errmsg(Handle)));
            id = SavepointID{};
        }
        sqlite3_reset(UpdateEntityStmt);
    }
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
    SavepointArchive archive;
    SavepointID id;
    sqlite3_bind_int(ReadEntitiesStmt, 1, level);
    while (sqlite3_step(ReadEntitiesStmt) == SQLITE_ROW)
    {
        id.Value = sqlite3_column_int(ReadEntitiesStmt, 0);
        void* data = const_cast<void*>(sqlite3_column_blob(ReadEntitiesStmt, 1));
        uint32_t size = sqlite3_column_bytes(ReadEntitiesStmt, 1);
        archive.Reader = {static_cast<uint8_t*>(data), size};
        archive(archive.Version);
        function(archive, id);
    }
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

void Savepoint::Clear()
{
    if (sqlite3_step(ClearEntitiesStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to clear entities: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(ClearEntitiesStmt);
    if (sqlite3_step(ClearTiles2DStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to clear tiles 2d: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(ClearTiles2DStmt);
    if (sqlite3_step(ClearTiles3DStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to clear tiles 3d: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(ClearTiles3DStmt);
}