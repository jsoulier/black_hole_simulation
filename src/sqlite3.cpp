#ifdef SAVEPOINT_SQLITE3

#include <sqlite3.h>

#include <savepoint/driver.hpp>
#include <savepoint/id.hpp>
#include <savepoint/log.hpp>
#include <savepoint/status.hpp>
#include <savepoint/version.hpp>
#include <savepoint/visitor.hpp>

#include <cstddef>
#include <format>
#include <string_view>

#include "sqlite3.hpp"

static constexpr const char* kSQL =
    "CREATE TABLE IF NOT EXISTS status ("
    "    id INTEGER PRIMARY KEY"
    ");"
    "CREATE TABLE IF NOT EXISTS header ("
    "    id INTEGER PRIMARY KEY,"
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
    "    PRIMARY KEY (x, y, level)"
    ");"
    "CREATE TABLE IF NOT EXISTS tiles_3d ("
    "    x INTEGER NOT NULL,"
    "    y INTEGER NOT NULL,"
    "    z INTEGER NOT NULL,"
    "    level INTEGER NOT NULL,"
    "    data BLOB NOT NULL,"
    "    PRIMARY KEY (x, y, z, level)"
    ");"
    "CREATE INDEX IF NOT EXISTS entities_index ON entities (level);"
    "CREATE INDEX IF NOT EXISTS tiles_2d_index ON tiles_2d (level);"
    "CREATE INDEX IF NOT EXISTS tiles_3d_index ON tiles_3d (level);"
    " ";

static constexpr const char* kWriteStatusSQL =
    "INSERT OR REPLACE INTO status (id) VALUES (0);";
static constexpr const char* kWriteSQL =
    "INSERT OR REPLACE INTO header (id, data) VALUES (0, ?);";
static constexpr const char* kInsertEntitySQL =
    "INSERT INTO entities (level, data) VALUES (?, ?);";
static constexpr const char* kUpdateEntitySQL =
    "UPDATE entities SET level = ?, data = ? WHERE id = ?;";
static constexpr const char* kWriteTile2DSQL =
    "INSERT OR REPLACE INTO tiles_2d (x, y, level, data) VALUES (?, ?, ?, ?);";
static constexpr const char* kWriteTile3DSQL =
    "INSERT OR REPLACE INTO tiles_3d (x, y, z, level, data) VALUES (?, ?, ?, ?, ?);";
static constexpr const char* kReadStatusSQL =
    "SELECT 0 FROM status WHERE id = 0;";
static constexpr const char* kReadSQL =
    "SELECT data FROM header WHERE id = 0;";
static constexpr const char* kReadEntitiesSQL =
    "SELECT id, data FROM entities WHERE level = ?;";
static constexpr const char* kReadTiles2DSQL =
    "SELECT x, y, data FROM tiles_2d WHERE level = ?;";
static constexpr const char* kReadTiles3DSQL =
    "SELECT x, y, z, data FROM tiles_3d WHERE level = ?;";
static constexpr const char* kDeleteEntitySQL =
    "DELETE FROM entities WHERE id = ?;";
static constexpr const char* kClearEntitiesSQL =
    "DELETE FROM entities;";
static constexpr const char* kClearTiles2DSQL =
    "DELETE FROM tiles_2d;";
static constexpr const char* kClearTiles3DSQL =
    "DELETE FROM tiles_3d;";

SavepointDriverSqlite3::SavepointDriverSqlite3()
    : ISavepointDriver()
    , Version{}
    , Visitor{}
    , Handle{nullptr}
    , WriteStatusStmt{nullptr}
    , WriteStmt{nullptr}
    , InsertEntityStmt{nullptr}
    , UpdateEntityStmt{nullptr}
    , WriteTile2DStmt{nullptr}
    , WriteTile3DStmt{nullptr}
    , ReadStatusStmt{nullptr}
    , ReadStmt{nullptr}
    , ReadEntitiesStmt{nullptr}
    , ReadTiles2DStmt{nullptr}
    , ReadTiles3DStmt{nullptr}
    , DeleteEntityStmt{nullptr}
    , ClearEntitiesStmt{nullptr}
    , ClearTiles2DStmt{nullptr}
    , ClearTiles3DStmt{nullptr}
{
}

SavepointStatus SavepointDriverSqlite3::Open(const std::string_view& path, SavepointVersion version)
{
    sqlite3* handle = nullptr;
    if (sqlite3_open(path.data(), &handle) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to open database: {}, {}", path, sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_exec(handle, kSQL, nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to execute kSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kWriteStatusSQL, -1, &WriteStatusStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteStatusSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kWriteSQL, -1, &WriteStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kInsertEntitySQL, -1, &InsertEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kInsertEntitySQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kUpdateEntitySQL, -1, &UpdateEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kUpdateEntitySQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kWriteTile2DSQL, -1, &WriteTile2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteTile2DSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kWriteTile3DSQL, -1, &WriteTile3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteTile3DSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kReadStatusSQL, -1, &ReadStatusStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadStatusSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kReadSQL, -1, &ReadStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kReadEntitiesSQL, -1, &ReadEntitiesStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadEntitiesSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kReadTiles2DSQL, -1, &ReadTiles2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadTiles2DSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kReadTiles3DSQL, -1, &ReadTiles3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadTiles3DSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kDeleteEntitySQL, -1, &DeleteEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kDeleteEntitySQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kClearEntitiesSQL, -1, &ClearEntitiesStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearEntitiesSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kClearTiles2DSQL, -1, &ClearTiles2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearTiles2DSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(handle, kClearTiles3DSQL, -1, &ClearTiles3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearTiles3DSQL: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_exec(handle, "BEGIN;", nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to begin transaction: {}", sqlite3_errmsg(handle)));
        return SavepointStatus::Failed;
    }
    SavepointStatus status;
    if (sqlite3_step(ReadStatusStmt) == SQLITE_ROW)
    {
        status = SavepointStatus::Existing;
    }
    else
    {
        status = SavepointStatus::New;
    }
    sqlite3_reset(ReadStatusStmt);
    Version = version;
    Handle = handle;
    return status;
}

bool SavepointDriverSqlite3::IsOpen() const
{
    return Handle != nullptr;
}

void SavepointDriverSqlite3::Write(SavepointVisitor& visitor)
{
    const void* data = visitor.GetData();
    size_t size = visitor.GetSize();
    sqlite3_bind_blob(WriteStmt, 1, data, size, SQLITE_TRANSIENT);
    if (sqlite3_step(WriteStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to write: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(WriteStmt);
}

SavepointID SavepointDriverSqlite3::Insert(SavepointVisitor& visitor, int level)
{
    SavepointID id;
    const void* data = visitor.GetData();
    size_t size = visitor.GetSize();
    sqlite3_bind_int(InsertEntityStmt, 1, level);
    sqlite3_bind_blob(InsertEntityStmt, 2, data, size, SQLITE_TRANSIENT);
    if (sqlite3_step(InsertEntityStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to insert entity: {}", sqlite3_errmsg(Handle)));
    }
    else
    {
        id.SetValue(sqlite3_last_insert_rowid(Handle));
    }
    sqlite3_reset(InsertEntityStmt);
    return id;
}

SavepointID SavepointDriverSqlite3::Update(SavepointVisitor& visitor, SavepointID id, int level)
{
    const void* data = visitor.GetData();
    size_t size = visitor.GetSize();
    sqlite3_bind_int(UpdateEntityStmt, 1, level);
    sqlite3_bind_blob(UpdateEntityStmt, 2, data, size, SQLITE_TRANSIENT);
    sqlite3_bind_int(UpdateEntityStmt, 3, id.GetValue());
    if (sqlite3_step(UpdateEntityStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to update entity: {}", sqlite3_errmsg(Handle)));
        id = SavepointID{};
    }
    sqlite3_reset(UpdateEntityStmt);
    return id;
}

void SavepointDriverSqlite3::Write(SavepointVisitor& visitor, int x, int y, int level)
{
    const void* data = visitor.GetData();
    size_t size = visitor.GetSize();
    sqlite3_bind_int(WriteTile2DStmt, 1, x);
    sqlite3_bind_int(WriteTile2DStmt, 2, y);
    sqlite3_bind_int(WriteTile2DStmt, 3, level);
    sqlite3_bind_blob(WriteTile2DStmt, 4, data, size, SQLITE_TRANSIENT);
    if (sqlite3_step(WriteTile2DStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to write tile: {}, {}, {}", x, y, sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(WriteTile2DStmt);
}

void SavepointDriverSqlite3::Write(SavepointVisitor& visitor, int x, int y, int z, int level)
{
    const void* data = visitor.GetData();
    size_t size = visitor.GetSize();
    sqlite3_bind_int(WriteTile3DStmt, 1, x);
    sqlite3_bind_int(WriteTile3DStmt, 2, y);
    sqlite3_bind_int(WriteTile3DStmt, 3, z);
    sqlite3_bind_int(WriteTile3DStmt, 4, level);
    sqlite3_bind_blob(WriteTile3DStmt, 5, data, size, SQLITE_TRANSIENT);
    if (sqlite3_step(WriteTile3DStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to write tile: {}, {}, {}, {}", x, y, z, sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(WriteTile3DStmt);
}

void SavepointDriverSqlite3::Read(const SavepointReadVisitorFunction& function)
{
    if (sqlite3_step(ReadStmt) == SQLITE_ROW)
    {
        const void* data = sqlite3_column_blob(ReadStmt, 0);
        size_t size = sqlite3_column_bytes(ReadStmt, 0);
        Visitor.BeginReading(data, size);
        function(Visitor);
    }
    else
    {
        SavepointLog(std::format("Failed to read: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(ReadStmt);
}

void SavepointDriverSqlite3::Read(const SavepointReadVisitorEntityFunction& function, int level)
{
    SavepointID id;
    sqlite3_bind_int(ReadEntitiesStmt, 1, level);
    while (sqlite3_step(ReadEntitiesStmt) == SQLITE_ROW)
    {
        id.SetValue(sqlite3_column_int(ReadEntitiesStmt, 0));
        const void* data = sqlite3_column_blob(ReadEntitiesStmt, 1);
        size_t size = sqlite3_column_bytes(ReadEntitiesStmt, 1);
        Visitor.BeginReading(data, size);
        function(Visitor, id);
    }
    sqlite3_reset(ReadEntitiesStmt);
}

void SavepointDriverSqlite3::Read(const SavepointReadVisitorTile2DFunction& function, int level)
{
    sqlite3_bind_int(ReadTiles2DStmt, 1, level);
    while (sqlite3_step(ReadTiles2DStmt) == SQLITE_ROW)
    {
        int x = sqlite3_column_int(ReadTiles2DStmt, 0);
        int y = sqlite3_column_int(ReadTiles2DStmt, 1);
        const void* data = sqlite3_column_blob(ReadTiles2DStmt, 2);
        size_t size = sqlite3_column_bytes(ReadTiles2DStmt, 2);
        Visitor.BeginReading(data, size);
        function(Visitor, x, y);
    }
    sqlite3_reset(ReadTiles2DStmt);
}

void SavepointDriverSqlite3::Read(const SavepointReadVisitorTile3DFunction& function, int level)
{
    sqlite3_bind_int(ReadTiles3DStmt, 1, level);
    while (sqlite3_step(ReadTiles3DStmt) == SQLITE_ROW)
    {
        int x = sqlite3_column_int(ReadTiles3DStmt, 0);
        int y = sqlite3_column_int(ReadTiles3DStmt, 1);
        int z = sqlite3_column_int(ReadTiles3DStmt, 2);
        const void* data = sqlite3_column_blob(ReadTiles3DStmt, 3);
        size_t size = sqlite3_column_bytes(ReadTiles3DStmt, 3);
        Visitor.BeginReading(data, size);
        function(Visitor, x, y, z);
    }
    sqlite3_reset(ReadTiles3DStmt);
}

void SavepointDriverSqlite3::Delete(const SavepointID id)
{
    sqlite3_bind_int(DeleteEntityStmt, 1, id.GetValue());
    if (sqlite3_step(DeleteEntityStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to delete entity: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(DeleteEntityStmt);
}

void SavepointDriverSqlite3::Close()
{
    sqlite3_finalize(WriteStatusStmt);
    sqlite3_finalize(WriteStmt);
    sqlite3_finalize(InsertEntityStmt);
    sqlite3_finalize(UpdateEntityStmt);
    sqlite3_finalize(WriteTile2DStmt);
    sqlite3_finalize(WriteTile3DStmt);
    sqlite3_finalize(ReadStatusStmt);
    sqlite3_finalize(ReadStmt);
    sqlite3_finalize(ReadEntitiesStmt);
    sqlite3_finalize(ReadTiles2DStmt);
    sqlite3_finalize(ReadTiles3DStmt);
    sqlite3_finalize(DeleteEntityStmt);
    sqlite3_finalize(ClearEntitiesStmt);
    sqlite3_finalize(ClearTiles2DStmt);
    sqlite3_finalize(ClearTiles3DStmt);
    sqlite3_close(Handle);
    Handle = nullptr;
}

void SavepointDriverSqlite3::Save()
{
    if (sqlite3_step(WriteStatusStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to write status: {}", sqlite3_errmsg(Handle)));
    }
    if (sqlite3_exec(Handle, "COMMIT;", nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to end transaction: {}", sqlite3_errmsg(Handle)));
    }
    if (sqlite3_exec(Handle, "BEGIN;", nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to begin transaction: {}", sqlite3_errmsg(Handle)));
    }
}

void SavepointDriverSqlite3::Clear()
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

#endif