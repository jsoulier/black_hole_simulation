#include <savepoint.hpp>

#include <cstdint>
#include <cstdio>
#include <format>
#include <string_view>

#include "sqlite3.h"

static SavepointLogFunction logFunction = SavepointDefaultLogFunction;

void SavepointSetLogFunction(const SavepointLogFunction& function)
{
    logFunction = function;
}

void SavepointDefaultLogFunction(const std::string_view& string)
{
    std::fprintf(stderr, "%s\n", string.data());
}

void SavepointLog(const std::string_view& string)
{
    logFunction(string);
}

Savepoint::Savepoint()
    : Handle{nullptr}
    , WriteStmt{nullptr}
    , InsertEntityStmt{nullptr}
    , UpdateEntityStmt{nullptr}
    , WriteTile2DStmt{nullptr}
    , WriteTile3DStmt{nullptr}
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

SavepointStatus Savepoint::Open(const std::string_view& path)
{
    if (sqlite3_open(path.data(), &Handle) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to open database: {}, {}", path, sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    static constexpr const char* kString =
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
    static constexpr const char* kWriteStatusString =
        "INSERT OR REPLACE INTO status (id) VALUES (0);";
    static constexpr const char* kWriteString =
        "INSERT OR REPLACE INTO header (id, data) VALUES (0, ?);";
    static constexpr const char* kInsertEntityString =
        "INSERT INTO entities (level, data) VALUES (?, ?);";
    static constexpr const char* kUpdateEntityString =
        "UPDATE entities SET level = ?, data = ? WHERE id = ?;";
    static constexpr const char* kWriteTile2DString =
        "INSERT OR REPLACE INTO tiles_2d (x, y, level, data) VALUES (?, ?, ?, ?);";
    static constexpr const char* kWriteTile3DString =
        "INSERT OR REPLACE INTO tiles_3d (x, y, z, level, data) VALUES (?, ?, ?, ?, ?);";
    static constexpr const char* kReadStatusString =
        "SELECT 0 FROM status WHERE id = 0;";
    static constexpr const char* kReadString =
        "SELECT data FROM header WHERE id = 0;";
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
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kWriteStatusString, -1, &WriteStatusStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteStatusString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kWriteString, -1, &WriteStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kInsertEntityString, -1, &InsertEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kInsertEntityString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kUpdateEntityString, -1, &UpdateEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kUpdateEntityString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kWriteTile2DString, -1, &WriteTile2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteTile2DString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kWriteTile3DString, -1, &WriteTile3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kWriteTile3DString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kReadStatusString, -1, &ReadStatusStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadStatusString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kReadString, -1, &ReadStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kReadEntitiesString, -1, &ReadEntitiesStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadEntitiesString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kReadTiles2DString, -1, &ReadTiles2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadTiles2DString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kReadTiles3DString, -1, &ReadTiles3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kReadTiles3DString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kDeleteEntityString, -1, &DeleteEntityStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kDeleteEntityString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kClearEntitiesString, -1, &ClearEntitiesStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearEntitiesString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kClearTiles2DString, -1, &ClearTiles2DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearTiles2DString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_prepare_v2(Handle, kClearTiles3DString, -1, &ClearTiles3DStmt, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to prepare kClearTiles3DString: {}", sqlite3_errmsg(Handle)));
        return SavepointStatus::Failed;
    }
    if (sqlite3_exec(Handle, "BEGIN;", nullptr, nullptr, nullptr) != SQLITE_OK)
    {
        SavepointLog(std::format("Failed to begin transaction: {}", sqlite3_errmsg(Handle)));
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
    return status;
}

void Savepoint::Close()
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
}

void Savepoint::Save()
{
    if (!Handle)
    {
        return;
    }
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

void Savepoint::Write(const SavepointVisitor& visitor)
{
    if (!Handle)
    {
        return;
    }
    if (visitor.Writer.empty())
    {
        SavepointLog("Tried to write an empty visitor");
        return;
    }
    const void* data = visitor.Writer.data();
    uint32_t size = visitor.Writer.size();
    sqlite3_bind_blob(WriteStmt, 1, data, size, SQLITE_TRANSIENT);
    if (sqlite3_step(WriteStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to write: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(WriteStmt);
}

void Savepoint::Write(const SavepointVisitor& visitor, SavepointID& id, int level)
{
    if (!Handle)
    {
        return;
    }
    if (visitor.Writer.empty())
    {
        SavepointLog("Tried to write an empty visitor");
        return;
    }
    const void* data = visitor.Writer.data();
    uint32_t size = visitor.Writer.size();
    if (!id)
    {
        sqlite3_bind_int(InsertEntityStmt, 1, level);
        sqlite3_bind_blob(InsertEntityStmt, 2, data, size, SQLITE_TRANSIENT);
        if (sqlite3_step(InsertEntityStmt) != SQLITE_DONE)
        {
            SavepointLog(std::format("Failed to insert entity: {}", sqlite3_errmsg(Handle)));
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
            SavepointLog(std::format("Failed to update entity: {}", sqlite3_errmsg(Handle)));
            id = SavepointID{};
        }
        sqlite3_reset(UpdateEntityStmt);
    }
}

void Savepoint::Write(const SavepointVisitor& visitor, int x, int y, int level)
{
    if (!Handle)
    {
        return;
    }
    if (visitor.Writer.empty())
    {
        SavepointLog("Tried to write an empty visitor");
        return;
    }
    const void* data = visitor.Writer.data();
    uint32_t size = visitor.Writer.size();
    sqlite3_bind_int(WriteTile2DStmt, 1, x);
    sqlite3_bind_int(WriteTile2DStmt, 2, y);
    sqlite3_bind_int(WriteTile2DStmt, 3, level);
    sqlite3_bind_blob(WriteTile2DStmt, 4, data, size, SQLITE_TRANSIENT);
    if (sqlite3_step(WriteTile2DStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to write tile: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(WriteTile2DStmt);
}

void Savepoint::Write(const SavepointVisitor& visitor, int x, int y, int z, int level)
{
    if (!Handle)
    {
        return;
    }
    if (visitor.Writer.empty())
    {
        SavepointLog("Tried to write an empty visitor");
        return;
    }
    const void* data = visitor.Writer.data();
    uint32_t size = visitor.Writer.size();
    sqlite3_bind_int(WriteTile3DStmt, 1, x);
    sqlite3_bind_int(WriteTile3DStmt, 2, y);
    sqlite3_bind_int(WriteTile3DStmt, 3, z);
    sqlite3_bind_int(WriteTile3DStmt, 4, level);
    sqlite3_bind_blob(WriteTile3DStmt, 5, data, size, SQLITE_TRANSIENT);
    if (sqlite3_step(WriteTile3DStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to write tile: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(WriteTile3DStmt);
}

void Savepoint::Read(const SavepointFunction& function)
{
    if (!Handle)
    {
        return;
    }
    SavepointVisitor visitor;
    if (sqlite3_step(ReadStmt) == SQLITE_ROW)
    {
        void* data = const_cast<void*>(sqlite3_column_blob(ReadStmt, 0));
        uint32_t size = sqlite3_column_bytes(ReadStmt, 0);
        visitor.Reset(data, size);
        function(visitor);
    }
    else
    {
        SavepointLog(std::format("Failed to read: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(ReadStmt);
}

void Savepoint::Read(const SavepointEntityFunction& function, int level)
{
    if (!Handle)
    {
        return;
    }
    SavepointVisitor visitor;
    SavepointID id;
    sqlite3_bind_int(ReadEntitiesStmt, 1, level);
    while (sqlite3_step(ReadEntitiesStmt) == SQLITE_ROW)
    {
        id.Value = sqlite3_column_int(ReadEntitiesStmt, 0);
        void* data = const_cast<void*>(sqlite3_column_blob(ReadEntitiesStmt, 1));
        uint32_t size = sqlite3_column_bytes(ReadEntitiesStmt, 1);
        visitor.Reset(data, size);
        function(visitor, id);
    }
    sqlite3_reset(ReadEntitiesStmt);
}

void Savepoint::Read(const SavepointTile2DFunction& function, int level)
{
    if (!Handle)
    {
        return;
    }
    SavepointVisitor visitor;
    sqlite3_bind_int(ReadTiles2DStmt, 1, level);
    while (sqlite3_step(ReadTiles2DStmt) == SQLITE_ROW)
    {
        int x = sqlite3_column_int(ReadTiles2DStmt, 0);
        int y = sqlite3_column_int(ReadTiles2DStmt, 1);
        void* data = const_cast<void*>(sqlite3_column_blob(ReadTiles2DStmt, 2));
        uint32_t size = sqlite3_column_bytes(ReadTiles2DStmt, 2);
        visitor.Reset(data, size);
        function(visitor, x, y);
    }
    sqlite3_reset(ReadTiles2DStmt);
}

void Savepoint::Read(const SavepointTile3DFunction& function, int level)
{
    if (!Handle)
    {
        return;
    }
    SavepointVisitor visitor;
    sqlite3_bind_int(ReadTiles3DStmt, 1, level);
    while (sqlite3_step(ReadTiles3DStmt) == SQLITE_ROW)
    {
        int x = sqlite3_column_int(ReadTiles3DStmt, 0);
        int y = sqlite3_column_int(ReadTiles3DStmt, 1);
        int z = sqlite3_column_int(ReadTiles3DStmt, 2);
        void* data = const_cast<void*>(sqlite3_column_blob(ReadTiles3DStmt, 3));
        uint32_t size = sqlite3_column_bytes(ReadTiles3DStmt, 3);
        visitor.Reset(data, size);
        function(visitor, x, y, z);
    }
    sqlite3_reset(ReadTiles3DStmt);
}

void Savepoint::Delete(const SavepointID id)
{
    if (!Handle)
    {
        return;
    }
    if (!id)
    {
        return;
    }
    sqlite3_bind_int(DeleteEntityStmt, 1, id.Value);
    if (sqlite3_step(DeleteEntityStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to delete entity: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(DeleteEntityStmt);
}

void Savepoint::Clear()
{
    if (!Handle)
    {
        return;
    }
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