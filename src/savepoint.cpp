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

#include <savepoint.hpp>

#include <cstddef>
#include <cstdio>
#include <format>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "sqlite3.h"

/* The current savepoint version */
static constexpr SavepointVersion kVersion{0, 0, 1};

static void DefaultLogFunction(const std::string_view& string)
{
    std::fprintf(stderr, "%s\n", string.data());
}

static SavepointLogFunction logFunction = DefaultLogFunction;

void SavepointSetLogFunction(const SavepointLogFunction& function)
{
    logFunction = function;
}

void SavepointLog(const std::string_view& string)
{
    logFunction(string);
}

/* Allow heterogeneous lookups */
struct Hash
{
    using is_transparent = void;

    size_t operator()(const std::string_view& string) const
    {
        return std::hash<std::string_view>{}(string);
    }

    size_t operator()(const std::string& string) const
    {
        return std::hash<std::string_view>{}(string);
    }
};

using DerivedFunctions = std::unordered_map<std::string, SavepointDerivedFunction, Hash, std::equal_to<>>;

/* For avoiding Static Initialization Order Fiasco */
static DerivedFunctions& GetDerivedFunctions()
{
    static DerivedFunctions functions;
    return functions;
}

void SavepointAddDerivedFunction(const std::string_view& string, const SavepointDerivedFunction& function)
{
    GetDerivedFunctions().emplace(string, function);
}

SavepointDatabase::SavepointDatabase()
    : Version{}
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

/* SQL tables and indexes */
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

/* SQL statements */
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

SavepointStatus SavepointDatabase::Open(const std::string_view& path, SavepointVersion version)
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
    Version = version;
    Visitor.SetApplicationVersion(Version);
    Visitor.SetSavepointVersion(kVersion);
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
    Handle = handle;
    return status;
}

bool SavepointDatabase::IsOpen() const
{
    return Handle != nullptr;
}

SavepointDatabase::~SavepointDatabase()
{
    if (IsOpen())
    {
        SavepointLog("Destroyed savepoint without closing");
    }
}

void SavepointDatabase::Close()
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

void SavepointDatabase::Save()
{
    if (!IsOpen())
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

void SavepointDatabase::Write(SavepointVisitor& visitor)
{
    if (!IsOpen())
    {
        return;
    }
    if (visitor.Empty())
    {
        SavepointLog("Tried to write an empty visitor");
        return;
    }
    visitor.SetApplicationVersion(Version);
    visitor.SetSavepointVersion(kVersion);
    const void* data = visitor.Writer.data();
    size_t size = visitor.Writer.size();
    sqlite3_bind_blob(WriteStmt, 1, data, size, SQLITE_TRANSIENT);
    if (sqlite3_step(WriteStmt) != SQLITE_DONE)
    {
        SavepointLog(std::format("Failed to write: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(WriteStmt);
}

void SavepointDatabase::Write(SavepointVisitor& visitor, SavepointID& id, int level)
{
    if (!IsOpen())
    {
        return;
    }
    if (visitor.Empty())
    {
        SavepointLog("Tried to write an empty visitor");
        return;
    }
    visitor.SetApplicationVersion(Version);
    visitor.SetSavepointVersion(kVersion);
    const void* data = visitor.Writer.data();
    size_t size = visitor.Writer.size();
    /* ID is invalid, insert new entity */
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
    /* ID should be valid, update entity */
    else
    {
        sqlite3_bind_int(UpdateEntityStmt, 1, level);
        sqlite3_bind_blob(UpdateEntityStmt, 2, data, size, SQLITE_TRANSIENT);
        sqlite3_bind_int(UpdateEntityStmt, 3, id.Value);
        if (sqlite3_step(UpdateEntityStmt) != SQLITE_DONE)
        {
            SavepointLog(std::format("Failed to update entity: {}", sqlite3_errmsg(Handle)));
            /* Invalidate ID and try to insert */
            id = SavepointID{};
            Write(visitor, id, level);
        }
        sqlite3_reset(UpdateEntityStmt);
    }
}

void SavepointDatabase::Write(SavepointVisitor& visitor, int x, int y, int level)
{
    if (!IsOpen())
    {
        return;
    }
    if (visitor.Empty())
    {
        SavepointLog("Tried to write an empty visitor");
        return;
    }
    visitor.SetApplicationVersion(Version);
    visitor.SetSavepointVersion(kVersion);
    const void* data = visitor.Writer.data();
    size_t size = visitor.Writer.size();
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

void SavepointDatabase::Write(SavepointVisitor& visitor, int x, int y, int z, int level)
{
    if (!IsOpen())
    {
        return;
    }
    if (visitor.Empty())
    {
        SavepointLog("Tried to write an empty visitor");
        return;
    }
    visitor.SetApplicationVersion(Version);
    visitor.SetSavepointVersion(kVersion);
    const void* data = visitor.Writer.data();
    size_t size = visitor.Writer.size();
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

bool SavepointDatabase::SetBase(SavepointBase* base)
{
    if (!IsOpen())
    {
        return false;
    }
    if (!base)
    {
        SavepointLog("Tried to write null base");
        return false;
    }
    const std::string_view& string = base->SavepointDerivedGetString();
    auto it = GetDerivedFunctions().find(string);
    if (it == GetDerivedFunctions().end())
    {
        SavepointLog(std::format("Failed to find base string: {}", string));
        return false;
    }
    size_t size = string.size();
    Visitor.Reset();
    Visitor(size);
    Visitor(string.data(), size, size);
    Visitor(*base);
    return true;
}

void SavepointDatabase::Write(SavepointBase* base)
{
    if (SetBase(base))
    {
        Write(Visitor);
    }
}

void SavepointDatabase::Write(SavepointBase* base, SavepointID& id, int level)
{
    if (SetBase(base))
    {
        Write(Visitor, id, level);
    }
}

void SavepointDatabase::Write(SavepointBase* base, int x, int y, int level)
{
    if (SetBase(base))
    {
        Write(Visitor, x, y, level);
    }
}

void SavepointDatabase::Write(SavepointBase* base, int x, int y, int z, int level)
{
    if (SetBase(base))
    {
        Write(Visitor, x, y, z, level);
    }
}

void SavepointDatabase::Read(const SavepointReadFunction& function)
{
    if (!IsOpen())
    {
        return;
    }
    if (sqlite3_step(ReadStmt) == SQLITE_ROW)
    {
        void* data = const_cast<void*>(sqlite3_column_blob(ReadStmt, 0));
        size_t size = sqlite3_column_bytes(ReadStmt, 0);
        Visitor.Reset(data, size);
        function(Visitor);
    }
    else
    {
        SavepointLog(std::format("Failed to read: {}", sqlite3_errmsg(Handle)));
    }
    sqlite3_reset(ReadStmt);
}

void SavepointDatabase::Read(const SavepointReadEntityFunction& function, int level)
{
    if (!IsOpen())
    {
        return;
    }
    SavepointID id;
    sqlite3_bind_int(ReadEntitiesStmt, 1, level);
    while (sqlite3_step(ReadEntitiesStmt) == SQLITE_ROW)
    {
        id.Value = sqlite3_column_int(ReadEntitiesStmt, 0);
        void* data = const_cast<void*>(sqlite3_column_blob(ReadEntitiesStmt, 1));
        size_t size = sqlite3_column_bytes(ReadEntitiesStmt, 1);
        Visitor.Reset(data, size);
        function(Visitor, id);
    }
    sqlite3_reset(ReadEntitiesStmt);
}

void SavepointDatabase::Read(const SavepointReadTile2DFunction& function, int level)
{
    if (!IsOpen())
    {
        return;
    }
    sqlite3_bind_int(ReadTiles2DStmt, 1, level);
    while (sqlite3_step(ReadTiles2DStmt) == SQLITE_ROW)
    {
        int x = sqlite3_column_int(ReadTiles2DStmt, 0);
        int y = sqlite3_column_int(ReadTiles2DStmt, 1);
        void* data = const_cast<void*>(sqlite3_column_blob(ReadTiles2DStmt, 2));
        size_t size = sqlite3_column_bytes(ReadTiles2DStmt, 2);
        Visitor.Reset(data, size);
        function(Visitor, x, y);
    }
    sqlite3_reset(ReadTiles2DStmt);
}

void SavepointDatabase::Read(const SavepointReadTile3DFunction& function, int level)
{
    if (!IsOpen())
    {
        return;
    }
    sqlite3_bind_int(ReadTiles3DStmt, 1, level);
    while (sqlite3_step(ReadTiles3DStmt) == SQLITE_ROW)
    {
        int x = sqlite3_column_int(ReadTiles3DStmt, 0);
        int y = sqlite3_column_int(ReadTiles3DStmt, 1);
        int z = sqlite3_column_int(ReadTiles3DStmt, 2);
        void* data = const_cast<void*>(sqlite3_column_blob(ReadTiles3DStmt, 3));
        size_t size = sqlite3_column_bytes(ReadTiles3DStmt, 3);
        Visitor.Reset(data, size);
        function(Visitor, x, y, z);
    }
    sqlite3_reset(ReadTiles3DStmt);
}

SavepointBase* SavepointDatabase::GetBase(SavepointVisitor& visitor)
{
    std::string string;
    visitor(string);
    auto it = GetDerivedFunctions().find(string);
    if (it == GetDerivedFunctions().end())
    {
        SavepointLog(std::format("Missing base function: {}, {} -> {}", string, visitor.Version.GetString(), Version.GetString()));
        return nullptr;
    }
    SavepointBase* base = it->second();
    if (!base)
    {
        SavepointLog(std::format("Failed to allocate base: {}", string));
        return nullptr;
    }
    visitor(*base);
    return base;
}

void SavepointDatabase::Read(const SavepointReadBaseFunction& function)
{
    Read([this, &function](SavepointVisitor& visitor)
    {
        SavepointBase* base = GetBase(visitor);
        if (base)
        {
            function(base);
        }
    });
}

void SavepointDatabase::Read(const SavepointReadBaseEntityFunction& function, int level)
{
    Read([this, &function](SavepointVisitor& visitor, SavepointID id)
    {
        SavepointBase* base = GetBase(visitor);
        if (base)
        {
            function(base, id);
        }
    }, level);
}

void SavepointDatabase::Read(const SavepointReadBaseTile2DFunction& function, int level)
{
    Read([this, &function](SavepointVisitor& visitor, int x, int y)
    {
        SavepointBase* base = GetBase(visitor);
        if (base)
        {
            function(base, x, y);
        }
    }, level);
}

void SavepointDatabase::Read(const SavepointReadBaseTile3DFunction& function, int level)
{
    Read([this, &function](SavepointVisitor& visitor, int x, int y, int z)
    {
        SavepointBase* base = GetBase(visitor);
        if (base)
        {
            function(base, x, y, z);
        }
    }, level);
}

void SavepointDatabase::Delete(const SavepointID id)
{
    if (!IsOpen())
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

void SavepointDatabase::Clear()
{
    if (!IsOpen())
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