#pragma once

#include <savepoint/savepoint.hpp>

#include <cstdint>
#include <string_view>

typedef struct sqlite3 sqlite;
typedef struct sqlite3_stmt sqlite_stmt;

class SavepointDriverSQLite3 : public ISavepointDriver
{
public:
    SavepointDriverSQLite3();
    SavepointStatus Open(const std::string_view& path, SavepointVersion version) override;
    bool IsOpen() const override;
    void Write(const void* data, size_t size) override;
    uint32_t Insert(const void* data, size_t size, int level) override;
    bool Update(const void* data, size_t size, uint32_t id, int level) override;
    void Write(const void* data, size_t size, int x, int y, int level) override;
    void Write(const void* data, size_t size, int x, int y, int z, int level) override;
    void Read(const SavepointReadDataFunction& function) override;
    void Read(const SavepointReadEntityDataFunction& function, int level) override;
    void Read(const SavepointReadTile2DDataFunction& function, int level) override;
    void Read(const SavepointReadTile3DDataFunction& function, int level) override;
    void Read(const SavepointReadLevelFunction& function) override;
    void Delete(uint32_t id) override;
    void Close() override;
    void Save() override;
    void Clear() override;

private:
    SavepointVersion Version;
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
    sqlite3_stmt* ReadLevelsStmt;
    sqlite3_stmt* DeleteEntityStmt;
    sqlite3_stmt* ClearEntitiesStmt;
    sqlite3_stmt* ClearTiles2DStmt;
    sqlite3_stmt* ClearTiles3DStmt;
};
