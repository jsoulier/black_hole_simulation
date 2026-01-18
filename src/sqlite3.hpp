#pragma once

#ifdef SAVEPOINT_SQLITE3

#include <savepoint/driver.hpp>
#include <savepoint/id.hpp>
#include <savepoint/status.hpp>
#include <savepoint/version.hpp>
#include <savepoint/visitor.hpp>

#include <string_view>

typedef struct sqlite3 sqlite;
typedef struct sqlite3_stmt sqlite_stmt;

class SavepointDriverSqlite3 : public ISavepointDriver
{
public:
    SavepointDriverSqlite3();
    SavepointStatus Open(const std::string_view& path, SavepointVersion version) override;
    bool IsOpen() const override;
    void Write(SavepointVisitor& visitor) override;
    SavepointID Insert(SavepointVisitor& visitor, int level) override;
    SavepointID Update(SavepointVisitor& visitor, SavepointID id, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level) override;
    void Read(const SavepointReadVisitorFunction& function) override;
    void Read(const SavepointReadVisitorEntityFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile2DFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile3DFunction& function, int level) override;
    void Delete(const SavepointID id) override;
    void Close() override;
    void Save() override;
    void Clear() override;

private:
    SavepointVersion Version;
    SavepointVisitor Visitor;
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
    sqlite3_stmt* DeleteEntityStmt;
    sqlite3_stmt* ClearEntitiesStmt;
    sqlite3_stmt* ClearTiles2DStmt;
    sqlite3_stmt* ClearTiles3DStmt;
};

#endif