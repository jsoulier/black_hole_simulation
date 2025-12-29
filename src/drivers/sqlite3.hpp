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

#pragma once

#include <savepoint/base.hpp>
#include <savepoint/driver.hpp>
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
    void Write(SavepointVisitor& visitor, SavepointID& id, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int level) override;
    void Write(SavepointVisitor& visitor, int x, int y, int z, int level) override;
    void Write(SavepointBase* base) override;
    void Write(SavepointBase* base, SavepointID& id, int level) override;
    void Write(SavepointBase* base, int x, int y, int level) override;
    void Write(SavepointBase* base, int x, int y, int z, int level) override;
    void Read(const SavepointReadVisitorFunction& function) override;
    void Read(const SavepointReadVisitorEntityFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile2DFunction& function, int level) override;
    void Read(const SavepointReadVisitorTile3DFunction& function, int level) override;
    void Read(const SavepointReadBaseFunction& function) override;
    void Read(const SavepointReadBaseEntityFunction& function, int level) override;
    void Read(const SavepointReadBaseTile2DFunction& function, int level) override;
    void Read(const SavepointReadBaseTile3DFunction& function, int level) override;
    void Delete(const SavepointID id) override;
    void Close() override;
    void Save() override;
    void Clear() override;

private:
    bool SetBase(SavepointBase* base);
    SavepointBase* GetBase(SavepointVisitor& visitor);

    SavepointVersion ApplicationVersion;
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