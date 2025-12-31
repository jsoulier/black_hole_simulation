#pragma once

#include <savepoint/fwd.hpp>
#include <savepoint/id.hpp>
#include <savepoint/status.hpp>

#include <functional>
#include <string_view>

/**
 * @brief The implementation for Savepoint's file operations
 */
enum class SavepointDriver
{
    Sqlite3, /**< Backed by sqlite3 */
    Null,    /**< Noop */
};

/** @cond INTERNAL */

using SavepointReadVisitorFunction = std::function<void(SavepointVisitor& visitor)>;
using SavepointReadVisitorEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;
using SavepointReadVisitorTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;
using SavepointReadVisitorTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;

class ISavepointDriver
{
public:
    virtual SavepointStatus Open(const std::string_view& path, SavepointVersion version) = 0;
    virtual bool IsOpen() const = 0;
    virtual void Write(SavepointVisitor& visitor) = 0;
    virtual void Write(SavepointVisitor& visitor, SavepointID& id, int level) = 0;
    virtual void Write(SavepointVisitor& visitor, int x, int y, int level) = 0;
    virtual void Write(SavepointVisitor& visitor, int x, int y, int z, int level) = 0;
    virtual void Read(const SavepointReadVisitorFunction& function) = 0;
    virtual void Read(const SavepointReadVisitorEntityFunction& function, int level) = 0;
    virtual void Read(const SavepointReadVisitorTile2DFunction& function, int level) = 0;
    virtual void Read(const SavepointReadVisitorTile3DFunction& function, int level) = 0;
    virtual void Delete(const SavepointID id) = 0;
    virtual void Close() = 0;
    virtual void Save() = 0;
    virtual void Clear() = 0;
};

/** @endcond */
