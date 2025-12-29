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

#include <savepoint/fwd.hpp>

#include <cstring>
#include <functional>
#include <string_view>

enum class SavepointDriver
{
    Null,
    Sqlite3,
};

enum class SavepointStatus
{
    Failed,
    Existing,
    New,
};

using SavepointReadVisitorFunction = std::function<void(SavepointVisitor& visitor)>;
using SavepointReadVisitorEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;
using SavepointReadVisitorTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;
using SavepointReadVisitorTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;

using SavepointReadBaseFunction = std::function<void(SavepointBase* base)>;
using SavepointReadBaseEntityFunction = std::function<void(SavepointBase* base, SavepointID id)>;
using SavepointReadBaseTile2DFunction = std::function<void(SavepointBase* base, int x, int y)>;
using SavepointReadBaseTile3DFunction = std::function<void(SavepointBase* base, int x, int y, int z)>;

class ISavepointDriver
{
public:
    virtual SavepointStatus Open(const std::string_view& path, SavepointVersion version) = 0;
    virtual bool IsOpen() const = 0;
    virtual void Close() = 0;
    virtual void Save() = 0;
    virtual void Write(SavepointVisitor& visitor) = 0;
    virtual void Write(SavepointVisitor& visitor, SavepointID& id, int level) = 0;
    virtual void Write(SavepointVisitor& visitor, int x, int y, int level) = 0;
    virtual void Write(SavepointVisitor& visitor, int x, int y, int z, int level) = 0;
    virtual void Write(SavepointBase* base) = 0;
    virtual void Write(SavepointBase* base, SavepointID& id, int level) = 0;
    virtual void Write(SavepointBase* base, int x, int y, int level) = 0;
    virtual void Write(SavepointBase* base, int x, int y, int z, int level) = 0;
    virtual void Read(const SavepointReadVisitorFunction& function) = 0;
    virtual void Read(const SavepointReadVisitorEntityFunction& function, int level) = 0;
    virtual void Read(const SavepointReadVisitorTile2DFunction& function, int level) = 0;
    virtual void Read(const SavepointReadVisitorTile3DFunction& function, int level) = 0;
    virtual void Read(const SavepointReadBaseFunction& function) = 0;
    virtual void Read(const SavepointReadBaseEntityFunction& function, int level) = 0;
    virtual void Read(const SavepointReadBaseTile2DFunction& function, int level) = 0;
    virtual void Read(const SavepointReadBaseTile3DFunction& function, int level) = 0;
    virtual void Delete(const SavepointID id) = 0;
    virtual void Clear() = 0;
};