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
#include <savepoint/id.hpp>
#include <savepoint/status.hpp>

#include <cstring>
#include <functional>
#include <string_view>

/**
 * @brief 
 * 
 */
enum class SavepointDriver
{
    Sqlite3, /**< Backed by sqlite3 */
    Null,    /**< Noop */
};

/**
 * @brief 
 * 
 * @param visitor
 */
using SavepointReadVisitorFunction = std::function<void(SavepointVisitor& visitor)>;

/**
 * @brief 
 * 
 * @param visitor
 * @param id
 */
using SavepointReadVisitorEntityFunction = std::function<void(SavepointVisitor& visitor, SavepointID id)>;

/**
 * @brief 
 * 
 * @param visitor
 * @param x
 * @param y
 */
using SavepointReadVisitorTile2DFunction = std::function<void(SavepointVisitor& visitor, int x, int y)>;

/**
 * @brief 
 * 
 * @param visitor
 * @param x
 * @param y
 * @param z
 */
using SavepointReadVisitorTile3DFunction = std::function<void(SavepointVisitor& visitor, int x, int y, int z)>;

/**
 * @brief 
 * 
 */
class ISavepointDriver
{
public:
    /**
     * @brief 
     * 
     * @param path 
     * @param version 
     * @return 
     */
    virtual SavepointStatus Open(const std::string_view& path, SavepointVersion version) = 0;

    /**
     * @brief 
     * 
     * @return 
     */
    virtual bool IsOpen() const = 0;

    /**
     * @brief 
     * 
     * @param visitor 
     */
    virtual void Write(SavepointVisitor& visitor) = 0;

    /**
     * @brief 
     * 
     * @param visitor 
     * @param id 
     * @param level 
     */
    virtual void Write(SavepointVisitor& visitor, SavepointID& id, int level) = 0;

    /**
     * @brief 
     * 
     * @param visitor 
     * @param x 
     * @param y 
     * @param level 
     */
    virtual void Write(SavepointVisitor& visitor, int x, int y, int level) = 0;

    /**
     * @brief 
     * 
     * @param visitor 
     * @param x 
     * @param y 
     * @param z 
     * @param level 
     */
    virtual void Write(SavepointVisitor& visitor, int x, int y, int z, int level) = 0;

    /**
     * @brief 
     * 
     * @param function 
     */
    virtual void Read(const SavepointReadVisitorFunction& function) = 0;

    /**
     * @brief 
     * 
     * @param function 
     * @param level 
     */
    virtual void Read(const SavepointReadVisitorEntityFunction& function, int level) = 0;

    /**
     * @brief 
     * 
     * @param function 
     * @param level 
     */
    virtual void Read(const SavepointReadVisitorTile2DFunction& function, int level) = 0;

    /**
     * @brief 
     * 
     * @param function 
     * @param level 
     */
    virtual void Read(const SavepointReadVisitorTile3DFunction& function, int level) = 0;

    /**
     * @brief 
     * 
     * @param id 
     */
    virtual void Delete(const SavepointID id) = 0;

    /**
     * @brief 
     * 
     */
    virtual void Close() = 0;

    /**
     * @brief 
     * 
     */
    virtual void Save() = 0;

    /**
     * @brief 
     * 
     */
    virtual void Clear() = 0;
};