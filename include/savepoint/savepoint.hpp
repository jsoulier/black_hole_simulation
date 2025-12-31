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
#include <savepoint/fwd.hpp>
#include <savepoint/id.hpp>
#include <savepoint/log.hpp>
#include <savepoint/traits.hpp>
#include <savepoint/version.hpp>
#include <savepoint/visitor.hpp>

#include <functional>
#include <string_view>

/**
 * @brief 
 * 
 * @tparam T 
 * @param item
 */
template<typename T>
using SavepointReadFunction = std::function<void(T& item)>;

/**
 * @brief 
 * 
 * @tparam T 
 * @param item
 * @param id
 */
template<typename T>
using SavepointReadEntityFunction = std::function<void(T& item, SavepointID id)>;

/**
 * @brief 
 * 
 * @tparam T 
 * @param item
 * @param x
 * @param y
 */
template<typename T>
using SavepointReadTile2DFunction = std::function<void(T& item, int x, int y)>;

/**
 * @brief 
 * 
 * @tparam T 
 * @param item
 * @param x
 * @param y
 * @param z
 */
template<typename T>
using SavepointReadTile3DFunction = std::function<void(T& item, int x, int y, int z)>;

class Savepoint
{
public:
    /**
     * @brief 
     * 
     */
    Savepoint() = default;

    /**
     * @brief 
     * 
     */
    ~Savepoint();

    /**
     * @brief 
     * 
     * @param other 
     */
    Savepoint(const Savepoint& other) = delete;
    
    /**
     * @brief 
     * 
     * @param other 
     * @return 
     */
    Savepoint& operator=(const Savepoint& other) = delete;
    
    /**
     * @brief 
     * 
     * @param other 
     */
    Savepoint(Savepoint&& other) = delete;
    
    /**
     * @brief 
     * 
     * @param other 
     * @return 
     */
    Savepoint& operator=(Savepoint&& other) = delete;
    
    /**
     * @brief 
     * 
     * @param driver 
     * @param path 
     * @param version 
     * @return 
     */
    SavepointStatus Open(SavepointDriver driver, const std::string_view& path, SavepointVersion version);

    /**
     * @brief 
     * 
     * @return 
     */
    bool IsOpen() const;

    /**
     * @brief 
     * 
     * @tparam T 
     * @param item 
     */
    template<SavepointReadableWritable T>
    void Write(T& item)
    {
        Visitor.BeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor);
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param item 
     * @param id 
     * @param level 
     */
    template<SavepointReadableWritable T>
    void Write(T& item, SavepointID& id, int level)
    {
        Visitor.BeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor, id, level);
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param item 
     * @param x 
     * @param y 
     * @param level 
     */
    template<SavepointReadableWritable T>
    void Write(T& item, int x, int y, int level)
    {
        Visitor.BeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor, x, y, level);
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param item 
     * @param x 
     * @param y 
     * @param z 
     * @param level 
     */
    template<SavepointReadableWritable T>
    void Write(T& item, int x, int y, int z, int level)
    {
        Visitor.BeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor, x, y, z, level);
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param function 
     */
    template<SavepointReadableWritable T>
    void Read(const SavepointReadFunction<T>& function)
    {
        Driver->Read([&function](SavepointVisitor& visitor)
        {
            T item;
            visitor(item);
            function(item);
        });
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param function 
     * @param level 
     */
    template<SavepointReadableWritable T>
    void Read(const SavepointReadEntityFunction<T>& function, int level)
    {
        Driver->Read([&function](SavepointVisitor& visitor, SavepointID id)
        {
            T item;
            visitor(item);
            function(item, id);
        }, level);
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param function 
     * @param level 
     */
    template<SavepointReadableWritable T>
    void Read(const SavepointReadTile2DFunction<T>& function, int level)
    {
        Driver->Read([&function](SavepointVisitor& visitor, int x, int y)
        {
            T item;
            visitor(item);
            function(item, x, y);
        }, level);
    }

    /**
     * @brief 
     * 
     * @tparam T 
     * @param function 
     * @param level 
     */
    template<SavepointReadableWritable T>
    void Read(const SavepointReadTile3DFunction<T>& function, int level)
    {
        Driver->Read([&function](SavepointVisitor& visitor, int x, int y, int z)
        {
            T item;
            visitor(item);
            function(item, x, y, z);
        }, level);
    }
    
    /**
     * @brief 
     * 
     * @param id 
     */
    void Delete(const SavepointID id);
    
    /**
     * @brief 
     * 
     */
    void Close();
    
    /**
     * @brief 
     * 
     */
    void Save();

    /**
     * @brief 
     * 
     */
    void Clear();

private:
    SavepointVersion Version;
    SavepointVisitor Visitor;
    std::unique_ptr<ISavepointDriver> Driver;
};