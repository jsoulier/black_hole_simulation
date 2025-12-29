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

template<typename T> using SavepointReadFunction = std::function<void(T& item)>;
template<typename T> using SavepointReadEntityFunction = std::function<void(T& item, SavepointID id)>;
template<typename T> using SavepointReadTile2DFunction = std::function<void(T& item, int x, int y)>;
template<typename T> using SavepointReadTile3DFunction = std::function<void(T& item, int x, int y, int z)>;

class Savepoint
{
public:
    Savepoint() = default;
    ~Savepoint();
    Savepoint(const Savepoint& other) = delete;
    Savepoint& operator=(const Savepoint& other) = delete;
    Savepoint(Savepoint&& other) = delete;
    Savepoint& operator=(Savepoint&& other) = delete;
    
    // Opens or creates a database at the specified path and connects to it. The version should be your application version
    SavepointStatus Open(SavepointDriver driver, const std::string_view& path, SavepointVersion version);

    // Check if connected
    bool IsOpen() const;

    // Write singleton
    template<SavepointReadableWritable T>
    void Write(T& item)
    {
        Visitor.Reset(Version, kSavepointVersion);
        Visitor(item);
        Driver->Write(Visitor);
    }

    // Write entity to a specific level. Will move across levels if ID exists
    template<SavepointReadableWritable T>
    void Write(T& item, SavepointID& id, int level)
    {
        Visitor.Reset(Version, kSavepointVersion);
        Visitor(item);
        Driver->Write(Visitor, id, level);
    }

    // Write object to XY coordinate for a specific level
    template<SavepointReadableWritable T>
    void Write(T& item, int x, int y, int level)
    {
        Visitor.Reset(Version, kSavepointVersion);
        Visitor(item);
        Driver->Write(Visitor, x, y, level);
    }

    // Write object to XYZ coordinate for a specific level
    template<SavepointReadableWritable T>
    void Write(T& item, int x, int y, int z, int level)
    {
        Visitor.Reset(Version, kSavepointVersion);
        Visitor(item);
        Driver->Write(Visitor, x, y, z, level);
    }

    // Polymorphic write variants
    void Write(SavepointBase* base);
    void Write(SavepointBase* base, SavepointID& id, int level);
    void Write(SavepointBase* base, int x, int y, int level);
    void Write(SavepointBase* base, int x, int y, int z, int level);

    // Read singleton
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

    // Read entities from a specific level
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

    // Read XY objects from a specific level
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

    // Read XYZ objects from a specific level
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

    // Polymorphic read variants
    void Read(const SavepointReadBaseFunction& function);
    void Read(const SavepointReadBaseEntityFunction& function, int level);
    void Read(const SavepointReadBaseTile2DFunction& function, int level);
    void Read(const SavepointReadBaseTile3DFunction& function, int level);
    
    // Delete entity
    void Delete(const SavepointID id);

    // Close connection
    void Close();
    
    // End/start transaction
    void Save();

    // Delete all rows
    void Clear();

private:
    SavepointVersion Version;
    SavepointVisitor Visitor;
    std::unique_ptr<ISavepointDriver> Driver;
};