#pragma once

#include <savepoint/base.hpp>
#include <savepoint/driver.hpp>
#include <savepoint/fwd.hpp>
#include <savepoint/id.hpp>
#include <savepoint/log.hpp>
#include <savepoint/status.hpp>
#include <savepoint/traits.hpp>
#include <savepoint/version.hpp>
#include <savepoint/visitor.hpp>

#include <functional>
#include <string_view>

/**
 * @brief The read function signature.
 * 
 * @tparam T The type to read.
 * @param item The read item.
 */
template<typename T>
using SavepointReadFunction = std::function<void(T& item)>;

/**
 * @brief The entity read function signature.
 * 
 * @tparam T The type to read.
 * @param item The read item.
 * @param id The read ID.
 * @see SavepointID
 */
template<typename T>
using SavepointReadEntityFunction = std::function<void(T& item, SavepointID id)>;

/**
 * @brief The 2D tile read function signature.
 * 
 * @tparam T The type to read.
 * @param item The read item.
 * @param x The x location.
 * @param y The y location.
 */
template<typename T>
using SavepointReadTile2DFunction = std::function<void(T& item, int x, int y)>;

/**
 * @brief The 3D tile read function signature.
 * 
 * @tparam T The type to read.
 * @param item The read item.
 * @param x The x location.
 * @param y The y location.
 * @param z The z location.
 */
template<typename T>
using SavepointReadTile3DFunction = std::function<void(T& item, int x, int y, int z)>;

/**
 * @brief The connection handle to a Savepoint file.
 * 
 * @snippet examples/basic_usage.cpp basic_usage
 */
class Savepoint
{
public:
    /**
     * @brief Default initializes the connection.
     */
    Savepoint() = default;

    /**
     * @brief If connected, closes the connection.
     */
    ~Savepoint();

    /**
     * @brief Deleted copy constructor.
     */
    Savepoint(const Savepoint& other) = delete;
    
    /**
     * @brief Deleted copy assignment operator.
     */
    Savepoint& operator=(const Savepoint& other) = delete;
    
    /**
     * @brief Deleted move constructor.
     */
    Savepoint(Savepoint&& other) = delete;
    
    /**
     * @brief Deleted move assignment operator.
     */
    Savepoint& operator=(Savepoint&& other) = delete;
    
    /**
     * @brief Opens a new connection.
     * 
     * @param driver The driver to use for file operations.
     * @param path The path to the Savepoint file.
     * @param version The user's application version.
     * @return The result of the attempt to open a connection.
     * @see IsOpen
     * @see Save
     * @see Close
     */
    SavepointStatus Open(SavepointDriver driver, const std::string_view& path, SavepointVersion version);

    /**
     * @brief Check if connected to a Savepoint file.
     * 
     * @return True if connected.
     * @see Open
     * @see Close
     */
    bool IsOpen() const;

    /**
     * @brief Write a singleton to the Savepoint.
     * 
     * For storing information such as date and time, the user can write a
     * singleton with the assumption that only one entry exists.
     * 
     * @tparam T The type to write.
     * @param item The item to write.
     */
    template<SavepointVisitable T>
    void Write(T& item)
    {
        Visitor.BeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor);
    }

    /**
     * @brief Write an entity to the Savepoint.
     * 
     * For items without a unique location, an ID can be used to ensure the item
     * gets a unique entry. If the ID is invalid, the ID will be written to and
     * a new entry will be inserted. If the ID is valid, an existing entry will
     * be updated (including the level).
     * 
     * @tparam T The type to write.
     * @param item The item to write.
     * @param id The ID.
     * @param level The level.
     * @see SavepointID
     */
    template<SavepointVisitable T>
    void Write(T& item, SavepointID& id, int level)
    {
        Visitor.BeginWriting(Version);
        Visitor(item);
        if (!id.IsValid())
        {
            id = Driver->Insert(Visitor, level);
        }
        else
        {
            id = Driver->Update(Visitor, id, level);
            if (!id.IsValid())
            {
                id = Driver->Insert(Visitor, level);
            }
        }
    }

    /**
     * @brief Write a 2D tile to the Savepoint.
     * 
     * Writes a tile to a entry using a key of x, y, and level. If an entry
     * already exists, the entry will be overridden.
     * 
     * @tparam T The type to write.
     * @param item The item to write.
     * @param x The x location.
     * @param y The y location.
     * @param level The level.
     */
    template<SavepointVisitable T>
    void Write(T& item, int x, int y, int level)
    {
        Visitor.BeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor, x, y, level);
    }

    /**
     * @brief Write a 3D tile to the Savepoint.
     * 
     * Writes a tile to a entry using a key of x, y, z, and level. If an entry
     * already exists, the entry will be overridden.
     * 
     * @tparam T The type to write.
     * @param item The item to write.
     * @param x The x location.
     * @param y The y location.
     * @param z The z location.
     * @param level The level.
     */
    template<SavepointVisitable T>
    void Write(T& item, int x, int y, int z, int level)
    {
        Visitor.BeginWriting(Version);
        Visitor(item);
        Driver->Write(Visitor, x, y, z, level);
    }

    /**
     * @brief Read a singleton from the Savepoint.
     * 
     * @tparam T The type to read.
     * @param function The function to use.
     */
    template<SavepointVisitable T>
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
     * @brief Read all entities in the specified level from the Savepoint.
     * 
     * @tparam T The type to read.
     * @param function The function to use.
     * @param level The level.
     * @see SavepointID
     */
    template<SavepointVisitable T>
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
     * @brief Read all 2D tiles in the specified level from the Savepoint.
     * 
     * @tparam T The type to read.
     * @param function The function to use.
     * @param level The level.
     */
    template<SavepointVisitable T>
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
     * @brief Read all 3D tiles in the specified level from the Savepoint.
     * 
     * @tparam T The type to read.
     * @param function The function to use.
     * @param level The level.
     */
    template<SavepointVisitable T>
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
     * @brief Deletes an entity from the Savepoint.
     * 
     * @param id The ID.
     */
    void Delete(const SavepointID id);
    
    /**
     * @brief Closes the connection. Does NOT call Savepoint::Save.
     * 
     * @see IsOpen
     * @see Save
     * @see Close
     */
    void Close();
    
    /**
     * @brief Save all pending changes.
     * 
     * Commits the current transaction and starts a new one. The next time
     * Savepoint::Open is called, it will return SavepointStatus::Existing
     * instead of SavepointStatus::New.
     * 
     * @see Open
     */
    void Save();

    /**
     * @brief Remove all entities and tiles from the Savepoint.
     */
    void Clear();

private:
    SavepointVersion Version;
    SavepointVisitor Visitor;
    std::unique_ptr<ISavepointDriver> Driver;
};