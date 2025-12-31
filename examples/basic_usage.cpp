// [basic_usage]
#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>

// The version of your application
static constexpr SavepointVersion kVersion{0, 0, 0};

struct Entity
{
    // The data we want to serialize
    int X;
    int Y;

    // An ID for uniquely identifying the entity
    SavepointID ID;

    // Any objects that use pointers or may be modified in the future should
    // implement the Visit function. The Visit function allows complex datatypes
    // to be serialized (pointers, vectors, maps, etc) and have versioning
    // to avoid breaking old saves
    void Visit(SavepointVisitor& visitor)
    {
        // Visit X and Y
        visitor(X);
        visitor(Y);
        // Don't visit the ID (used later)
    }

    // Optional
    bool operator==(const Entity& other) const
    {
        return X == other.X && Y == other.Y;
    }
};

int main()
{
    // Clear old saves
    std::filesystem::remove("savepoint.sqlite3");

    // Open a Savepoint (only SQLite is supported right now)
    Savepoint savepoint;
    switch (savepoint.Open(SavepointDriver::Sqlite3, "savepoint.sqlite3", kVersion))
    {
    case SavepointStatus::Failed:
        // Failed to open for any reason
        return 1;
    case SavepointStatus::Existing:
        // Opened an existing Savepoint
        break;
    case SavepointStatus::New:
        // Opened a new Savepoint
        break;
    }

    // Create and write an entity along with their ID to level 0
    Entity inEntity{1, 2};
    assert(!inEntity.ID.IsValid());
    savepoint.Write(inEntity, inEntity.ID, 0);
    assert(inEntity.ID.IsValid());

    // Provide a callback to read entities from level 0
    int reads = 0;
    savepoint.Read<Entity>([&](Entity& outEntity, SavepointID id)
    {
        // Make sure to hold onto the ID
        outEntity.ID = id;

        // Optional
        assert(outEntity == inEntity);
        assert(id == inEntity.ID);
        reads++;
    }, 0);
    assert(reads == 1);

    // Commit the transaction and start a new one. Next time the Savepoint
    // is opened, it will return Existing instead of New
    savepoint.Save();
    
    savepoint.Close();
    return 0;
}
// [basic_usage]
