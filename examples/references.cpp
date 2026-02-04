// [references]

// Supporting references in Savepoint comes with some issues.
//
// 1. I need to be able to load anything on demand. If Savepoint loads an entity
// with a reference, Savepoint needs to load the entity to ensure it exists in a
// shared_ptr. That means that you can only reference SAVEPOINT_POLY entities
// (since they're the only entities that provide Savepoint with a factory).
// To do that, I'd need another class that inherits both SavepointEntity and SavepointPoly. 
//
// 2. The roles get a bit messy as well. When visiting a reference, the visitor would
// have to query the Savepoint to load the entity on demand. Currently, the visitor
// doesn't have to care about the Savepoint.
//
// 3. To avoid loading the same entity multiple times, Savepoint needs to keep track
// of all loaded entities AND avoid loading them again when requested to load all
// entities.
//
// 4. Since there's currently a single visitor per Savepoint, I'd have to recursively
// create visitors when entities are loaded on demand. Otherwise you'll clobber the
// current visitor.
//
// 5. Loaded entities have to track what level they were loaded from. Typically the user
// is expected to assign the entity to a level. Since the user didn't ask for that entity
// explicitely, we don't know what level it belongs to (without asking the driver).
//
// 6. Look how much simpler the following code is.

#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>
#include <map>
#include <memory>
#include <memory>
#include <unordered_map>
#include <vector>

static constexpr SavepointVersion kVersion{0, 0, 0};

struct PolyEntity : SavepointEntity, SavepointPoly
{
    virtual void OnDeserialized() {}
    virtual bool IsValid() const { return true; }
};

// IDs are guaranteed to be unique. Since they can be serialized, you can safely map them to entities
static std::unordered_map<SavepointID, std::weak_ptr<PolyEntity>> References;

// You'll probably want a list of strong references for your entities
static std::vector<std::shared_ptr<PolyEntity>> Entities;

struct EntityReference
{
    // The concrete ID to the entity
    SavepointID ID;

    // A weak reference to the entity
    std::weak_ptr<PolyEntity> Entity;

    void Visit(SavepointVisitor& visitor)
    {
        // Don't serialize Entity, only the ID
        visitor(ID);
    }

    void SetEntity(const std::shared_ptr<PolyEntity>& entity)
    {
        ID = entity->GetID();
        assert(ID.IsValid());
        Entity = entity;
    }

    void OnDeserialized()
    {
        // Search for the reference using the serialized ID
        assert(ID.IsValid());
        auto it = References.find(ID);
        if (it != References.end())
        {
            Entity = it->second;
        }
        else
        {
            ID = SavepointID{};
        }
    }

    bool IsValid() const
    {
        return ID.IsValid() && !Entity.expired();
    }
};

struct PlayerEntity : PolyEntity
{
    SAVEPOINT_POLY(PlayerEntity)
};

struct ZombieEntity : PolyEntity
{
    SAVEPOINT_POLY(ZombieEntity)

    EntityReference Player;

    void Visit(SavepointVisitor& visitor)
    {
        PolyEntity::Visit(visitor);
        visitor(Player);
    }

    void OnDeserialized() override
    {
        Player.OnDeserialized();
    }

    bool IsValid() const override
    {
        return Player.IsValid() && Player.Entity.lock()->GetClassName() == "PlayerEntity";
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", kVersion);

    std::shared_ptr<PlayerEntity> inPlayer1 = std::make_shared<PlayerEntity>();
    std::shared_ptr<PlayerEntity> inPlayer2 = std::make_shared<PlayerEntity>();
    std::shared_ptr<ZombieEntity> inZombie1 = std::make_shared<ZombieEntity>();
    std::shared_ptr<ZombieEntity> inZombie2 = std::make_shared<ZombieEntity>();
    std::shared_ptr<ZombieEntity> inZombie3 = std::make_shared<ZombieEntity>();
    std::shared_ptr<ZombieEntity> inZombie4 = std::make_shared<ZombieEntity>();

    // Sadly, a limitation is that you need a valid ID, which means you need to have already be written
    savepoint.Write(inPlayer1, 0);
    savepoint.Write(inPlayer2, 0);
    savepoint.Write(inZombie1, 0);
    savepoint.Write(inZombie2, 0);
    savepoint.Write(inZombie3, 0);
    savepoint.Write(inZombie4, 0);
    
    // Now that everything has a valid ID, we can establish references
    inZombie1->Player.SetEntity(inPlayer1);
    inZombie2->Player.SetEntity(inPlayer1);
    inZombie3->Player.SetEntity(inPlayer2);
    inZombie4->Player.SetEntity(inPlayer2);
    savepoint.Write(inZombie1, 0);
    savepoint.Write(inZombie2, 0);
    savepoint.Write(inZombie3, 0);
    savepoint.Write(inZombie4, 0);

    savepoint.Read<std::shared_ptr<PolyEntity>>([&](std::shared_ptr<PolyEntity>& entity)
    {
        References[entity->GetID()] = entity;
        Entities.push_back(entity);
    }, 0);
    assert(References.size() == 6);
    assert(Entities.size() == 6);
    for (std::shared_ptr<PolyEntity>& entity : Entities)
    {
        entity->OnDeserialized();
        assert(entity->IsValid());
    }

    savepoint.Close();
    return 0;
}
// [references]
