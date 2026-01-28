// [polymorphic_types]
#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>
#include <memory>

static constexpr SavepointVersion kVersion{0, 0, 0};

// Your base class inherits from SavepointBase (and optionally SavepointEntity)
struct Entity : SavepointBase, SavepointEntity
{
    int X;
    int Y;

    Entity()
        : X{0}
        , Y{0}
    {
    }

    // Optionally implement Visit
    void Visit(SavepointVisitor& visitor) override
    {
        visitor(X);
        visitor(Y);
    }

    bool operator==(const Entity& other) const
    {
        return X == other.X && Y == other.Y;
    }
};

// Your derived classes inherit from Entity as usual
struct ZombieEntity : public Entity
{
    // Your concrete derived classes use SAVEPOINT_DERIVED to implement required methods
    SAVEPOINT_DERIVED(ZombieEntity);

    int Strength;

    ZombieEntity()
        : Entity()
        , Strength{5}
    {
    }

    void Visit(SavepointVisitor& visitor) override
    {
        // Make sure to use the base class' Visit function
        Entity::Visit(visitor);
        visitor(Strength);
    }

    bool operator==(const ZombieEntity& other) const
    {
        return Entity::operator==(other) && Strength == other.Strength;
    }
};

struct SpiderEntity : public Entity
{
    SAVEPOINT_DERIVED(SpiderEntity);

    int Eyes;

    SpiderEntity()
        : Entity()
        , Eyes{8}
    {
    }

    void Visit(SavepointVisitor& visitor) override
    {
        Entity::Visit(visitor);
        visitor(Eyes);
    }

    bool operator==(const SpiderEntity& other) const
    {
        return Entity::operator==(other) && Eyes == other.Eyes;
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", kVersion);

    // Write derived classes as usual
    std::shared_ptr<ZombieEntity> inZombie = std::make_shared<ZombieEntity>();
    std::shared_ptr<SpiderEntity> inSpider = std::make_shared<SpiderEntity>();
    savepoint.Write(inZombie, 0);
    savepoint.Write(inSpider, 0);

    // Read using your base class' class name
    int reads = 0;
    savepoint.Read<std::shared_ptr<Entity>>([&](std::shared_ptr<Entity>& entity)
    {
        // The read entity is either a ZombieEntity or a SpiderEntity. You can safely take ownership of it
        if (ZombieEntity* outZombie = dynamic_cast<ZombieEntity*>(entity.get()))
        {
            assert(*outZombie == *inZombie);
            reads++;
        }
        else if (SpiderEntity* outSpider = dynamic_cast<SpiderEntity*>(entity.get()))
        {
            assert(*outSpider == *inSpider);
            reads++;
        }
        else
        {
            assert(false);
        }
    }, 0);
    assert(reads == 2);

    savepoint.Close();
    return 0;
}
// [polymorphic_types]
