#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>
#include <memory>

static constexpr SavepointVersion kVersion1{0, 0, 0};

// Inherit from SavepointBase
struct Entity : public SavepointBase
{
    int X;
    int Y;
    SavepointID ID;

    Entity()
        : X{0}
        , Y{0}
    {
    }

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Y);
    }

    bool operator==(const Entity& other) const
    {
        return X == other.X && Y == other.Y;
    }
};

struct ZombieEntity : public Entity
{
    // Implement the derived methods
    SAVEPOINT_DERIVED(ZombieEntity);

    int Strength;

    ZombieEntity()
        : Entity()
        , Strength{5}
    {
    }

    void Visit(SavepointVisitor& visitor)
    {
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
    // Implement the derived methods
    SAVEPOINT_DERIVED(SpiderEntity);

    int Eyes;

    SpiderEntity()
        : Entity()
        , Eyes{8}
    {
    }

    void Visit(SavepointVisitor& visitor)
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
    savepoint.Open(SavepointDriver::Sqlite3, "savepoint.sqlite3", kVersion1);

    std::shared_ptr<ZombieEntity> inZombie = std::make_shared<ZombieEntity>();
    std::shared_ptr<SpiderEntity> inSpider = std::make_shared<SpiderEntity>();

    // Writing the entities
    savepoint.Write(inZombie.get(), inZombie->ID, 0);
    savepoint.Write(inSpider.get(), inSpider->ID, 0);

    // Reading the entities
    int reads = 0;
    savepoint.Read([&](SavepointBase* base, SavepointID id)
    {
        if (ZombieEntity* outZombie = dynamic_cast<ZombieEntity*>(base))
        {
            assert(id == inZombie->ID);
            assert(*outZombie == *inZombie);
            reads++;
        }
        else if (SpiderEntity* outSpider = dynamic_cast<SpiderEntity*>(base))
        {
            assert(id == inSpider->ID);
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
