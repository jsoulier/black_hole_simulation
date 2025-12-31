#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>
#include <unordered_set>
#include <vector>

static constexpr SavepointVersion kVersion{0, 0, 0};

struct Vector
{
    int X;
    int Y;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Y);
    }

    bool operator==(const Vector& other) const
    {
        return X == other.X && Y == other.Y;
    }
};

enum Effect
{
    EffectStrength,
    EffectWeakness,
    EffectSwiftness,
    EffectSlowness,
};

struct Item
{
    int Count;
    int Durability;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Count);
        visitor(Durability);
    }

    bool operator==(const Item& other) const
    {
        return Count == other.Count && Durability == other.Durability;
    }
};

struct EntityInventory
{
    std::vector<Item> Items;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Items);
    }

    bool operator==(const EntityInventory& other) const
    {
        return Items == other.Items;
    }
};

struct Entity
{
    std::shared_ptr<EntityInventory> Inventory;
    std::unordered_set<Effect> Effects;
    Vector Position;
    SavepointID ID;

    Entity()
        : Inventory{std::make_shared<EntityInventory>()}
        , Effects{}
        , Position{}
    {
    }

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Inventory);
        visitor(Effects);
        visitor(Position);
    }

    bool operator==(const Entity& other) const
    {
        return *Inventory == *other.Inventory &&
            Effects == other.Effects &&
            Position == other.Position;
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open(SavepointDriver::Sqlite3, "savepoint.sqlite3", kVersion);

    Entity inEntity;
    inEntity.Inventory->Items = {{1, 50}, {2, 50}, {5, 50}};
    inEntity.Effects = {EffectStrength, EffectSlowness};
    inEntity.Position = {100, 200};
    savepoint.Write(inEntity, inEntity.ID, 0);
    savepoint.Read<Entity>([&](Entity& outEntity, SavepointID id)
    {
        assert(outEntity == inEntity);
    }, 0);

    savepoint.Close();
    return 0;
}
