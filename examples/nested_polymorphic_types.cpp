// [nested_polymorphic_types]
#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>
#include <memory>

static constexpr SavepointVersion kVersion{0, 0, 0};

enum ItemID
{
    ItemIDShirt,
    ItemIDPants,
    ItemIDShoes,
};

struct Item
{
    ItemID ID;
    int Count;

    bool operator==(const Item& other) const
    {
        return ID == other.ID && Count == other.Count;
    }
};

struct PolyInventory : public SavepointPoly
{
    void Visit(SavepointVisitor& visitor) override
    {
        visitor(Items);
    }

    virtual bool IsEqual(const std::shared_ptr<PolyInventory>& inventory) const
    {
        return Items == inventory->Items;
    }

    std::vector<Item> Items;
};

struct PolyEntity : public SavepointPoly, SavepointEntity
{
    PolyEntity(const std::shared_ptr<PolyInventory>& inventory)
        : Inventory{inventory}
    {
    }

    virtual void OnCreate() {}

    void Visit(SavepointVisitor& visitor) override
    {
        visitor(Inventory);
    }

    virtual bool IsEqual(const std::shared_ptr<PolyEntity>& entity) const
    {
        return Inventory->IsEqual(entity->Inventory);
    }

    std::shared_ptr<PolyInventory> Inventory;
};

struct PlayerInventory : public PolyInventory
{
    SAVEPOINT_POLY(PlayerInventory)

    PlayerInventory()
        : PolyInventory()
        , ShirtIndex{0}
        , PantsIndex{0}
        , ShoesIndex{0}
    {
    }

    void Visit(SavepointVisitor& visitor) override
    {
        PolyInventory::Visit(visitor);
        visitor(ShirtIndex);
        visitor(PantsIndex);
        visitor(ShoesIndex);
    }

    bool IsEqual(const std::shared_ptr<PolyInventory>& inventory) const override
    {
        const std::shared_ptr<PlayerInventory> player = std::dynamic_pointer_cast<PlayerInventory>(inventory);
        return PolyInventory::IsEqual(inventory) && ShirtIndex == player->ShirtIndex &&
            PantsIndex == player->PantsIndex && ShoesIndex == player->ShoesIndex;
    }

    int ShirtIndex;
    int PantsIndex;
    int ShoesIndex;
};

struct PlayerEntity : public PolyEntity
{
    SAVEPOINT_POLY(PlayerEntity)

    // You could also create PlayerInventory in OnCreate
    PlayerEntity()
        : PolyEntity(std::make_shared<PlayerInventory>())
    {
    }

    void OnCreate() override
    {
        std::shared_ptr<PlayerInventory> inventory = std::dynamic_pointer_cast<PlayerInventory>(Inventory);
        inventory->Items.emplace_back(ItemIDShirt, 1);
        inventory->Items.emplace_back(ItemIDPants, 1);
        inventory->Items.emplace_back(ItemIDShoes, 2);
        inventory->ShirtIndex = 0;
        inventory->PantsIndex = 1;
        inventory->ShoesIndex = 2;
        Health = 100;
        Hunger = 100;
    }

    void Visit(SavepointVisitor& visitor) override
    {
        PolyEntity::Visit(visitor);
        visitor(Health);
        visitor(Hunger);
    }

    bool IsEqual(const std::shared_ptr<PolyEntity>& entity) const override
    {
        const std::shared_ptr<PlayerEntity> player = std::dynamic_pointer_cast<PlayerEntity>(entity);
        return PolyEntity::IsEqual(entity) && Health == player->Health && Hunger == player->Health;
    }

    int Health;
    int Hunger;
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", kVersion);

    std::shared_ptr<PolyEntity> inEntity = std::make_shared<PlayerEntity>();
    inEntity->OnCreate();
    savepoint.Write(inEntity, 0);

    int reads = 0;
    savepoint.Read<std::shared_ptr<PolyEntity>>([&](std::shared_ptr<PolyEntity>& outEntity)
    {
        assert(outEntity->IsEqual(inEntity));
        reads++;
    }, 0);
    assert(reads == 1);

    savepoint.Close();
    return 0;
}
// [nested_polymorphic_types]
