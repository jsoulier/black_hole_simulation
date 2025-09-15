#include <savepoint.hpp>
#include <cassert>
#include <filesystem>
#include <memory>
#include <string>

static const std::string kFileName = "savepoint.sqlite3";

static constexpr SavepointVersion kVersion1{0, 0, 1};
static constexpr SavepointVersion kVersion2{0, 1, 0};
static constexpr SavepointVersion kVersion3{0, 2, 0};
static constexpr SavepointVersion kVersion4{1, 1, 0};

struct ItemV2
{
    int Durability = 1;
    int Damage = 2;

    void Visit(SavepointArchive& archive)
    {
        archive(Durability);
        archive(Damage);
    }
};

struct ItemV3
{
    int Durability = 1;
    int Rarity = 2;
    int Damage = 3;

    void Visit(SavepointArchive& archive)
    {
        archive(Durability);
        archive(Damage);
        archive(Rarity, kVersion3);
    }
};

struct EntityV1
{
    float X = 1.0f;
    float Y = 2.0f;
    int Health = 3;

    void Visit(SavepointArchive& archive)
    {
        archive(X);
        archive(Y);
        archive(Health);
    }

    bool operator==(const EntityV1& other) const
    {
        return X == other.X &&
            Y == other.Y &&
            Health == other.Health;
    }
};

struct EntityV2
{
    int Strength = 1;
    float X = 2.0f;
    float Y = 3.0f;
    ItemV2 Item;
    int Health = 4;

    void Visit(SavepointArchive& archive)
    {
        archive(Strength, kVersion2);
        archive(X);
        archive(Y);
        archive(Item, kVersion2);
        archive(Health);
    }
};

struct EntityV3
{
    int Strength = 1;
    float X = 2.0f;
    float Y = 3.0f;
    ItemV2 Item;
    int Health = 4;
    int Intelligence = 5;

    void Visit(SavepointArchive& archive)
    {
        archive(Strength, kVersion2);
        archive(X);
        archive(Y);
        archive(Item, kVersion2);
        archive(Health);
        archive(Intelligence);
    }
};

struct EntityV4
{
    int Strength = 1;
    float X = 2.0f;
    float Z = 3.0f;
    float Y = 4.0f;
    ItemV3 Item;
    int Health = 5;
    int Intelligence = 6;

    void Visit(SavepointArchive& archive)
    {
        archive(Strength, kVersion2);
        archive(X);
        archive(Y);
        archive(Z);
        archive(Item, kVersion3);
        archive(Health);
        archive(Intelligence);
    }
};

struct ZombieV1 : EntityV1
{
    float Speed = 1.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV1::Visit(archive);
        archive(Speed);
    }
};

struct ZombieV21 : EntityV1
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV1::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
    }
};

struct ZombieV22 : EntityV2
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV2::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
    }
};

struct ZombieV4 : EntityV4
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;
    float VelocityY = 2.0f;
    float VelocityZ = 2.0f;

    void Visit(SavepointArchive& archive)
    {
        EntityV4::Visit(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
        archive(VelocityY, kVersion4);
        archive(VelocityZ, kVersion4);
    }
};

int main()
{
    std::filesystem::remove(kFileName);
    std::filesystem::remove(kFileName + "-journal");
    Savepoint savepoint;
    assert(savepoint.Open(kFileName));
    {
        SavepointArchive archive{kVersion1};
        SavepointID inEntityID;
        std::unique_ptr<EntityV1> inEntity = std::make_unique<EntityV1>();
        inEntity->Visit(archive);
        savepoint.Write(archive, inEntityID);
        int i = 0;
        savepoint.Read([&](SavepointArchive& archive, SavepointID outEntityID)
        {
            std::unique_ptr<EntityV1> outEntity = std::make_unique<EntityV1>();
            outEntity->Visit(archive);
            assert(*inEntity == *outEntity);
            assert(inEntityID == outEntityID);
            i++;
        });
        assert(i == 1);
        savepoint.Clear();
    }
    savepoint.Save();
    savepoint.Close();
    return 0;
}