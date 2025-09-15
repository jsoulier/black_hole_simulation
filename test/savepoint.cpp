#include <cassert>
#include <memory>
#include <savepoint.hpp>
#include <filesystem>

static const std::filesystem::path kFileName = "savepoint.sqlite3";

static constexpr SavepointVersion kVersion1{0, 0, 1};
static constexpr SavepointVersion kVersion2{0, 1, 0};
static constexpr SavepointVersion kVersion3{0, 2, 0};
static constexpr SavepointVersion kVersion4{1, 1, 0};

struct ItemV2
{
    int Durability = 1;
    int Damage = 2;

    void Archive(SavepointArchive& archive)
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

    void Archive(SavepointArchive& archive)
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

    void Archive(SavepointArchive& archive)
    {
        archive(X);
        archive(Y);
        archive(Health);
    }
};

struct EntityV2
{
    int Strength = 1;
    float X = 2.0f;
    float Y = 3.0f;
    ItemV2 Item;
    int Health = 4;

    void Archive(SavepointArchive& archive)
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

    void Archive(SavepointArchive& archive)
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

    void Archive(SavepointArchive& archive)
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

    void Archive(SavepointArchive& archive)
    {
        EntityV1::Archive(archive);
        archive(Speed);
    }
};

struct ZombieV21 : EntityV1
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;

    void Archive(SavepointArchive& archive)
    {
        EntityV1::Archive(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
    }
};

struct ZombieV22 : EntityV2
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;

    void Archive(SavepointArchive& archive)
    {
        EntityV2::Archive(archive);
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

    void Archive(SavepointArchive& archive)
    {
        EntityV4::Archive(archive);
        archive(Speed);
        archive(VelocityX, kVersion2);
        archive(VelocityY, kVersion4);
        archive(VelocityZ, kVersion4);
    }
};

int main()
{
    Savepoint savepoint;
    assert(savepoint.Open(kFileName));
    {
        auto entityV1 = std::make_unique<EntityV1>();
        // TODO:
    }
    savepoint.Save();
    savepoint.Close();
    return 0;
}