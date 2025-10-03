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

struct Header
{
    int Major;
    int Minor;
    int Patch;
};

struct ItemV2
{
    int Durability = 1;
    int Damage = 2;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Durability);
        visitor(Damage);
    }

    bool operator==(const ItemV2& other) const
    {
        return Durability == other.Durability &&
            Damage == other.Damage;
    }
};

struct ItemV4
{
    int Durability = 1;
    int Rarity = 2;
    int Damage = 3;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Durability);
        visitor(Damage);
        visitor(Rarity, kVersion4);
    }

    bool operator==(const ItemV2& other) const
    {
        return Durability == other.Durability &&
            Rarity == 2 &&
            Damage == other.Damage;
    }

    bool operator==(const ItemV4& other) const
    {
        return Durability == other.Durability &&
            Rarity == other.Rarity &&
            Damage == other.Damage;
    }
};

struct EntityV1
{
    float X = 1.0f;
    float Y = 2.0f;
    int Health = 3;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Y);
        visitor(Health);
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
    int Strength = 4;
    float X = 1.0f;
    float Y = 2.0f;
    ItemV2 Item;
    int Health = 3;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Strength, kVersion2);
        visitor(X);
        visitor(Y);
        visitor(Item, kVersion2);
        visitor(Health);
    }

    bool operator==(const EntityV1& other) const
    {
        return Strength == 4 &&
            X == other.X &&
            Y == other.Y &&
            Item == ItemV2{} &&
            Health == other.Health;
    }

    bool operator==(const EntityV2& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }
};

struct EntityV3
{
    int Strength = 4;
    float X = 1.0f;
    float Y = 2.0f;
    ItemV2 Item;
    int Health = 3;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Strength, kVersion2);
        visitor(X);
        visitor(Y);
        visitor(Item, kVersion2);
        visitor(Health);
    }

    bool operator==(const EntityV1& other) const
    {
        return Strength == 4 &&
            X == other.X &&
            Y == other.Y &&
            Item == ItemV2{} &&
            Health == other.Health;
    }

    bool operator==(const EntityV2& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }

    bool operator==(const EntityV3& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }
};

struct EntityV4
{
    int Strength = 4;
    float X = 1.0f;
    float Z = 6.0f;
    float Y = 2.0f;
    ItemV4 Item;
    int Health = 3;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Strength, kVersion2);
        visitor(X);
        visitor(Y);
        visitor(Z, kVersion4);
        visitor(Item, kVersion2);
        visitor(Health);
    }

    bool operator==(const EntityV1& other) const
    {
        return Strength == 4 &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == ItemV4{} &&
            Health == other.Health;
    }

    bool operator==(const EntityV2& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }

    bool operator==(const EntityV3& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == 6.0f &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }

    bool operator==(const EntityV4& other) const
    {
        return Strength == other.Strength &&
            X == other.X &&
            Z == other.Z &&
            Y == other.Y &&
            Item == other.Item &&
            Health == other.Health;
    }
};

struct ZombieV1 : EntityV1
{
    float Speed = 1.0f;

    void Visit(SavepointVisitor& visitor)
    {
        EntityV1::Visit(visitor);
        visitor(Speed);
    }

    bool operator==(const ZombieV1& other) const
    {
        return EntityV1::operator==(other) &&
            Speed == other.Speed;
    }
};

struct ZombieV2 : EntityV2
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;

    void Visit(SavepointVisitor& visitor)
    {
        EntityV2::Visit(visitor);
        visitor(Speed);
        visitor(VelocityX, kVersion2);
    }

    bool operator==(const ZombieV1& other) const
    {
        return EntityV2::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == 2.0f;
    }

    bool operator==(const ZombieV2& other) const
    {
        return EntityV2::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX;
    }
};

struct ZombieV3 : EntityV3
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;
    float VelocityY = 3.0f;
    float VelocityZ = 4.0f;

    void Visit(SavepointVisitor& visitor)
    {
        EntityV3::Visit(visitor);
        visitor(Speed);
        visitor(VelocityX, kVersion2);
        visitor(VelocityY, kVersion3);
        visitor(VelocityZ, kVersion3);
    }

    bool operator==(const ZombieV1& other) const
    {
        return EntityV3::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == 2.0f &&
            VelocityY == 3.0f &&
            VelocityZ == 4.0f;
    }

    bool operator==(const ZombieV2& other) const
    {
        return EntityV3::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == 3.0f &&
            VelocityZ == 4.0f;
    }

    bool operator==(const ZombieV3& other) const
    {
        return EntityV3::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == other.VelocityY &&
            VelocityZ == other.VelocityZ;
    }
};

struct ZombieV4 : EntityV4
{
    float Speed = 1.0f;
    float VelocityX = 2.0f;
    float VelocityY = 3.0f;
    float VelocityZ = 4.0f;

    void Visit(SavepointVisitor& visitor)
    {
        EntityV4::Visit(visitor);
        visitor(Speed);
        visitor(VelocityX, kVersion2);
        visitor(VelocityY, kVersion3);
        visitor(VelocityZ, kVersion3);
    }

    bool operator==(const ZombieV1& other) const
    {
        return EntityV4::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == 2.0f &&
            VelocityY == 3.0f &&
            VelocityZ == 4.0f;
    }

    bool operator==(const ZombieV2& other) const
    {
        return EntityV4::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == 3.0f &&
            VelocityZ == 4.0f;
    }

    bool operator==(const ZombieV3& other) const
    {
        return EntityV4::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == other.VelocityY &&
            VelocityZ == other.VelocityZ;
    }

    bool operator==(const ZombieV4& other) const
    {
        return EntityV4::operator==(other) &&
            Speed == other.Speed &&
            VelocityX == other.VelocityX &&
            VelocityY == other.VelocityY &&
            VelocityZ == other.VelocityZ;
    }
};

struct Tile2D
{
    int X;
    int Y;
};

struct Tile3D
{
    int X;
    int Y;
    int Z;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(X);
        visitor(Y);
        visitor(Z);
    }
};

template<typename InT, typename OutT>
void Test(Savepoint& savepoint, SavepointVersion inVersion)
{
    SavepointVisitor inVisitor{inVersion};
    SavepointID inEntityID;
    std::shared_ptr<InT> inEntity = std::make_shared<InT>();
    inVisitor(*inEntity);
    savepoint.Write(inVisitor, inEntityID, 0);
    int i = 0;
    savepoint.Read([&](SavepointVisitor& outVisitor, SavepointID outEntityID)
    {
        std::shared_ptr<OutT> outEntity = std::make_shared<OutT>();
        outVisitor(*outEntity);
        assert(*outEntity == *inEntity);
        assert(outEntityID == inEntityID);
        i++;
    }, 0);
    assert(i == 1);
    savepoint.Clear();
};

int main()
{
    std::filesystem::remove(kFileName);
    std::filesystem::remove(kFileName + "-journal");
    Savepoint savepoint;
    assert(savepoint.Open(kFileName));
    {
        SavepointVisitor inVisitor{SavepointVersion{}};
        Header header;
        header.Major = 1;
        header.Minor = 2;
        header.Patch = 3;
        inVisitor(header);
        savepoint.Write(inVisitor);
        savepoint.Read([&](SavepointVisitor& outVisitor)
        {
            outVisitor(header);
            assert(header.Major = 1);
            assert(header.Minor = 2);
            assert(header.Patch = 3);
        });
        header.Major = 4;
        header.Minor = 5;
        header.Patch = 6;
        inVisitor(header);
        savepoint.Write(inVisitor);
        savepoint.Read([&](SavepointVisitor& outVisitor)
        {
            outVisitor(header);
            assert(header.Major = 4);
            assert(header.Minor = 5);
            assert(header.Patch = 6);
        });
    }
    Test<EntityV1, EntityV1>(savepoint, kVersion1);
    Test<EntityV1, EntityV2>(savepoint, kVersion1);
    Test<EntityV2, EntityV2>(savepoint, kVersion2);
    Test<EntityV1, EntityV3>(savepoint, kVersion1);
    Test<EntityV2, EntityV3>(savepoint, kVersion2);
    Test<EntityV3, EntityV3>(savepoint, kVersion3);
    Test<EntityV1, EntityV4>(savepoint, kVersion1);
    Test<EntityV2, EntityV4>(savepoint, kVersion2);
    Test<EntityV3, EntityV4>(savepoint, kVersion3);
    Test<EntityV4, EntityV4>(savepoint, kVersion4);
    Test<ZombieV1, ZombieV1>(savepoint, kVersion1);
    Test<ZombieV1, ZombieV2>(savepoint, kVersion1);
    Test<ZombieV2, ZombieV2>(savepoint, kVersion2);
    Test<ZombieV1, ZombieV3>(savepoint, kVersion1);
    Test<ZombieV2, ZombieV3>(savepoint, kVersion2);
    Test<ZombieV3, ZombieV3>(savepoint, kVersion3);
    Test<ZombieV1, ZombieV4>(savepoint, kVersion1);
    Test<ZombieV2, ZombieV4>(savepoint, kVersion2);
    Test<ZombieV3, ZombieV4>(savepoint, kVersion3);
    Test<ZombieV4, ZombieV4>(savepoint, kVersion4);
    {
        SavepointVisitor inVisitor{SavepointVersion{}};
        for (int inX = 0; inX < 256; inX++)
        for (int inY = 0; inY < 256; inY++)
        {
            inVisitor.Reset();
            Tile2D inTile2D{inX, inY};
            inVisitor(inTile2D);
            savepoint.Write(inVisitor, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read([&](SavepointVisitor& outVisitor, int outX, int outY)
        {
            Tile2D outTile2D;
            outVisitor(outTile2D);
            assert(outTile2D.X == outX);
            assert(outTile2D.Y == outY);
            i++;
        }, 0);
        assert(i == 256 * 256);
    }
    {
        SavepointVisitor inVisitor{SavepointVersion{}};
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        for (int inZ = 0; inZ < 32; inZ++)
        {
            inVisitor.Reset();
            Tile3D inTile3D{inX, inY, inZ};
            inVisitor(inTile3D);
            savepoint.Write(inVisitor, inX, inY, inZ, 0);
        }
        int i = 0;
        savepoint.Read([&](SavepointVisitor& outVisitor, int outX, int outY, int outZ)
        {
            Tile3D outTile3D;
            outVisitor(outTile3D);
            assert(outTile3D.X == outX);
            assert(outTile3D.Y == outY);
            assert(outTile3D.Z == outZ);
            i++;
        }, 0);
        assert(i == 32 * 32 * 32);
    }
    savepoint.Clear();
    {
        SavepointVisitor inVisitor{SavepointVersion{}};
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        {
            inVisitor.Reset();
            Tile2D inTile2D{5, 5};
            inVisitor(inTile2D);
            savepoint.Write(inVisitor, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read([&](SavepointVisitor& outVisitor, int outX, int outY)
        {
            Tile2D outTile2D;
            outVisitor(outTile2D);
            assert(outTile2D.X == 5);
            assert(outTile2D.Y == 5);
            i++;
        }, 0);
        assert(i == 32 * 32);
    }
    {
        SavepointVisitor inVisitor{SavepointVersion{}};
        for (int inX = 0; inX < 32; inX++)
        for (int inY = 0; inY < 32; inY++)
        {
            inVisitor.Reset();
            Tile2D inTile2D{10, 10};
            inVisitor(inTile2D);
            savepoint.Write(inVisitor, inX, inY, 0);
        }
        int i = 0;
        savepoint.Read([&](SavepointVisitor& outVisitor, int outX, int outY)
        {
            Tile2D outTile2D;
            outVisitor(outTile2D);
            assert(outTile2D.X == 10);
            assert(outTile2D.Y == 10);
            i++;
        }, 0);
        assert(i == 32 * 32);
    }
    savepoint.Save();
    savepoint.Close();
    return 0;
}