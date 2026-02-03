// [spatial_types]
#include <savepoint/savepoint.hpp>

#include <cassert>
#include <filesystem>
#include <random>

static constexpr SavepointVersion kVersion{0, 0, 0};

enum TileType
{
    TileTypeGrass,
    TileTypeDirt,
    TileTypeStone,
};

struct Tile
{
    TileType Type;
    int X;
    int Y;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Type);
        visitor(X);
        visitor(Y);
    }

    bool operator==(const Tile& other) const
    {
        return Type == other.Type && X == other.X && Y == other.Y;
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", kVersion);

    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<std::mt19937::result_type> distribution(0, 2);

    std::array<std::array<Tile, 32>, 32> inTiles;
    for (int x = 0; x < 32; x++)
    for (int y = 0; y < 32; y++)
    {
        inTiles[x][y].Type = TileType(distribution(generator));
        inTiles[x][y].X = x;
        inTiles[x][y].Y = y;
        savepoint.Write(inTiles[x][y], x, y, 0);
    }

    int reads = 0;
    savepoint.Read<Tile>([&](Tile& outTile, int x, int y)
    {
        assert(outTile == inTiles[x][y]);
        reads++;
    }, 0);
    assert(reads == 32 * 32);

    for (int x = 0; x < 32; x++)
    for (int y = 0; y < 32; y++)
    {
        Tile tile;
        bool exists = savepoint.Read(tile, x, y, 0);
        assert(exists);
        assert(tile.X == x);
        assert(tile.Y == y);
    }

    savepoint.Close();
    return 0;
}
// [spatial_types]
