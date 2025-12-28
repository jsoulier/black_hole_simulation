#include <savepoint.hpp>

#include <cassert>
#include <filesystem>
#include <random>

static constexpr SavepointVersion kVersion1{0, 0, 0};

enum TileType
{
    TileTypeGrass,
    TileTypeDirt,
    TileTypeStone,
};

struct Tile
{
    TileType Type;

    void Visit(SavepointVisitor& visitor)
    {
        visitor(Type);
    }

    bool operator==(const Tile& other) const
    {
        return Type == other.Type;
    }
};

int main()
{
    std::filesystem::remove("savepoint.sqlite3");

    Savepoint savepoint;
    savepoint.Open("savepoint.sqlite3", kVersion1);

    // Optional
    std::random_device device;
    std::mt19937 generator(device());
    std::uniform_int_distribution<std::mt19937::result_type> distribution(0, 2);

    // Writing tiles
    SavepointVisitor inVisitor;
    std::array<std::array<Tile, 32>, 32> inTiles;
    for (int x = 0; x < 32; x++)
    for (int y = 0; y < 32; y++)
    {
        inTiles[x][y].Type = TileType(distribution(generator));
        inVisitor.Reset();
        inVisitor(inTiles[x][y]);
        savepoint.Write(inVisitor, x, y, 0);
    }

    // Reading tiles
    int reads = 0;
    savepoint.Read([&](SavepointVisitor& outVisitor, int x, int y)
    {
        Tile outTile;
        outVisitor(outTile);
        assert(outTile == inTiles[x][y]);
        reads++;
    }, 0);
    assert(reads == 32 * 32);

    savepoint.Close();
    return 0;
}
