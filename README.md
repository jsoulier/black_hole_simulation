# Savepoint

A simple, fast, and reliable key/value database for games in C++ (built on sqlite3)

### Features

- Automatic transactions
- Automatic progressive upgrading
- UUID and spatial (2D/3D) keys
- Inherited and nested values

### Limitations
- Saves are binary and may be incompatible across architectures
- Members cannot be reordered in the visit function
- Member types cannot be changed
- Upgraded members must be manually versioned

### CMake

```cmake
add_subdirectory(<path/to/savepoint>)
target_link_libraries(<target> savepoint)
```

### Examples

See [here](test/savepoint.cpp) for a full working example

#### Reading/Writing

```c++
#include <savepoint.hpp>

static constexpr SavepointVersion kCurrentVersion{0, 1, 0};

struct Entity
{
    SavepointID ID;
    int X = 1;
    int Y = 2;

    void Visit(SavepointArchive& archive)
    {
        archive(X);
        archive(Y);
    }
};

Savepoint savepoint;
savepoint.Open("savepoint.sqlite3");

Entity entity;
SavepointArchive archive{kCurrentVersion};
archive(entity);
savepoint.Write(archive, entity.ID);
savepoint.Read([](SavepointArchive& archive, SavepointID id)
{
    Entity entity;
    archive(entity);
    assert(entity.X == 1);
    assert(entity.Y == 2);
});

savepoint.Close();
```

#### Upgrading

```c++
#include <savepoint.hpp>

static constexpr SavepointVersion kVersion1{0, 1, 0};
static constexpr SavepointVersion kVersion2{0, 2, 0};

struct Entity1
{
    SavepointID ID;
    int X = 1;
    int Y = 2;

    void Visit(SavepointArchive& archive)
    {
        archive(X);
        archive(Y);
    }
};

struct Entity2
{
    SavepointID ID;
    int X = 1;
    int Y = 2;
    int Z = 3;

    void Visit(SavepointArchive& archive)
    {
        archive(X);
        archive(Y);
        archive(Z, kVersion2, 999);
    }
};

Savepoint savepoint;
savepoint.Open("savepoint.sqlite3");

Entity1 entity;
SavepointArchive archive{kVersion1};
archive(entity);
savepoint.Write(archive, entity.ID);
savepoint.Read([](SavepointArchive& archive, SavepointID id)
{
    Entity2 entity;
    archive(entity);
    assert(entity.X == 1);
    assert(entity.Y == 2);
    assert(entity.Z == 999);
});

savepoint.Close();
```