# Savepoint

Savepoint is a lightweight ORM-style serializer for C++ games.
Inspired by [cereal](https://github.com/USCiLab/cereal) and built on databases like [SQLite](https://sqlite.org/), it provides a simple interface for saving and loading C++ objects.
Including documentation, the library is roughly 2K lines of code.

### Features

- Automatic transactions
- Automatic schema upgrades
- UUID and spatial keys
- Inherited and nested fields
- Polymorphic types
- Pointers, containers, tuples, etc
- Fast development workflow

### Limitations

- Automatic schema upgrades require [versioning](examples/basic_upgraded.cpp) for newly added members
- Saves are binary and not compatible across different bit-widths or endianness
- Entity references can't be serialized (use `SavepointID` instead)

### Documentation

The source contains Doxygen-style comments.
You can generate HTML docs with:

```shell
doxygen Doxyfile
```

### Examples

You can find all examples [here](examples)

```c++
#include <savepoint/savepoint.hpp>

#include <cassert>

struct Entity : SavepointEntity
{
    int X;
    int Y;

    Entity() = default;
    Entity(int x, int y)
        : X{x}
        , Y{y}
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

int main()
{
    Savepoint savepoint;
    savepoint.Open(SavepointDriver::SQLite3, "savepoint.sqlite3", SavepointVersion{});

    Entity inEntity{1, 2};
    savepoint.Write(inEntity, 0);
    savepoint.Read<Entity>([&](Entity& outEntity)
    {
        assert(outEntity == inEntity);
    }, 0);
    
    savepoint.Close();
    return 0;
}
```

### CMake

You can clone and add the following to your `CMakeLists.txt`:

```cmake
add_subdirectory(<path>)
target_link_libraries(<name> PRIVATE savepoint::savepoint)
```
