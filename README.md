# Savepoint

Savepoint is a lightweight, all-in-one solution for serialization, versioning, and persistence in C++ applications.
Inspired by [cereal](https://github.com/USCiLab/cereal) and built on proven databases like [SQLite](https://sqlite.org/),
it provides a simple, fast, and reliable interface for persisting C++ objects. Including documentation, the library is
roughly 2K lines of code.

### Features

- Automatic transactions
- Automatic version upgrading
- UUID and spatial keys
- Inherited and nested fields
- Polymorphic types
- Vectors, sets, maps, pointers, and more
- Support for multiple backends (only SQLite3 is supported right now)

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

struct Entity
{
    int X;
    int Y;
    SavepointID ID;

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
    savepoint.Write(inEntity, inEntity.ID, 0);
    savepoint.Read<Entity>([&](Entity& outEntity, SavepointID id)
    {
        assert(outEntity == inEntity);
        assert(id == inEntity.ID);
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
