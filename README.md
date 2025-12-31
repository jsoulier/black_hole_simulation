# Savepoint

C++ lacks built-in object serialization.
Libraries like [cereal](https://github.com/USCiLab/cereal) handle serialization, but persistence and versioning are still left to the user.
Savepoint combines serialization, storage, and versioning into a single system.

### Features

- Automatic transactions
- Automatic version upgrading
- UUID and spatial keys
- Inherited and nested fields
- Polymorphic types
- Vectors, sets, maps, pointers, and more

### CMake

You can copy the source and add the following to your CMakeLists.txt:

```cmake
add_subdirectory(<path>)
target_link_libraries(<name> PRIVATE savepoint::savepoint)
```

### Documentation

The source contains Doxygen-style comments.
You can generate HTML docs with:

```shell
doxygen Doxyfile
```

### Examples

You can find all examples [here](examples)

#### Basic Usage

```c++
#include <savepoint/savepoint.hpp>

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
    savepoint.Open(SavepointDriver::Sqlite3, "savepoint.sqlite3", SavepointVersion{});

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
