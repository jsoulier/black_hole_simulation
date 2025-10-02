# Savepoint

A simple, fast, and reliable key-value database for games written in C++

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

### Usage

See [here](test/savepoint.cpp) for a example usage
