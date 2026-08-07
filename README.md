# Black Hole Simulation

Black Hole Simulation using SDL3 GPU with a compute shader

![](doc/image2.png)
*Euler's method implementation (VSync at 1600x1200)*

![](doc/image1.png)
*Polar coordinate implementation (30 FPS at 200x150)*

### Building

#### Windows

```bash
git clone https://github.com/jsoulier/black_hole_simulation --recurse-submodules
cd black_hole_simulation
mkdir build
cd build
cmake ..
cmake --build . --parallel 8 --config Release
cd bin
./black_hole_simulation.exe
```

#### Linux

```bash
git clone https://github.com/jsoulier/black_hole_simulation --recurse-submodules
cd black_hole_simulation
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel 8
cd bin
./black_hole_simulation
```

#### Shaders

Shaders are precompiled.
To build locally, add [SDL_shadercross](https://github.com/libsdl-org/SDL_shadercross) to your path

### References

- [Youtube Video](https://www.youtube.com/watch?v=8-B6ryuBkCM) by Kavan
