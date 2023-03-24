# Epoch

`epoch` is a collection of utilities, examples, and binaries used to adapt finite element methods to Numerical Relativity

## Project Structure
- `include` - Public header files
- `src` - Private source files
- `bins` - Executable projects
- `tests` - Executable tests
- `libs` - External libraries
- `packages` - Packages which must be compiled
- `scripts` - Scripts used by CMAKE and for analysis

## Prerequisites
The following pacakges should be installed and accessible on the path to successfully compile `epoch`.

- `git` 
- `cmake`
- `python3`

## Coding Guidelines
- All files and programming folders should be in *snake_case*
- All variables and functions should be in *snake_case*
- All consts should be in *SCREAMING_SNAKE_CASE*
- All typenames other than primatives should be in *PascalCase*
- C++ headers should end in `.hpp` and contain `#pragma once` as a first directive
- C++ source files should end in `.cpp`
- Python files end in `.py`
- All cmake build artifacts should be placed in a untracked `build` directory (or `packages/build` for packages).