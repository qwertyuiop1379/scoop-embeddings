# Scoop Vectors

Header-only C++ library for storing, searching, and serializing vectors.

**This is a proof-of-concept and not intended for production environments.**

## Vector Store

The vector store supports:

- in-memory storage of fixed-size vectors
- cosine similarity search
- exclusion by index during search
- serialization and reloading from text
- concurrent access with internal locking

## Usage

The core library lives in `VectorStore.hpp` and is exposed as a header-only CMake target:

```cmake
target_link_libraries(your_target PRIVATE Scoop::Vectors)
```