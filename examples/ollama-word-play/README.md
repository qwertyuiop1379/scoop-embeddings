# Demo: "Ollama Word Play"

To show the library in action, we can perform arithmetic operations on embedding vectors, effectively navigating through semantic space. The demo program loads an input corpus, generates embeddings for each entry, then generates equations from randomly sampled inputs.

The outputs aren't useful for anything, but it's interesting to see what it produces.

Examples:

```
1.0 * possible evidence + 1.3 * single nucleotide + 1.3 * seat - 0.8 * family = ?
   1. sit                            score=0.7268
   2. evidence                       score=0.7062
   3. sitting                        score=0.6995
   4. DNA                            score=0.6678
   5. seed                           score=0.6673

0.9 * hydrous + 1.4 * Archaeological evidence + 1.3 * choice - 0.4 * international intrigue - 0.9 * clear advantages = ?
   1. archaeological evidence        score=0.7669
   2. historical archaeology         score=0.7334
   3. evidence                       score=0.6877
   4. choose                         score=0.6864
   5. archaeological materials       score=0.6857
```

## Requirements

- CMake
- A C++23 compiler
- [Ollama](https://ollama.com/)
- The `embeddinggemma` model available locally (any embedding model works)

Install the model if needed:

```bash
ollama pull embeddinggemma
```

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run The Demo

```bash
./build/examples/ollama-word-play/ollama-word-play
```

Optional arguments:

1. output directory for generated cache files
2. path to a custom corpus file

Example:

```bash
./build/examples/ollama-word-play/ollama-word-play ./build/generated/ollama-word-play ./examples/ollama-word-play/data/inputs.txt
```

On first run, the demo embeds the full corpus and writes:

- `store.csv`
- `manifest.txt`

On later runs, if the manifest still matches the current corpus and model settings, the demo reuses the cached store instead of re-embedding everything.

## Notes

- The experiment generator is random but reproducible because it uses a fixed seed in the example source.
- Changing the corpus file invalidates the cache automatically.
- The cache currently keys on corpus contents, model name, and embedding dimension. If your local Ollama model changes without the model name changing, you may still want to delete the cached output directory and rebuild the store manually.

