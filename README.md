# mng_compiler

Compiles [Marches & Gnats](https://mng.quest/) rules to LLVM IR via `llvmlite`.

## Usage

`just run <input_tape>` compiles `rules.txt`, with the tape specified by `<input_tape>`, to
`rules.ll`. You can then use `llc` and `clang` to compile the LLVM IR to a native executable.
