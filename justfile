run tape:
    uv run main.py rules.txt "{{tape}}"
    clang -target x86_64-w64-mingw32 rules.ll -o rules.exe -O3
    ./rules.exe
