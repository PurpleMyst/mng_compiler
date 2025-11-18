import typing as t
from pathlib import Path

from llvmlite import binding, ir
from typer import Typer

TAPE_SIZE = 30_000

PRELUDE = """
; Declare external putchar function
declare i32 @putchar(i32)

; print_utf8(c: i32)
define void @print_utf8(i32 %c) {
entry:
  ; Test 1-byte range: c <= 0x7F (127)
  %is1 = icmp ule i32 %c, 127
  br i1 %is1, label %one, label %two_check

one:
  ; 1-byte ASCII
  call i32 @putchar(i32 %c)
  ret void


two_check:
  ; Test 2-byte range: c <= 0x7FF (2047)
  %is2 = icmp ule i32 %c, 2047
  br i1 %is2, label %two, label %three_check

two:
  ; 2-byte UTF-8
  ; byte1 = 0xC0 | (c >> 6)  -> 192 | (c >> 6)
  %c_sh6 = lshr i32 %c, 6
  %b1 = or i32 %c_sh6, 192

  ; byte2 = 0x80 | (c & 0x3F) -> 128 | (c & 63)
  %c_lo6 = and i32 %c, 63
  %b2 = or i32 %c_lo6, 128

  call i32 @putchar(i32 %b1)
  call i32 @putchar(i32 %b2)
  ret void


three_check:
  ; Test 3-byte range: c <= 0xFFFF (65535)
  %is3 = icmp ule i32 %c, 65535
  br i1 %is3, label %three, label %four

three:
  ; 3-byte UTF-8
  ; byte1 = 0xE0 | (c >> 12) -> 224 | (c >> 12)
  %c_sh12 = lshr i32 %c, 12
  %t1 = or i32 %c_sh12, 224

  ; byte2 = 0x80 | ((c >> 6) & 0x3F) -> 128 | ((c >> 6) & 63)
  %c_sh6_b = lshr i32 %c, 6
  %c_sh6_b_lo = and i32 %c_sh6_b, 63
  %t2 = or i32 %c_sh6_b_lo, 128

  ; byte3 = 0x80 | (c & 0x3F) -> 128 | (c & 63)
  %c_lo3 = and i32 %c, 63
  %t3 = or i32 %c_lo3, 128

  call i32 @putchar(i32 %t1)
  call i32 @putchar(i32 %t2)
  call i32 @putchar(i32 %t3)
  ret void


four:
  ; 4-byte UTF-8
  ; byte1 = 0xF0 | (c >> 18) -> 240 | (c >> 18)
  %c_sh18 = lshr i32 %c, 18
  %u1 = or i32 %c_sh18, 240

  ; byte2 = 0x80 | ((c >> 12) & 0x3F) -> 128 | ((c >> 12) & 63)
  %c_sh12_b = lshr i32 %c, 12
  %c_sh12_b_lo = and i32 %c_sh12_b, 63
  %u2 = or i32 %c_sh12_b_lo, 128

  ; byte3 = 0x80 | ((c >> 6) & 0x3F) -> 128 | ((c >> 6) & 63)
  %c_sh6_c = lshr i32 %c, 6
  %c_sh6_c_lo = and i32 %c_sh6_c, 63
  %u3 = or i32 %c_sh6_c_lo, 128

  ; byte4 = 0x80 | (c & 0x3F) -> 128 | (c & 63)
  %c_lo4 = and i32 %c, 63
  %u4 = or i32 %c_lo4, 128

  call i32 @putchar(i32 %u1)
  call i32 @putchar(i32 %u2)
  call i32 @putchar(i32 %u3)
  call i32 @putchar(i32 %u4)
  ret void
}

; print_utf8_slice(ptr: i32*)
; Prints a slice of UTF-8 characters from a pointer, until a null terminator is found.
define void @print_utf8_slice(i32* %ptr) {
entry:
    %cursor = alloca i32*
    store i32* %ptr, i32* %cursor
    br label %loop
loop:
    %cur_ptr = load i32*, i32** %cursor
    %cur_char = load i32, i32* %cur_ptr
    %is_null = icmp eq i32 %cur_char, 0
    br i1 %is_null, label %done, label %print
print:
    call void @print_utf8(i32 %cur_char)
    ; Increment cursor
    %next_ptr = getelementptr i32, i32* %cur_ptr, i32 1
    store i32* %next_ptr, i32** %cursor
    br label %loop
done:
    ret void
}
"""

app = Typer()


class Transition(t.NamedTuple):
    to_state: str
    to_symbol: str
    direction: t.Literal["L", "R"]


@app.command()
def main(input_path: Path, input_tape: str) -> None:
    binding.parse_assembly(PRELUDE).verify()

    transitions = parse(input_path)
    module = compile(transitions, input_tape)
    my_ir = (PRELUDE + "\n" + str(module)).splitlines()
    my_ir = [
        l for l in my_ir if ('declare void @"print_utf8"(i32 %".1")' not in l
        and 'declare void @"print_utf8_slice"(i32* %".1")' not in l
        and "target" not in l)
    ]
    my_ir = "\n".join(my_ir)
    output_path = input_path.with_suffix(".ll")
    output_path.write_text(my_ir)


def compile(transitions: t.Dict[str, t.Dict[str, Transition]], input_tape: str) -> ir.Module:
    char = ir.IntType(32)  # unicode scalar
    i32 = ir.IntType(32)
    usize = ir.IntType(64)
    void = ir.VoidType()

    # Create module and main function
    module = ir.Module(name="turing_machine")
    func = ir.Function(
        module,
        ir.FunctionType(i32, ()),
        name="main",
    )

    # Build entry block
    entry_block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_block)

    # Declare print_utf8 function; it's defined in PRELUDE
    print_utf8_ty = ir.FunctionType(void, (char,))
    print_utf8 = ir.Function(module, print_utf8_ty, name="print_utf8")

    # Declare print_utf8_slice function; it's defined in PRELUDE
    print_utf8_slice_ty = ir.FunctionType(void, (ir.PointerType(char),))
    print_utf8_slice = ir.Function(module, print_utf8_slice_ty, name="print_utf8_slice")

    # Allocate tape and pointer
    tape = builder.alloca(char, size=usize(TAPE_SIZE), name="tape")
    ptr = builder.alloca(usize, name="ptr")
    builder.store(usize(TAPE_SIZE // 2), ptr)  # start in the middle of the tape

    # Index that will later be used to zero out the tape
    zero_idx = builder.alloca(usize, name="zero_tape.idx")
    builder.store(usize(0), zero_idx)

    # Indices for printing output
    print_start_idx = builder.alloca(usize, name="print_idx.start")
    builder.store(usize(0), print_start_idx)
    print_end_idx = builder.alloca(usize, name="print_idx.end")
    builder.store(usize(TAPE_SIZE), print_end_idx)

    # Zero out the tape
    zero_block = func.append_basic_block(name="zero_tape")
    builder.branch(zero_block)
    builder.position_at_start(zero_block)
    load_zero_idx = builder.load(zero_idx, name="zero_tape.idx.load")
    tape_ptr = builder.gep(tape, [load_zero_idx], name="zero_tape.ptr")
    builder.store(ir.Constant(char, ord("_")), tape_ptr)
    new_zero_idx = builder.add(load_zero_idx, usize(1), name="zero_tape.idx.inc")
    builder.store(new_zero_idx, zero_idx)
    inbounds = builder.icmp_unsigned("<", new_zero_idx, usize(TAPE_SIZE), name="inbounds")
    builder.cbranch(
        inbounds, zero_block, init_tape_block := func.append_basic_block(name="init_tape")
    )

    # Initialize tape with input_tape
    builder.position_at_start(init_tape_block)
    for i, c in enumerate(input_tape, start=TAPE_SIZE // 2):
        tape_ptr = builder.gep(tape, [usize(i)], name=f"init_tape.ptr.{i}")
        builder.store(ir.Constant(char, ord(c)), tape_ptr)

    unknown_block = func.append_basic_block(name="unknown_symbol")

    state_blocks = {
        state: [func.append_basic_block(name=f"state {state}")]
        for state in
        sorted(transitions, key=lambda s: -1 if s== "INIT" else 0)
    }
    state_blocks["HALT"] = [func.append_basic_block(name="state HALT")]

    cake = []

    for state, [block] in state_blocks.items():
        if state == "HALT":
            continue
        with builder.goto_block(block):
            ptr_val = builder.load(ptr, name="ptr_val")
            tape_ptr = builder.gep(tape, [ptr_val], name="tape_ptr")
            current_symbol = builder.load(tape_ptr, name="current_symbol")
            transition_dict = transitions[state]

            state_scalars = list(map(ord, state)) + [0]  # null-terminated
            state_global_var = ir.GlobalVariable(
                module, ir.ArrayType(char, len(state_scalars)), name=f"state_str_{state}"
            )
            state_global_var.linkage = "internal"
            state_global_var.global_constant = True
            state_global_var.initializer = ir.Constant(            ir
            .ArrayType(char, len(state_scalars)),
                [ir.Constant(char, c) for c in state_scalars],)
            state_global_ptr = builder.gep(state_global_var, [usize(0), usize(0)], name=f"state_ptr_{state}")
            cake.append((state_global_ptr, block))

            # Create switch instruction for current symbol
            switch_inst = builder.switch(current_symbol, unknown_block)
            for from_symbol, transition in transition_dict.items():
                to_state = transition.to_state
                to_symbol = transition.to_symbol
                direction = transition.direction

                to_block = func.append_basic_block(name=f"state {state} on {from_symbol}")
                switch_inst.add_case(ir.Constant(char, ord(from_symbol)), to_block)

                builder.position_at_start(to_block)
                builder.comment(
                    f"Transition on '{from_symbol}' to state '{to_state}', write '{to_symbol}', move '{direction}'"
                )
                # Write the new symbol
                builder.store(ir.Constant(char, ord(to_symbol)), tape_ptr)

                # Move the head
                if direction == "R":
                    new_ptr_val = builder.add(ptr_val, usize(1), name="ptr_inc")
                else:  # direction == "L"
                    new_ptr_val = builder.sub(ptr_val, usize(1), name="ptr_dec")
                builder.store(new_ptr_val, ptr)

                # Transition to the next state
                builder.branch(state_blocks[to_state][0])


    with builder.goto_block(init_tape_block):
        builder.branch(state_blocks["INIT"][0])

    # Build unknown symbol block, using phi
    builder.position_at_start(unknown_block)
    phi_state = builder.phi(ir.PointerType(char), name="unknown_symbol_state")
    for s, b in cake:
        phi_state.add_incoming(s, b)
    builder.comment("Unknown symbol encountered; printing current state and symbol.")
    for c in map(ord, "UNKNOWN SYMBOL '"):
        builder.call(print_utf8, [ir.Constant(char, c)])
    ptr_val = builder.load(ptr, name="ptr_val")
    tape_ptr = builder.gep(tape, [ptr_val], name="tape_ptr")
    current_symbol = builder.load(tape_ptr, name="current_symbol")
    builder.call(print_utf8, [current_symbol])
    for c in map(ord, "' IN STATE "):
        builder.call(print_utf8, [ir.Constant(char, c)])
    builder.call(print_utf8_slice, [phi_state])
    for c in map(ord, "\n"):
        builder.call(print_utf8, [ir.Constant(char, c)])

    builder.ret(i32(1))

    builder.position_at_start(block_halt := state_blocks["HALT"][0])

    # Skip over leading _
    load_start_idx = builder.load(print_start_idx, name="load_print_start_idx")
    tape_ptr = builder.gep(tape, [load_start_idx], name="tape_ptr")
    current_symbol = builder.load(tape_ptr, name="current_symbol")
    is_underscore = builder.icmp_unsigned(
        "==", current_symbol, ir.Constant(char, ord("_")), name="is_underscore"
    )
    builder.cbranch(
        is_underscore,
        increment_block := func.append_basic_block(name="increment_print_start_idx"),
        trim_trailing_block := func.append_basic_block(name="trim_trailing"),
    )
    builder.position_at_start(increment_block)
    new_print_idx = builder.add(load_start_idx, usize(1), name="new_print_idx")
    builder.store(new_print_idx, print_start_idx)
    builder.branch(block_halt)

    # Calculate end index value (skipping trailing _)
    builder.position_at_start(trim_trailing_block)
    load_end_idx = builder.load(print_end_idx, name="load_print_end_idx")
    tape_ptr = builder.gep(tape, [builder.sub(load_end_idx, usize(1))], name="tape_ptr")
    current_symbol_end = builder.load(tape_ptr, name="current_symbol_end")
    is_underscore_end = builder.icmp_unsigned(
        "==", current_symbol_end, ir.Constant(char, ord("_")), name="is_underscore_end"
    )
    builder.cbranch(
        is_underscore_end,
        decrement_block := func.append_basic_block(name="decrement_print_end_idx"),
        print_loop_block := func.append_basic_block(name="print_loop"),
    )
    builder.position_at_start(decrement_block)
    new_end_idx = builder.sub(load_end_idx, usize(1), name="new_end_idx")
    builder.store(new_end_idx, print_end_idx)
    builder.branch(trim_trailing_block)

    # Print from print_idx to end_idx
    builder.position_at_start(print_loop_block)
    load_start_idx = builder.load(print_start_idx, name="load_print_start_idx")
    load_end_idx = builder.load(print_end_idx, name="load_print_end_idx")
    is_done_printing = builder.icmp_unsigned(
        ">=", load_start_idx, load_end_idx, name="is_done_printing"
    )
    builder.cbranch(
        is_done_printing,
        done_printing_block := func.append_basic_block(name="done_printing"),
        do_print_block := func.append_basic_block(name="do_print"),
    )
    builder.position_at_start(do_print_block)
    tape_ptr_print = builder.gep(tape, [load_start_idx], name="tape_ptr")
    current_symbol = builder.load(tape_ptr_print, name="current_symbol")
    builder.call(print_utf8, [current_symbol])
    new_print_idx = builder.add(load_start_idx, usize(1), name="new_print_idx")
    builder.store(new_print_idx, print_start_idx)
    builder.branch(print_loop_block)
    builder.position_at_start(done_printing_block)

    builder.ret(i32(0))

    return module


def parse(input_path: Path) -> t.Dict[str, t.Dict[str, Transition]]:
    transitions = {}
    found_init = False
    found_halt = False
    for line in input_path.read_text().splitlines():
        # Remove comments
        line = line.split("//")[0].strip()
        if not line:
            continue
        from_state, from_symbol, to_state, to_symbol, direction = line.split()
        assert direction in ("L", "R"), "Direction must be 'L' or 'R'"
        found_init |= from_state == "INIT"
        found_halt |= to_state == "HALT"
        transitions.setdefault(from_state, {})[from_symbol] = Transition(
            to_state, to_symbol, direction
        )
    assert found_init, "No INIT state found"
    assert found_halt, "No HALT state found"
    return transitions


if __name__ == "__main__":
    app()
