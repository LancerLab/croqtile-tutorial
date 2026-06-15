### Synchronization and Asynchronous `parallel-by`

Similar to `inthreads`, `parallel-by` can also be made asynchronous by appending `.async`:

```choreo
parallel.async p by 6 : block {
  // asynchronous kernel launch (no host-side synchronization)
}
```

Without `.async`, the host waits for the kernel to finish before continuing:

```choreo
parallel p by 6 : block {
  // synchronous kernel launch (host blocks until completion)
}
// host continues only after the kernel finishes
```

With `.async`, the host code continues immediately after launching the kernel.
This is essential for overlapping computation with data transfers or launching
independent kernels concurrently.

### Stream Binding

CUDA streams allow multiple kernels and memory transfers to execute concurrently
on the GPU. In Choreo, you can bind a `parallel-by` block to a specific stream
using angle-bracket syntax:

```choreo
__co__ void foo(stream s0, stream s1, f32[M] a, f32[N] b) {
  parallel.async<s0> p by M : block {
    a.at(p) = 1.0f;
  }
  parallel.async<s1> q by N : block {
    b.at(q) = 2.0f;
  }
}
```

In this example:

- **`stream s0, stream s1`**: Stream parameters passed from the host.
- **`parallel.async<s0>`**: Launches the kernel asynchronously on stream `s0`.
- **`parallel.async<s1>`**: Launches a second kernel on stream `s1`.

Since both kernels are launched on different streams with `.async`, they can
execute concurrently on the GPU.

### Stream Binding without `.async`

Stream binding also works with synchronous `parallel-by`. In this case, the host
launches the kernel on the specified stream and then waits for that stream to
complete:

```choreo
__co__ void bar(stream _s, f32[M] a) {
  parallel<_s> p by M : block {
    a.at(p) = 1.0f;
  }
  // host synchronizes on stream _s before continuing
}
```

### Rules

- Stream binding is only allowed on **block-level** parallel (`: block`).
- The identifier inside `<...>` must be a `stream`-typed variable.
- Without stream binding, synchronous `parallel-by` uses device-wide
  synchronization, and asynchronous `parallel-by` uses the default stream.
