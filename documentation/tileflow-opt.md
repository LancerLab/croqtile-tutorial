# Tiling Optimisation in Choreo

In this section, we will extend our discussion on tiling optimization in Choreo by examining some practical examples.

## Example 1: `matmul`
The first example is matrix multiplication on a device program. This example demonstrates how Choreo’s tiling and memory management mechanisms work together to optimize matrix operations, including how DMA (Direct Memory Access) is leveraged for efficient data movement.

---

### Example Code: Matrix Multiplication with Tiling

```choreo
__co__ s32 [512, 1024] matmul(s32 [512, 1024] lhs, s32 [1024, 1024] rhs) { 
  // Device program
  s32[lhs.span(0), rhs.span(1)] output; // Use same shape as lhs

  parallel p by 1 {
    with index = {m_tile, k_tile, n_tile} in [4, 4, 8] {
      foreach m_tile, n_tile {
        shared s32[128, 128] l2_out;
        foreach k_tile {
          // Initiate DMA operations
          lhs_load = dma.copy.async lhs.chunkat(m_tile, k_tile) => shared;
          rhs_load = dma.copy.async rhs.chunkat(k_tile, n_tile) => shared;
          out_store_s = dma.any;
          parallel q by 1 {
            with index = {m_tile_s, k_tile_s, n_tile_s} in [4, 4, 8] {
              // Local buffers for tiling
              local s32 [32, 64] lhs_load_s_buffer_0 {0}, lhs_load_s_buffer_1 {0};
              local s32 [64, 16] rhs_load_s_buffer_0 {0}, rhs_load_s_buffer_1 {0};
              foreach index {
                // Add "after" primitive to chain up paired DMA operations
                lhs_load_s = dma.copy.async lhs_load.data.chunkat(m_tile_s, k_tile_s) => select(k_tile_s % 2, lhs_load_s_buffer_0, lhs_load_s_buffer_1) after lhs_load;
                rhs_load_s = dma.copy.async rhs_load.data.chunkat(k_tile_s, n_tile_s) => select(k_tile_s % 2, rhs_load_s_buffer_0, rhs_load_s_buffer_1) after rhs_load;
                out_load_s = dma.copy.async l2_out.chunkat(m_tile_s, n_tile_s) => local;
                wait out_load_s;

                // Perform the matrix multiplication (dot product) operation
                wait lhs_load_s, rhs_load_s;
                call dot_general_fp32(lhs_load_s.data, rhs_load_s.data, 32, 16, 64, 0, 8, out_load_s.data);

                // Store the result back to l2_out
                out_store_s = dma.copy.async out_load_s.data => l2_out.chunkat(m_tile_s, n_tile_s); 
              }
            }
          }

          // Add "after" primitive to chain up DMA operations for the final output
          out_store = dma.copy.async l2_out => output.chunkat(m_tile, n_tile) after out_store_s;
          wait out_store;
        }
      }
    }
  }
  return output;
}
```
---

### Memory Tiling with `chunkat`

In the matrix multiplication example, **memory tiling** is achieved using the `chunkat` method, which slices the input matrices `lhs` (left-hand side) and `rhs` (right-hand side) into smaller chunks to be processed in parallel.

##### Tiling via `chunkat`:

- The `lhs` and `rhs` matrices are tiled using the `chunkat` method. The chunks are defined by `m_tile`, `k_tile`, and `n_tile`, which represent different dimensions of the matrices involved in the multiplication.
- The chunks for both matrices are loaded asynchronously into the **shared storage** (`lhs_load` and `rhs_load`) with the `dma.copy.async` operation. This operation loads a portion of the matrix from **global storage** to **shared storage**, which is a common optimization in large matrix operations to improve data locality.

##### Bounded Variables:

- The variables `m_tile`, `k_tile`, and `n_tile` are **bounded variables**. They are defined in the context of `parallel-by` and `with-in` constructs, which define the iteration space for the tiling.
- These bounded variables allow the chunks to be computed in parallel, where each thread works on a different slice of the matrix. This parallelization enables efficient computation of matrix multiplications across multiple threads.


## Example 2: rgb2gray
The second example is the RGB to Grayscale conversion algorithm. The algorithm converts a color (RGB) image to a grayscale image using a weighted average method. This example shows how physical parallelism work together with virtual parallelism.

---
### Example Code

```choreo
__co__ auto rgb2gray(f32 [N, 3, H, W] input) {
  f32 [N, H, W] out;
  parallel q by 6 {
    with index={n, h, w} in [N, H, W]/{#q, 16, 512} {
      foreach n, h, w {
        // _ means no tiling
        input_L1_A = dma.copy input.chunkat(q#n, _, h, w) => local;
        dims : input_L1_A.span[(0), (2), (3)];
        local f32 [dims] out_L1;
        call rgb2gray_kernel_fp32_fp32(input_L1_A.data, out_L1, |out_L1|, 1);
        dma.copy out_L1 => out.chunkat(q#n, h, w);
      }
    }
  }
  return out;
}
```
---
### Compose Two Bounded Variables for Tiling with `#`

In the example above, there is a need to tile the matrices in a manner that involves **two bounded variables**. This is where the **`#` operator** comes into play, allowing the program to compute the **Cartesian product** of two bounded variables (**not commutative**).

##### The `#` Operator:

- The code defines the tiling parameters with four bounded variables: `q`, `n`, `h`, and `w`, whose upper bounders are `6`, `N/6`, `H/16`, and `W/512`.
- Suppose `N` is 18, `q` represents physical thread index. We want each thread to process data that is contiguous in the first dimension. That is, the first thread handles input[0~2][xxx], the second thread handles input[3~5][xxx], and so on.
- `q # n` will result in a new bounded variable implicitly, whose upper bound is `N`: bound of `q` multiply bound of `n`.
- For every data move, the stride of each dimension is 1, 3, 16, 512. In the last dimension, the index is `w`. In the first dimension, the index is `q * (bound of n) + n`. 


##### Why this is Important:

- The combination of `q` and `n` ensures that physical and virtual parallelism can work together efficiently.
- This helps optimize **memory access patterns**, especially when working with large matrices, where efficient memory use is crucial.
- Choreo’s approach here is more aligned with a **physical storage hierarchy** (local, shared, and global storage). Unlike higher-level loop scheduling techniques that abstract away the actual data movement, Choreo explicitly links **virtual loops** to **physical DMA operations**. This results in more efficient data movement and computation.
