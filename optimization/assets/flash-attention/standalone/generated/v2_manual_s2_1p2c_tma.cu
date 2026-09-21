
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

#include "cutlass/arch/barrier.h"
#include "cutlass/cutlass.h"
#include <cutlass/arch/reg_reconfig.h>

#ifdef __CUDACC__
#pragma nv_diag_suppress 20054
#endif
// include the choreo header;
#include "choreo.h"
namespace cde = cuda::device::experimental;
#include <cooperative_groups.h>
using namespace choreo;
using Barrier = cutlass::arch::ClusterTransactionBarrier;

template <bool ZeroFirst = false>
__device__ static __forceinline__ void __choreo_wgmma_group_bf16_m64n128k16_ss_k128(uint64_t a_desc, uint64_t b_desc, float* d) {
  uint64_t a0 = a_desc + 0x0;
  uint64_t a1 = a_desc + 0x2;
  uint64_t a2 = a_desc + 0x4;
  uint64_t a3 = a_desc + 0x6;
  uint64_t a4 = a_desc + 0x400;
  uint64_t a5 = a_desc + 0x402;
  uint64_t a6 = a_desc + 0x404;
  uint64_t a7 = a_desc + 0x406;
  uint64_t b0 = b_desc + 0x0;
  uint64_t b1 = b_desc + 0x2;
  uint64_t b2 = b_desc + 0x4;
  uint64_t b3 = b_desc + 0x6;
  uint64_t b4 = b_desc + 0x400;
  uint64_t b5 = b_desc + 0x402;
  uint64_t b6 = b_desc + 0x404;
  uint64_t b7 = b_desc + 0x406;
  asm volatile(
      "{\n\t"
      ".reg .pred p0, p1;\n\t"
      "setp.eq.b32 p0, %80, 0;\n\t"
      "setp.ne.b32 p1, 1, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16."
      "bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, %72, p0, 1, 1, 0, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16."
      "bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %65, %73, p1, 1, 1, 0, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16."
      "bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %66, %74, p1, 1, 1, 0, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16."
      "bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %67, %75, p1, 1, 1, 0, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16."
      "bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %68, %76, p1, 1, 1, 0, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16."
      "bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %69, %77, p1, 1, 1, 0, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16."
      "bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %70, %78, p1, 1, 1, 0, 0;\n\t"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16."
      "bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %71, %79, p1, 1, 1, 0, 0;\n\t"
      "}"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "l"(a0), "l"(a1), "l"(a2), "l"(a3), "l"(a4), "l"(a5), "l"(a6), "l"(a7), "l"(b0), "l"(b1), "l"(b2), "l"(b3), "l"(b4), "l"(b5), "l"(b6), "l"(b7), "n"(ZeroFirst ? 1 : 0));
}

__device__ static __forceinline__ void __choreo_wgmma_group_bf16_m64n128k16_rs_k128(const bf16* a, uint64_t b_desc, float* d) {
  using Op = cute::SM90::GMMA::MMA_64x128x16_F32BF16BF16_RS<
      cute::SM90::GMMA::Major::K, cute::SM90::GMMA::Major::MN>;
  Op::fma(reinterpret_cast<const uint32_t*>(a + 7 * 8)[0],
          reinterpret_cast<const uint32_t*>(a + 7 * 8)[1],
          reinterpret_cast<const uint32_t*>(a + 7 * 8)[2],
          reinterpret_cast<const uint32_t*>(a + 7 * 8)[3],
          b_desc + 0x380, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
  Op::fma(reinterpret_cast<const uint32_t*>(a + 6 * 8)[0],
          reinterpret_cast<const uint32_t*>(a + 6 * 8)[1],
          reinterpret_cast<const uint32_t*>(a + 6 * 8)[2],
          reinterpret_cast<const uint32_t*>(a + 6 * 8)[3],
          b_desc + 0x300, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
  Op::fma(reinterpret_cast<const uint32_t*>(a + 5 * 8)[0],
          reinterpret_cast<const uint32_t*>(a + 5 * 8)[1],
          reinterpret_cast<const uint32_t*>(a + 5 * 8)[2],
          reinterpret_cast<const uint32_t*>(a + 5 * 8)[3],
          b_desc + 0x280, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
  Op::fma(reinterpret_cast<const uint32_t*>(a + 4 * 8)[0],
          reinterpret_cast<const uint32_t*>(a + 4 * 8)[1],
          reinterpret_cast<const uint32_t*>(a + 4 * 8)[2],
          reinterpret_cast<const uint32_t*>(a + 4 * 8)[3],
          b_desc + 0x200, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
  Op::fma(reinterpret_cast<const uint32_t*>(a + 0 * 8)[0],
          reinterpret_cast<const uint32_t*>(a + 0 * 8)[1],
          reinterpret_cast<const uint32_t*>(a + 0 * 8)[2],
          reinterpret_cast<const uint32_t*>(a + 0 * 8)[3],
          b_desc + 0x0, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
  Op::fma(reinterpret_cast<const uint32_t*>(a + 1 * 8)[0],
          reinterpret_cast<const uint32_t*>(a + 1 * 8)[1],
          reinterpret_cast<const uint32_t*>(a + 1 * 8)[2],
          reinterpret_cast<const uint32_t*>(a + 1 * 8)[3],
          b_desc + 0x80, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
  Op::fma(reinterpret_cast<const uint32_t*>(a + 2 * 8)[0],
          reinterpret_cast<const uint32_t*>(a + 2 * 8)[1],
          reinterpret_cast<const uint32_t*>(a + 2 * 8)[2],
          reinterpret_cast<const uint32_t*>(a + 2 * 8)[3],
          b_desc + 0x100, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
  Op::fma(reinterpret_cast<const uint32_t*>(a + 3 * 8)[0],
          reinterpret_cast<const uint32_t*>(a + 3 * 8)[1],
          reinterpret_cast<const uint32_t*>(a + 3 * 8)[2],
          reinterpret_cast<const uint32_t*>(a + 3 * 8)[3],
          b_desc + 0x180, d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15], d[16], d[17], d[18], d[19], d[20], d[21], d[22], d[23], d[24], d[25], d[26], d[27], d[28], d[29], d[30], d[31], d[32], d[33], d[34], d[35], d[36], d[37], d[38], d[39], d[40], d[41], d[42], d[43], d[44], d[45], d[46], d[47], d[48], d[49], d[50], d[51], d[52], d[53], d[54], d[55], d[56], d[57], d[58], d[59], d[60], d[61], d[62], d[63]);
}

// Warp-specialized 1p2c with 2-stage pipeline, TMA + 128B swizzle
// D=128, bf16, causal prefill
//
// Compared to v1_manual_baseline (sequential dma.copy):
//   - Async TMA loads for Q, K, V (single-thread producer)
//   - 1p2c warp specialization: 1 producer warpgroup + 2 consumer warpgroups
//   - 2-stage double buffering with shared events for K and V
//   - Producer/consumer overlap hides memory latency
//   - 128B swizzle on TMA and WGMMA to avoid bank conflicts
//
// The compiler auto-splits the 128-element inner dim into [2, 64] groups
// so that each TMA box fits the 128B swizzle constraint (64 * 2B = 128B).

#define BLOCK_M 128
#define BLOCK_N 128
#define WG_M 64
#define WG_K 16
#define SWIZ 128
#define STAGES 2

#define DIM 128

__global__ __launch_bounds__(384, 1) void __choreo_device_flash_atten(bf16 * Q, bf16 * K, bf16 * V, bf16 * O, float scale, unsigned B, unsigned H, unsigned KV_SEQ, unsigned Q_SEQ, const __grid_constant__ CUtensorMap __choreo_tma_0_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_1_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_2_tensor_map, const __grid_constant__ CUtensorMap __choreo_tma_3_tensor_map) {
  extern __shared__ char __choreo_device_flash_atten__runtime_shared_buffer__raw[];
  auto __choreo_device_flash_atten__runtime_shared_buffer__ = reinterpret_cast<char*>(aligned_up_ptr<128 * 8>(__choreo_device_flash_atten__runtime_shared_buffer__raw));
  { // parallel-by: kernels/v2_manual_s2_1p2c_tma.co:32.18
  auto anon_1 = (unsigned char*)__choreo_device_flash_atten__runtime_shared_buffer__;
  [[maybe_unused]] auto Q_head = (H * Q_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + Q);
  [[maybe_unused]] auto K_head = (H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + K);
  [[maybe_unused]] auto V_head = (H * KV_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + V);
  [[maybe_unused]] auto O_head = (H * Q_SEQ * (blockIdx.z * 128) + blockIdx.y * 128 + O);
  bf16* q_shared = (bf16*)(anon_1 + 163840);
  bf16* k_buf = (bf16*)(anon_1 + 0);
  bf16* v_buf = (bf16*)(anon_1 + 65536);
  __shared__ __align__(16) uint64_t qf__mem; // raw mbarrier storage
  Barrier& qf = *reinterpret_cast<Barrier*>(&qf__mem);
  unsigned int qf__phase = 0; // implicit scalar event generation
  __shared__ __align__(16) uint64_t kf__mem[2]; // raw mbarrier storage
  Barrier* kf = reinterpret_cast<Barrier*>(kf__mem);
  __shared__ __align__(16) uint64_t ke__mem[2]; // raw mbarrier storage
  Barrier* ke = reinterpret_cast<Barrier*>(ke__mem);
  __shared__ __align__(16) uint64_t vf__mem[2]; // raw mbarrier storage
  Barrier* vf = reinterpret_cast<Barrier*>(vf__mem);
  __shared__ __align__(16) uint64_t ve__mem[2]; // raw mbarrier storage
  Barrier* ve = reinterpret_cast<Barrier*>(ve__mem);
  if (threadIdx.x == 0) {
    cute::prefetch_tma_descriptor(&__choreo_tma_0_tensor_map);
    cute::prefetch_tma_descriptor(&__choreo_tma_1_tensor_map);
    cute::prefetch_tma_descriptor(&__choreo_tma_2_tensor_map);
    cute::prefetch_tma_descriptor(&__choreo_tma_3_tensor_map);
  }
  if (threadIdx.x == 0) {
    qf.init(1);
    kf[0].init(1);
    kf[1].init(1);
    ke[0].init(256);
    ke[1].init(256);
    vf[0].init(1);
    vf[1].init(1);
    ve[0].init(256);
    ve[1].init(256);
  }
  cde::fence_proxy_async_shared_cta();
  __syncthreads();
  int kv_bound = ((blockIdx.x + 1) * 128 + 128 - 1) / 128;
  [[maybe_unused]] auto __choreo_vg4id_x = threadIdx.x / 128;
  [[maybe_unused]] auto __choreo_vtid_x = threadIdx.x % 128;
  // inthreads: kernels/v2_manual_s2_1p2c_tma.co:46.7
  if ((__choreo_vg4id_x == 0)) {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    if ((threadIdx.x == 0)) {
      qf.arrive_and_expect_tx(32768);
      choreo::tma_load_3d_shared_cta_global_mbarrier((void*)(q_shared), (const void*)&__choreo_tma_0_tensor_map, (uint64_t*)&(qf), (blockIdx.y * 128 % 64), (Q_SEQ * blockIdx.z + blockIdx.x * 128), (blockIdx.y * 2));
    }
    // trigger-after folded into native async completion
    // with-in: kernels/v2_manual_s2_1p2c_tma.co:52.9
    {
      int __iv_bn = 0;
      // foreach: kernels/v2_manual_s2_1p2c_tma.co:52.9
      for (__iv_bn = 0; __iv_bn < (((blockIdx.x + 1) * 128 + 127) / 128); ++__iv_bn) {
        int stage = __iv_bn % 2;
        // wait event(raw mbarrier) ke.at(bn)
        ke[(__iv_bn) % 2].wait(((((__iv_bn) / 2) & 1)));
        if ((threadIdx.x == 0)) {
          kf[(__iv_bn) % 2].arrive_and_expect_tx(32768);
          choreo::tma_load_3d_shared_cta_global_mbarrier((void*)(k_buf + (stage)*(16384)), (const void*)&__choreo_tma_1_tensor_map, (uint64_t*)&(kf[(__iv_bn) % 2]), (blockIdx.y * 128 % 64), (KV_SEQ * blockIdx.z + __iv_bn * 128), (blockIdx.y * 2));
        }
        // trigger-after folded into native async completion
        // wait event(raw mbarrier) ve.at(bn)
        ve[(__iv_bn) % 2].wait(((((__iv_bn) / 2) & 1)));
        if ((threadIdx.x == 0)) {
          vf[(__iv_bn) % 2].arrive_and_expect_tx(32768);
          choreo::tma_load_3d_shared_cta_global_mbarrier((void*)(v_buf + (stage)*(16384)), (const void*)&__choreo_tma_2_tensor_map, (uint64_t*)&(vf[(__iv_bn) % 2]), (blockIdx.y * 128 % 64), (KV_SEQ * blockIdx.z + __iv_bn * 128), (blockIdx.y * 2));
        }
        // trigger-after folded into native async completion
      } // bn
      __iv_bn = 0;
    }
  } // end inthreads
  // inthreads: kernels/v2_manual_s2_1p2c_tma.co:66.7
  if ((__choreo_vg4id_x > 0)) {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
    int cid = __choreo_vg4id_x - 1;
    // with-in: kernels/v2_manual_s2_1p2c_tma.co:69.9
    {
      int __iv_s = 0;
      // foreach: kernels/v2_manual_s2_1p2c_tma.co:69.9
      for (__iv_s = 0; __iv_s < 2; ++__iv_s) {
        // trigger event(barrier) ke.at(s)
        (void)ke[(__iv_s) % 2].arrive();
        // trigger event(barrier) ve.at(s)
        (void)ve[(__iv_s) % 2].arrive();
      } // s
      __iv_s = 0;
    }
    qf.wait(qf__phase); // wait event(raw mbarrier)
    qf__phase ^= 1;
    float scores_max[2];
    for (int __frag_init = 0; __frag_init < 2; ++__frag_init)
      scores_max[__frag_init] = (-INFINITY);
    float scores_max_prev[2];
    float scores_scale[2];
    float scores_sum[2];
    float logsum[2];
    for (int __frag_init = 0; __frag_init < 2; ++__frag_init)
      logsum[__frag_init] = 0.000000f;
    float acc_o[64];
    float __frag_init_val0 = 0.000000f;
    for (int idx = 0; idx < 64; ++idx)
      acc_o[idx] = __frag_init_val0;
    // with-in: kernels/v2_manual_s2_1p2c_tma.co:83.9
    {
      int __iv_bn = 0;
      // foreach: kernels/v2_manual_s2_1p2c_tma.co:83.9
      for (__iv_bn = 0; __iv_bn < (((blockIdx.x + 1) * 128 + 127) / 128); ++__iv_bn) {
        auto stage = __iv_bn % 2;
        float acc_s[64];
        // wait event(raw mbarrier) kf.at(bn)
        kf[(__iv_bn) % 2].wait(((((__iv_bn) / 2) & 1)));
        warpgroup_fence_operand(acc_s);
        warpgroup_arrive();
        auto* __choreo_wgmma_ptr_0 = (bf16*)((q_shared + ((__choreo_vg4id_x - 1) * 4096)));
        uint64_t __choreo_wgmma_desc_0 = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(__choreo_wgmma_ptr_0);
        auto* __choreo_wgmma_ptr_1 = (bf16*)(k_buf + (stage)*(16384));
        uint64_t __choreo_wgmma_desc_1 = wgmma_make_smem_desc<WGMMA_MajorOrder::K_MAJOR, WGMMA_Swizzle::B128>(__choreo_wgmma_ptr_1);
        __choreo_wgmma_group_bf16_m64n128k16_ss_k128<true>(__choreo_wgmma_desc_0, __choreo_wgmma_desc_1, acc_s);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_s);
        // trigger event(barrier) ke.at(bn)
        (void)ke[(__iv_bn) % 2].arrive();
        int causal_row_base = (blockIdx.x * 128 + (__choreo_vg4id_x - 1) * 64);
        // if-else: kernels/v2_manual_s2_1p2c_tma.co:92.11
        if (((__iv_bn + 1) * 128 > causal_row_base + 1)) {
          { // apply acc_s
            int __frag_iv_i = 0;
            int __frag_iv_j = 0;
            #pragma unroll
            for (int __r = 0; __r < 64; ++__r) {
              __frag_iv_i = ((threadIdx.x % 128) / 32 * 16 + ((__r) % 4) / 2 * 8 + (threadIdx.x % 128) % 32 / 4);
              __frag_iv_j = ((__r) / 4 * 8 + (threadIdx.x % 128) % 32 % 4 * 2 + (__r) % 2);
              // if-else: kernels/v2_manual_s2_1p2c_tma.co:94.15
              if (__iv_bn * 128 + __frag_iv_j > blockIdx.x * 128 + cid * 64 + __frag_iv_i) {
                acc_s[__r] = (-INFINITY);
              } // end if-else: kernels/v2_manual_s2_1p2c_tma.co:94.15
            }
          } // apply acc_s
        } // end if-else: kernels/v2_manual_s2_1p2c_tma.co:92.11
        { // frag.copy scores_max_prev, scores_max
          #pragma unroll
          for (int __r = 0; __r < 2; ++__r)
            scores_max_prev[__r] = scores_max[__r];
        } // frag.copy
        { // frag.reduce_max acc_s -> scores_max
          #pragma unroll
          for (int __row = 0; __row < 2; ++__row) {
            float __local_reduce = acc_s[(((0 / 2) * 4) + (__row * 2) + (0 % 2))];
            #pragma unroll
            for (int __rv = 1; __rv < 32; ++__rv) {
              __local_reduce = fmaxf(__local_reduce, acc_s[(((__rv / 2) * 4) + (__row * 2) + (__rv % 2))]);
            }
            {
              float __shfl_v0 = __shfl_xor_sync(0xffffffff, __local_reduce, 2);
              __local_reduce = fmaxf(__local_reduce, __shfl_v0);
            }
            {
              float __shfl_v1 = __shfl_xor_sync(0xffffffff, __local_reduce, 1);
              __local_reduce = fmaxf(__local_reduce, __shfl_v1);
            }
            scores_max[__row] = __local_reduce;
          }
        } // frag.reduce_max
        { // apply scores_max
          #pragma unroll
          for (int __r = 0; __r < 2; ++__r) {
            scores_max[__r] = (choreo::nv_cute::numerics::max)(scores_max[__r], scores_max_prev[__r]);
            scores_scale[__r] = choreo::nv_cute::numerics::exp2f(scores_max_prev[__r] * scale - scores_max[__r] * scale);
          }
        } // apply scores_max
        { // apply acc_s
          #pragma unroll
          for (int __r = 0; __r < 64; ++__r) {
            acc_s[__r] = choreo::nv_cute::numerics::exp2f(acc_s[__r] * scale - scores_max[(((__r) & 3) >> 1)] * scale);
          }
        } // apply acc_s
        { // frag.reduce_sum acc_s -> scores_sum
          #pragma unroll
          for (int __row = 0; __row < 2; ++__row) {
            float __local_reduce = 0.0f;
            #pragma unroll
            for (int __rv = 0; __rv < 32; ++__rv) {
              __local_reduce += acc_s[(((__rv / 2) * 4) + (__row * 2) + (__rv % 2))];
            }
            {
              float __shfl_v0 = __shfl_xor_sync(0xffffffff, __local_reduce, 2);
              __local_reduce = __local_reduce + __shfl_v0;
            }
            {
              float __shfl_v1 = __shfl_xor_sync(0xffffffff, __local_reduce, 1);
              __local_reduce = __local_reduce + __shfl_v1;
            }
            scores_sum[__row] = __local_reduce;
          }
        } // frag.reduce_sum
        { // apply logsum
          #pragma unroll
          for (int __r = 0; __r < 2; ++__r) {
            logsum[__r] = logsum[__r] * scores_scale[__r] + scores_sum[__r];
          }
        } // apply logsum
        { // apply acc_o
          #pragma unroll
          for (int __r = 0; __r < 64; ++__r) {
            acc_o[__r] = acc_o[__r] * scores_scale[(((__r) & 3) >> 1)];
          }
        } // apply acc_o
        bf16 acc_s_cast[64];
        { // apply acc_s_cast
          #pragma unroll
          for (int __r = 0; __r < 64; ++__r) {
            acc_s_cast[__r] = choreo::f32_to_bf16(acc_s[__r]);
          }
        } // apply acc_s_cast
        // wait event(raw mbarrier) vf.at(bn)
        vf[(__iv_bn) % 2].wait(((((__iv_bn) / 2) & 1)));
        warpgroup_fence_operand(acc_o);
        warpgroup_arrive();
        warpgroup_fence_operand(acc_s_cast);
        auto* __choreo_wgmma_ptr_2 = (bf16*)(v_buf + (stage)*(16384));
        uint64_t __choreo_wgmma_desc_2 = wgmma_make_smem_desc<WGMMA_MajorOrder::MN_MAJOR, WGMMA_Swizzle::B128, 16384>(__choreo_wgmma_ptr_2);
        __choreo_wgmma_group_bf16_m64n128k16_rs_k128(acc_s_cast, __choreo_wgmma_desc_2, acc_o);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        warpgroup_fence_operand(acc_o);
        // trigger event(barrier) ve.at(bn)
        (void)ve[(__iv_bn) % 2].arrive();
      } // bn
      __iv_bn = 0;
    }
    { // apply acc_o
      #pragma unroll
      for (int __r = 0; __r < 64; ++__r) {
        acc_o[__r] = acc_o[__r] / logsum[(((__r) & 3) >> 1)];
      }
    } // apply acc_o
    bf16* o_buf = (bf16*)(anon_1 + 131072);
    auto __shape1_o_buf = cute::make_shape(cute::Int<64>{}, cute::Int<128>{});
    auto __stride1_o_buf = cute::make_stride(cute::Int<128>{}, cute::Int<1>{});
    auto __layout1_o_buf = cute::make_layout(__shape1_o_buf, __stride1_o_buf);
    auto __tensor1_o_buf = cute::make_tensor(cute::make_smem_ptr<bf16>((bf16*)o_buf + ((__choreo_vg4id_x - 1) * 8192)), __layout1_o_buf);
    store_fragment_d_stmatrix_f32_bf16<CUTE_WGMMA_M64K16, 128>(__tensor1_o_buf, reinterpret_cast<float*>(acc_o));
    choreo::named_barrier_sync(256, 15);
    cde::fence_proxy_async_shared_cta();
    choreo::named_barrier_sync(256, 15);
    if ((threadIdx.x == 128 || threadIdx.x == 256) && __choreo_vg4id_x == 1) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&__choreo_tma_3_tensor_map, (blockIdx.y * 128), (Q_SEQ * blockIdx.z + blockIdx.x * 128), o_buf);
      cde::cp_async_bulk_commit_group();
      cde::cp_async_bulk_wait_group_read<0>();
    }
  } // end inthreads
  } // end parallel-by
}

void flash_atten(const choreo::spanned_view<choreo::bf16, 4> & Q, const choreo::spanned_view<choreo::bf16, 4> & K, const choreo::spanned_view<choreo::bf16, 4> & V, const choreo::spanned_view<choreo::bf16, 4> & O) {
  auto &B = Q.shape()[0];
  auto &H = Q.shape()[2];
  auto &KV_SEQ = K.shape()[1];
  auto &Q_SEQ = Q.shape()[1];
  float scale = 0.127517f;
  uint64_t __choreo_tma_0_shape[] = {64, (B * H * Q_SEQ * 128 / (H * 128)), (H * 2)};
  uint64_t __choreo_tma_0_strides[] = {(H * 256), 128};
  uint32_t __choreo_tma_0_box_shape[] = {(uint32_t)(64), (uint32_t)(128), (uint32_t)(2)};
  uint32_t __choreo_tma_0_elem_strides[] = {1, 1, 1};
  alignas(64) CUtensorMap __choreo_tma_0_tensor_map{};
  CUresult __choreo_tma_0_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_0_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          3,
          Q.data(),
          __choreo_tma_0_shape,
          __choreo_tma_0_strides,
          __choreo_tma_0_box_shape,
          __choreo_tma_0_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_0_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_1_shape[] = {64, (B * H * KV_SEQ * 128 / (H * 128)), (H * 2)};
  uint64_t __choreo_tma_1_strides[] = {(H * 256), 128};
  uint32_t __choreo_tma_1_box_shape[] = {(uint32_t)(64), (uint32_t)(128), (uint32_t)(2)};
  uint32_t __choreo_tma_1_elem_strides[] = {1, 1, 1};
  alignas(64) CUtensorMap __choreo_tma_1_tensor_map{};
  CUresult __choreo_tma_1_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_1_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          3,
          K.data(),
          __choreo_tma_1_shape,
          __choreo_tma_1_strides,
          __choreo_tma_1_box_shape,
          __choreo_tma_1_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_1_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_2_shape[] = {64, (B * H * KV_SEQ * 128 / (H * 128)), (H * 2)};
  uint64_t __choreo_tma_2_strides[] = {(H * 256), 128};
  uint32_t __choreo_tma_2_box_shape[] = {(uint32_t)(64), (uint32_t)(128), (uint32_t)(2)};
  uint32_t __choreo_tma_2_elem_strides[] = {1, 1, 1};
  alignas(64) CUtensorMap __choreo_tma_2_tensor_map{};
  CUresult __choreo_tma_2_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_2_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          3,
          V.data(),
          __choreo_tma_2_shape,
          __choreo_tma_2_strides,
          __choreo_tma_2_box_shape,
          __choreo_tma_2_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_2_tensor_map_res != CUDA_SUCCESS);
  uint64_t __choreo_tma_3_shape[] = {(H * 128), (B * H * Q_SEQ * 128 / (H * 128))};
  uint64_t __choreo_tma_3_strides[] = {(H * 256)};
  uint32_t __choreo_tma_3_box_shape[] = {(uint32_t)(128), (uint32_t)(128)};
  uint32_t __choreo_tma_3_elem_strides[] = {1, 1};
  alignas(64) CUtensorMap __choreo_tma_3_tensor_map{};
  CUresult __choreo_tma_3_tensor_map_res = cuTensorMapEncodeTiled(
          &__choreo_tma_3_tensor_map,
          CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
          2,
          O.data(),
          __choreo_tma_3_shape,
          __choreo_tma_3_strides,
          __choreo_tma_3_box_shape,
          __choreo_tma_3_elem_strides,
          CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
          CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
          CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
          CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  choreo::abend_true(__choreo_tma_3_tensor_map_res != CUDA_SUCCESS);
  dim3 __flash_atten_gdims0(((Q_SEQ + 127) / 128), H, B);
  dim3 __flash_atten_bdims0(384, 1, 1);
  cudaFuncSetAttribute(__choreo_device_flash_atten, cudaFuncAttributeMaxDynamicSharedMemorySize, 196608 + (128 - 1));
  __choreo_device_flash_atten<<<__flash_atten_gdims0, __flash_atten_bdims0, 196608 + (128 - 1)>>>(Q.data(), K.data(), V.data(), O.data(), scale, B, H, KV_SEQ, Q_SEQ, __choreo_tma_0_tensor_map, __choreo_tma_1_tensor_map, __choreo_tma_2_tensor_map, __choreo_tma_3_tensor_map);
}




#define MHA_DIM DIM
#define MHA_DTYPE choreo::bf16
#include "../fa_helper.hpp"

int main() {
  #include "../bench_configs.inc"
  return mha_helper::RunBenchmarks(
      "Choreo Flash Attention v2 Manual S2 1p2c TMA (D=128 bf16 causal)",
      kBenchConfigs, kBenchConfigCount,
      [&](const mha_helper::BenchConfig& cfg,
          const mha_helper::TensorViews& views) {
        (void)cfg;
        flash_atten(views.Q, views.K, views.V, views.O);
      });
}

