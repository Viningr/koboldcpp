R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(#pragma OPENCL EXTENSION cl_khr_subgroups : enable
)"
R"(
)"
R"(#ifdef cl_qcom_reqd_sub_group_size
)"
R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
R"(#define ADRENO_GPU 1
)"
R"(#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
)"
R"(#endif
)"
R"(
)"
R"(#define QK1_0 128
)"
R"(#define N_SIMDGROUP 4
)"
R"(
)"
R"(#define dequantizeBlockAccum_q1(total, bits, scale, regB, lb)                                       \
)"
R"(    total += (2.0f*(float)((bits >>  0) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+0); \
)"
R"(    total += (2.0f*(float)((bits >>  1) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+0); \
)"
R"(    total += (2.0f*(float)((bits >>  2) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+0); \
)"
R"(    total += (2.0f*(float)((bits >>  3) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+0); \
)"
R"(    total += (2.0f*(float)((bits >>  4) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+0); \
)"
R"(    total += (2.0f*(float)((bits >>  5) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+0); \
)"
R"(    total += (2.0f*(float)((bits >>  6) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+0); \
)"
R"(    total += (2.0f*(float)((bits >>  7) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+0); \
)"
R"(    total += (2.0f*(float)((bits >>  8) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+1); \
)"
R"(    total += (2.0f*(float)((bits >>  9) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+1); \
)"
R"(    total += (2.0f*(float)((bits >> 10) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+1); \
)"
R"(    total += (2.0f*(float)((bits >> 11) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+1); \
)"
R"(    total += (2.0f*(float)((bits >> 12) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+1); \
)"
R"(    total += (2.0f*(float)((bits >> 13) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+1); \
)"
R"(    total += (2.0f*(float)((bits >> 14) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+1); \
)"
R"(    total += (2.0f*(float)((bits >> 15) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+1); \
)"
R"(    total += (2.0f*(float)((bits >> 16) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+2); \
)"
R"(    total += (2.0f*(float)((bits >> 17) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+2); \
)"
R"(    total += (2.0f*(float)((bits >> 18) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+2); \
)"
R"(    total += (2.0f*(float)((bits >> 19) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+2); \
)"
R"(    total += (2.0f*(float)((bits >> 20) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+2); \
)"
R"(    total += (2.0f*(float)((bits >> 21) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+2); \
)"
R"(    total += (2.0f*(float)((bits >> 22) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+2); \
)"
R"(    total += (2.0f*(float)((bits >> 23) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+2); \
)"
R"(    total += (2.0f*(float)((bits >> 24) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s0, lb+3); \
)"
R"(    total += (2.0f*(float)((bits >> 25) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s1, lb+3); \
)"
R"(    total += (2.0f*(float)((bits >> 26) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s2, lb+3); \
)"
R"(    total += (2.0f*(float)((bits >> 27) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s3, lb+3); \
)"
R"(    total += (2.0f*(float)((bits >> 28) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s4, lb+3); \
)"
R"(    total += (2.0f*(float)((bits >> 29) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s5, lb+3); \
)"
R"(    total += (2.0f*(float)((bits >> 30) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s6, lb+3); \
)"
R"(    total += (2.0f*(float)((bits >> 31) & 1u) - 1.0f) * scale * sub_group_broadcast(regB.s7, lb+3);
)"
R"(
)"
R"(
)"
R"(#ifdef ADRENO_GPU
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(#endif
)"
R"(__kernel void kernel_gemv_noshuffle_q1_0_f32(
)"
R"(        read_only  image1d_buffer_t src0_q,
)"
R"(        global half  * src0_d,
)"
R"(        read_only  image1d_buffer_t src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        int ne01,
)"
R"(        int ne02,
)"
R"(        int ne10,
)"
R"(        int ne12,
)"
R"(        int ne0,
)"
R"(        int ne1,
)"
R"(        int r2,
)"
R"(        int r3)
)"
R"({
)"
R"(    uint groupId = get_local_id(1);
)"
R"(    uint gid     = get_global_id(0);
)"
R"(    ushort slid  = get_sub_group_local_id();
)"
R"(
)"
R"(    uint K = ne00;
)"
R"(    uint M = ne01;
)"
R"(
)"
R"(    uint LINE_STRIDE_A  = M;
)"
R"(    uint BLOCK_STRIDE_A = 4 * M;
)"
R"(
)"
R"(    uint4  regA;
)"
R"(    half   regS;
)"
R"(    float8 regB;
)"
R"(
)"
R"(    float totalSum = 0.0f;
)"
R"(
)"
R"(    #pragma unroll 1
)"
R"(    for (uint kb = groupId; kb < (K / QK1_0); kb += N_SIMDGROUP) {
)"
R"(        regS = src0_d[gid + kb * LINE_STRIDE_A]; // each fiber loads its row's scale
)"
R"(
)"
R"(        // first 16 fibers load 8 B values each -> 128 activations for this block
)"
R"(        if (slid < 16) {
)"
R"(            regB.s0123 = read_imagef(src1, (slid * 2 + kb * 32));
)"
R"(            regB.s4567 = read_imagef(src1, (1 + slid * 2 + kb * 32));
)"
R"(        }
)"
R"(
)"
R"(        // load this row's 4 uint32 (128 sign bits)
)"
R"(        regA.s0 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 0)).x;
)"
R"(        regA.s1 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 1)).x;
)"
R"(        regA.s2 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 2)).x;
)"
R"(        regA.s3 = read_imageui(src0_q, (gid + kb * BLOCK_STRIDE_A + LINE_STRIDE_A * 3)).x;
)"
R"(
)"
R"(        float scale = (float)regS;
)"
R"(        dequantizeBlockAccum_q1(totalSum, regA.s0, scale, regB, 0);
)"
R"(        dequantizeBlockAccum_q1(totalSum, regA.s1, scale, regB, 4);
)"
R"(        dequantizeBlockAccum_q1(totalSum, regA.s2, scale, regB, 8);
)"
R"(        dequantizeBlockAccum_q1(totalSum, regA.s3, scale, regB, 12);
)"
R"(    }
)"
R"(
)"
R"(    // reduction in local memory, assumes #wave = N_SIMDGROUP = 4
)"
R"(    local float reduceLM[SIMDGROUP_WIDTH * 3];
)"
R"(    if (groupId == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = totalSum;
)"
R"(    if (groupId == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = totalSum;
)"
R"(    if (groupId == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = totalSum;
)"
R"(    barrier(CLK_LOCAL_MEM_FENCE);
)"
R"(    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
)"
R"(    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
)"
R"(    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];
)"
R"(
)"
R"(    if (groupId == 0) {
)"
R"(        dst = (global float*)((global char*)dst + offsetd);
)"
R"(        // Guard the output row. The x-grid is padded to CEIL_DIV(M,wavesize)*wavesize,
)"
R"(        // so when ne01 is not a multiple of the wave size the tail work-items run past
)"
R"(        // row ne01 and would overrun dst into the adjacent tensor. No-op / byte-identical
)"
R"(        // when ne01 is wave-aligned (no padding).
)"
R"(        if (gid < M) dst[gid] = totalSum;
)"
R"(    }
)"
R"(}
)"
