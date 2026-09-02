R"(// Mamba2 fused SSM scan kernel. One workgroup per (head, dim, seq); WG size =
)"
R"(// 64 threads. Each thread owns c_factor = d_state/64 state elements in
)"
R"(// private registers; the state stays resident across the n_tokens t-loop
)"
R"(//
)"
R"(// References:
)"
R"(//   ggml/src/ggml-cuda/ssm-scan.cu:117 ssm_scan_f32_group
)"
R"(//   ggml/src/ggml-cpu/ops.cpp:9368 ggml_compute_forward_ssm_scan_f32
)"
R"(
)"
R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(#ifdef cl_khr_subgroups
)"
R"(#pragma OPENCL EXTENSION cl_khr_subgroups : enable
)"
R"(#endif
)"
R"(
)"
R"(#if defined(cl_qcom_reqd_sub_group_size)
)"
R"(#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
)"
R"(#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
)"
R"(#else
)"
R"(#define REQD_SUBGROUP_SIZE_64
)"
R"(#endif
)"
R"(
)"
R"(inline float softplus_f32(float x) {
)"
R"(    return (x <= 20.0f) ? log(1.0f + exp(x)) : x;
)"
R"(}
)"
R"(
)"
R"(// d_state = 128 (most Mamba-2 models, e.g. mamba2-2.7B, Codestral-Mamba).
)"
R"(// WG = 64 threads, each holds 2 state elements (tid and tid+64).
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(kernel void kernel_ssm_scan_f32_mamba2_d128(
)"
R"(    global const char * src0_base, ulong src0_off,
)"
R"(    global const char * src1_base, ulong src1_off,
)"
R"(    global const char * src2_base, ulong src2_off,
)"
R"(    global const char * src3_base, ulong src3_off,
)"
R"(    global const char * src4_base, ulong src4_off,
)"
R"(    global const char * src5_base, ulong src5_off,
)"
R"(    global const char * src6_base, ulong src6_off,
)"
R"(    global       char * dst_base,  ulong dst_off,
)"
R"(    ulong s0_nb2, ulong s0_nb3,
)"
R"(    ulong x_nb2,  ulong x_nb3,
)"
R"(    ulong dt_nb1, ulong dt_nb2,
)"
R"(    ulong A_nb1,
)"
R"(    ulong B_nb2,  ulong B_nb3,
)"
R"(    ulong C_nb2,  ulong C_nb3,
)"
R"(    ulong s_off_bytes,
)"
R"(    int   head_dim, int n_head, int n_group, int n_tokens
)"
R"() {
)"
R"(    const int d_state = 128;
)"
R"(
)"
R"(    const int tid     = (int) get_local_id(0);
)"
R"(    const int wg_x    = (int) get_group_id(0);
)"
R"(    const int seq_id  = (int) get_group_id(1);
)"
R"(
)"
R"(    const int head_id = wg_x / head_dim;
)"
R"(    const int dim_id  = wg_x - head_id * head_dim;
)"
R"(    const int g       = head_id / (n_head / n_group);
)"
R"(
)"
R"(    src0_base += src0_off;
)"
R"(    src1_base += src1_off;
)"
R"(    src2_base += src2_off;
)"
R"(    src3_base += src3_off;
)"
R"(    src4_base += src4_off;
)"
R"(    src5_base += src5_off;
)"
R"(    src6_base += src6_off;
)"
R"(    dst_base  += dst_off;
)"
R"(
)"
R"(    const int seq_slot = ((global const int *) src6_base)[seq_id];
)"
R"(
)"
R"(    const ulong state_base_off = (ulong)seq_slot * s0_nb3 + (ulong)head_id * s0_nb2
)"
R"(                                + (ulong)dim_id * d_state * sizeof(float);
)"
R"(    global const float * s0_warp = (global const float *)(src0_base + state_base_off);
)"
R"(    const ulong state_out_off = (ulong)seq_id * s0_nb3 + (ulong)head_id * s0_nb2
)"
R"(                              + (ulong)dim_id * d_state * sizeof(float);
)"
R"(    global float * s_warp = (global float *)(dst_base + s_off_bytes + state_out_off);
)"
R"(
)"
R"(    global const char * x_seq  = src1_base + (ulong)seq_id * x_nb3;
)"
R"(    global const char * dt_seq = src2_base + (ulong)seq_id * dt_nb2;
)"
R"(    global const char * B_seq  = src4_base + (ulong)seq_id * B_nb3 + (ulong)g * d_state * sizeof(float);
)"
R"(    global const char * C_seq  = src5_base + (ulong)seq_id * C_nb3 + (ulong)g * d_state * sizeof(float);
)"
R"(
)"
R"(    const ulong y_dim_total = (ulong)n_head * head_dim;
)"
R"(    global float * y_seq = (global float *)dst_base
)"
R"(                           + (ulong)seq_id * (ulong)n_tokens * y_dim_total;
)"
R"(
)"
R"(    const float A_val = ((global const float *)src3_base)[(ulong)head_id * A_nb1 / sizeof(float)];
)"
R"(
)"
R"(    // c_factor = 2: each thread owns 2 state elements (tid and tid+64).
)"
R"(    float state0 = s0_warp[tid];
)"
R"(    float state1 = s0_warp[tid + 64];
)"
R"(
)"
R"(    for (int t = 0; t < n_tokens; ++t) {
)"
R"(        const float dt_h        = ((global const float *)(dt_seq + (ulong)t * dt_nb1))[head_id];
)"
R"(        const float dt_softplus = softplus_f32(dt_h);
)"
R"(        const float dA          = exp(dt_softplus * A_val);
)"
R"(        const float x_val       = ((global const float *)(x_seq + (ulong)t * x_nb2))[(ulong)head_id * head_dim + dim_id];
)"
R"(        const float x_dt        = x_val * dt_softplus;
)"
R"(
)"
R"(        const float B0 = ((global const float *)(B_seq + (ulong)t * B_nb2))[tid];
)"
R"(        const float B1 = ((global const float *)(B_seq + (ulong)t * B_nb2))[tid + 64];
)"
R"(        const float C0 = ((global const float *)(C_seq + (ulong)t * C_nb2))[tid];
)"
R"(        const float C1 = ((global const float *)(C_seq + (ulong)t * C_nb2))[tid + 64];
)"
R"(
)"
R"(        state0 = state0 * dA + B0 * x_dt;
)"
R"(        state1 = state1 * dA + B1 * x_dt;
)"
R"(        const float partial = state0 * C0 + state1 * C1;
)"
R"(
)"
R"(        const float sum = sub_group_reduce_add(partial);
)"
R"(        if (tid == 0) {
)"
R"(            y_seq[(ulong)t * y_dim_total + (ulong)head_id * head_dim + dim_id] = sum;
)"
R"(        }
)"
R"(    }
)"
R"(
)"
R"(    s_warp[tid]      = state0;
)"
R"(    s_warp[tid + 64] = state1;
)"
R"(}
)"
R"(
)"
R"(// d_state = 256 (Falcon-H1). WG = 64 threads, each holds 4 state elements.
)"
R"(REQD_SUBGROUP_SIZE_64
)"
R"(kernel void kernel_ssm_scan_f32_mamba2_d256(
)"
R"(    global const char * src0_base, ulong src0_off,
)"
R"(    global const char * src1_base, ulong src1_off,
)"
R"(    global const char * src2_base, ulong src2_off,
)"
R"(    global const char * src3_base, ulong src3_off,
)"
R"(    global const char * src4_base, ulong src4_off,
)"
R"(    global const char * src5_base, ulong src5_off,
)"
R"(    global const char * src6_base, ulong src6_off,
)"
R"(    global       char * dst_base,  ulong dst_off,
)"
R"(    ulong s0_nb2, ulong s0_nb3,
)"
R"(    ulong x_nb2,  ulong x_nb3,
)"
R"(    ulong dt_nb1, ulong dt_nb2,
)"
R"(    ulong A_nb1,
)"
R"(    ulong B_nb2,  ulong B_nb3,
)"
R"(    ulong C_nb2,  ulong C_nb3,
)"
R"(    ulong s_off_bytes,
)"
R"(    int   head_dim, int n_head, int n_group, int n_tokens
)"
R"() {
)"
R"(    const int d_state = 256;
)"
R"(
)"
R"(    const int tid     = (int) get_local_id(0);
)"
R"(    const int wg_x    = (int) get_group_id(0);
)"
R"(    const int seq_id  = (int) get_group_id(1);
)"
R"(
)"
R"(    const int head_id = wg_x / head_dim;
)"
R"(    const int dim_id  = wg_x - head_id * head_dim;
)"
R"(    const int g       = head_id / (n_head / n_group);
)"
R"(
)"
R"(    src0_base += src0_off;
)"
R"(    src1_base += src1_off;
)"
R"(    src2_base += src2_off;
)"
R"(    src3_base += src3_off;
)"
R"(    src4_base += src4_off;
)"
R"(    src5_base += src5_off;
)"
R"(    src6_base += src6_off;
)"
R"(    dst_base  += dst_off;
)"
R"(
)"
R"(    const int seq_slot = ((global const int *) src6_base)[seq_id];
)"
R"(
)"
R"(    const ulong state_base_off = (ulong)seq_slot * s0_nb3 + (ulong)head_id * s0_nb2
)"
R"(                                + (ulong)dim_id * d_state * sizeof(float);
)"
R"(    global const float * s0_warp = (global const float *)(src0_base + state_base_off);
)"
R"(    const ulong state_out_off = (ulong)seq_id * s0_nb3 + (ulong)head_id * s0_nb2
)"
R"(                              + (ulong)dim_id * d_state * sizeof(float);
)"
R"(    global float * s_warp = (global float *)(dst_base + s_off_bytes + state_out_off);
)"
R"(
)"
R"(    global const char * x_seq  = src1_base + (ulong)seq_id * x_nb3;
)"
R"(    global const char * dt_seq = src2_base + (ulong)seq_id * dt_nb2;
)"
R"(    global const char * B_seq  = src4_base + (ulong)seq_id * B_nb3 + (ulong)g * d_state * sizeof(float);
)"
R"(    global const char * C_seq  = src5_base + (ulong)seq_id * C_nb3 + (ulong)g * d_state * sizeof(float);
)"
R"(
)"
R"(    const ulong y_dim_total = (ulong)n_head * head_dim;
)"
R"(    global float * y_seq = (global float *)dst_base
)"
R"(                           + (ulong)seq_id * (ulong)n_tokens * y_dim_total;
)"
R"(
)"
R"(    const float A_val = ((global const float *)src3_base)[(ulong)head_id * A_nb1 / sizeof(float)];
)"
R"(
)"
R"(    // c_factor = 4: each thread owns 4 state elements.
)"
R"(    float state0 = s0_warp[tid];
)"
R"(    float state1 = s0_warp[tid + 64];
)"
R"(    float state2 = s0_warp[tid + 128];
)"
R"(    float state3 = s0_warp[tid + 192];
)"
R"(
)"
R"(    for (int t = 0; t < n_tokens; ++t) {
)"
R"(        const float dt_h        = ((global const float *)(dt_seq + (ulong)t * dt_nb1))[head_id];
)"
R"(        const float dt_softplus = softplus_f32(dt_h);
)"
R"(        const float dA          = exp(dt_softplus * A_val);
)"
R"(        const float x_val       = ((global const float *)(x_seq + (ulong)t * x_nb2))[(ulong)head_id * head_dim + dim_id];
)"
R"(        const float x_dt        = x_val * dt_softplus;
)"
R"(
)"
R"(        global const float * B_t = (global const float *)(B_seq + (ulong)t * B_nb2);
)"
R"(        global const float * C_t = (global const float *)(C_seq + (ulong)t * C_nb2);
)"
R"(
)"
R"(        const float B0 = B_t[tid];
)"
R"(        const float B1 = B_t[tid + 64];
)"
R"(        const float B2 = B_t[tid + 128];
)"
R"(        const float B3 = B_t[tid + 192];
)"
R"(        const float C0 = C_t[tid];
)"
R"(        const float C1 = C_t[tid + 64];
)"
R"(        const float C2 = C_t[tid + 128];
)"
R"(        const float C3 = C_t[tid + 192];
)"
R"(
)"
R"(        state0 = state0 * dA + B0 * x_dt;
)"
R"(        state1 = state1 * dA + B1 * x_dt;
)"
R"(        state2 = state2 * dA + B2 * x_dt;
)"
R"(        state3 = state3 * dA + B3 * x_dt;
)"
R"(        const float partial = state0 * C0 + state1 * C1 + state2 * C2 + state3 * C3;
)"
R"(
)"
R"(        const float sum = sub_group_reduce_add(partial);
)"
R"(        if (tid == 0) {
)"
R"(            y_seq[(ulong)t * y_dim_total + (ulong)head_id * head_dim + dim_id] = sum;
)"
R"(        }
)"
R"(    }
)"
R"(
)"
R"(    s_warp[tid]       = state0;
)"
R"(    s_warp[tid + 64]  = state1;
)"
R"(    s_warp[tid + 128] = state2;
)"
R"(    s_warp[tid + 192] = state3;
)"
R"(}
)"
