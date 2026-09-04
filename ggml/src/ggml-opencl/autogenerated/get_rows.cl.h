R"(#pragma OPENCL EXTENSION cl_khr_fp16 : enable
)"
R"(
)"
R"(typedef char int8_t;
)"
R"(typedef uchar uint8_t;
)"
R"(typedef short int16_t;
)"
R"(typedef ushort uint16_t;
)"
R"(typedef int int32_t;
)"
R"(typedef uint uint32_t;
)"
R"(
)"
R"(#define QK4_0                   32
)"
R"(#define QK5_1                   32
)"
R"(#define Q5_1_BLOCK_BYTES        24
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// block_q4_0
)"
R"(//------------------------------------------------------------------------------
)"
R"(struct block_q4_0
)"
R"({
)"
R"(    half d;
)"
R"(    uint8_t qs[QK4_0 / 2];
)"
R"(};
)"
R"(
)"
R"(struct block_q5_1
)"
R"({
)"
R"(    half d;
)"
R"(    half m;
)"
R"(    uint8_t qh[4];
)"
R"(    uint8_t qs[QK5_1 / 2];
)"
R"(};
)"
R"(
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// dequantize_q4_0_f32, dequantize_q4_0_f16
)"
R"(//------------------------------------------------------------------------------
)"
R"(void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
)"
R"(    global ushort * qs = ((global ushort *)xb + 1);
)"
R"(    float d1 = il ? (xb->d / 16.h) : xb->d;
)"
R"(    float d2 = d1 / 256.f;
)"
R"(    float md = -8.h * xb->d;
)"
R"(    ushort mask0 = il ? 0x00F0 : 0x000F;
)"
R"(    ushort mask1 = mask0 << 8;
)"
R"(
)"
R"(    reg->s0 = d1 * (qs[0] & mask0) + md;
)"
R"(    reg->s1 = d2 * (qs[0] & mask1) + md;
)"
R"(
)"
R"(    reg->s2 = d1 * (qs[1] & mask0) + md;
)"
R"(    reg->s3 = d2 * (qs[1] & mask1) + md;
)"
R"(
)"
R"(    reg->s4 = d1 * (qs[2] & mask0) + md;
)"
R"(    reg->s5 = d2 * (qs[2] & mask1) + md;
)"
R"(
)"
R"(    reg->s6 = d1 * (qs[3] & mask0) + md;
)"
R"(    reg->s7 = d2 * (qs[3] & mask1) + md;
)"
R"(
)"
R"(    reg->s8 = d1 * (qs[4] & mask0) + md;
)"
R"(    reg->s9 = d2 * (qs[4] & mask1) + md;
)"
R"(
)"
R"(    reg->sa = d1 * (qs[5] & mask0) + md;
)"
R"(    reg->sb = d2 * (qs[5] & mask1) + md;
)"
R"(
)"
R"(    reg->sc = d1 * (qs[6] & mask0) + md;
)"
R"(    reg->sd = d2 * (qs[6] & mask1) + md;
)"
R"(
)"
R"(    reg->se = d1 * (qs[7] & mask0) + md;
)"
R"(    reg->sf = d2 * (qs[7] & mask1) + md;
)"
R"(}
)"
R"(
)"
R"(
)"
R"(//------------------------------------------------------------------------------
)"
R"(// get_rows
)"
R"(//------------------------------------------------------------------------------
)"
R"(kernel void kernel_get_rows_f32(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne10,
)"
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int nchunks = get_num_groups(0) / ne10;
)"
R"(    int g       = get_group_id(0);
)"
R"(    int i10     = g / nchunks;
)"
R"(    int chunk   = g - i10 * nchunks;
)"
R"(    int i11     = get_group_id(1);
)"
R"(    int i12     = get_group_id(2);
)"
R"(
)"
R"(    int r = ((global int *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
)"
R"(
)"
R"(    int i02 = i11;
)"
R"(    int i03 = i12;
)"
R"(
)"
R"(    global float * dst_row = (global float *) ((global char *) dst  + i12*nb3 + i11*nb2 + i10*nb1);
)"
R"(    global float * src_row = (global float *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03);
)"
R"(
)"
R"(    int span  = (ne00 + nchunks - 1) / nchunks;
)"
R"(    int start = chunk * span;
)"
R"(    int end   = min(start + span, ne00);
)"
R"(
)"
R"(    for (int ind = start + get_local_id(0); ind < end; ind += get_local_size(0)) {
)"
R"(        dst_row[ind] = src_row[ind];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_get_rows_f16(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne10,
)"
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int i10 = get_group_id(0);
)"
R"(    int i11 = get_group_id(1);
)"
R"(    int i12 = get_group_id(2);
)"
R"(
)"
R"(    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
)"
R"(
)"
R"(    int i02 = i11;
)"
R"(    int i03 = i12;
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
)"
R"(        if (ind >= ne00) {
)"
R"(            return;
)"
R"(        }
)"
R"(        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
)"
R"(            ((global half *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_get_rows_q4_0(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne10,
)"
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    const int NL = 2;
)"
R"(
)"
R"(    int i10 = get_group_id(0);
)"
R"(    int i11 = get_group_id(1);
)"
R"(    int i12 = get_group_id(2);
)"
R"(
)"
R"(    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
)"
R"(
)"
R"(    int i02 = i11;
)"
R"(    int i03 = i12;
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
)"
R"(        float16 temp;
)"
R"(        if (ind >= ne00) {
)"
R"(            return;
)"
R"(        }
)"
R"(        dequantize_q4_0_f32(
)"
R"(            ((global struct block_q4_0 *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03)) + ind/NL, ind%NL, &temp);
)"
R"(        *(((global float16 *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) = temp;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_get_rows_q5_1(
)"
R"(        global void * src0,
)"
R"(        ulong offset0,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3
)"
R"() {
)"
R"(    src0 = (global void*)((global char*)src0 + offset0);
)"
R"(    src1 = (global int*)((global char*)src1 + offset1);
)"
R"(    dst = (global float*)((global char*)dst + offsetd);
)"
R"(
)"
R"(    int i10 = get_group_id(0);
)"
R"(    int i11 = get_group_id(1);
)"
R"(    int i12 = get_group_id(2);
)"
R"(
)"
R"(    int r = ((global int *) ((global char *) src1 + i12*nb11 + i11*nb10))[i10];
)"
R"(    int i02 = i11;
)"
R"(    int i03 = i12;
)"
R"(
)"
R"(    global struct block_q5_1 * row = (global struct block_q5_1 *)
)"
R"(        ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03);
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
)"
R"(        int block = ind / QK5_1;
)"
R"(        int lane = ind % QK5_1;
)"
R"(        int q_index = lane & 15;
)"
R"(        uchar packed = row[block].qs[q_index];
)"
R"(        uint qh = *((global uint *) row[block].qh);
)"
R"(        uint q = (lane < 16 ? (packed & 0x0F) : (packed >> 4)) |
)"
R"(            (((qh >> lane) & 1u) << 4);
)"
R"(        *(((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) =
)"
R"(            (float) row[block].d * (float) q + (float) row[block].m;
)"
R"(    }
)"
R"(}
)"
R"(
)"
R"(kernel void kernel_get_rows_q5_1_soa(
)"
R"(        global uchar * src0_qs,
)"
R"(        global uchar * src0_qh,
)"
R"(        global half * src0_d,
)"
R"(        global half * src0_m,
)"
R"(        ulong src0_block_offset,
)"
R"(        global int * src1,
)"
R"(        ulong offset1,
)"
R"(        global float * dst,
)"
R"(        ulong offsetd,
)"
R"(        int ne00,
)"
R"(        ulong nb01,
)"
R"(        ulong nb02,
)"
R"(        ulong nb03,
)"
R"(        int ne10,
)"
R"(        ulong nb10,
)"
R"(        ulong nb11,
)"
R"(        ulong nb12,
)"
R"(        ulong nb1,
)"
R"(        ulong nb2,
)"
R"(        ulong nb3,
)"
R"(        int ne01,
)"
R"(        int transposed
)"
R"() {
)"
R"(    src1 = (global int *) ((global char *) src1 + offset1);
)"
R"(    dst = (global float *) ((global char *) dst + offsetd);
)"
R"(
)"
R"(    int i10 = get_group_id(0);
)"
R"(    int i11 = get_group_id(1);
)"
R"(    int i12 = get_group_id(2);
)"
R"(    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];
)"
R"(
)"
R"(    global float * dst_row = (global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1);
)"
R"(
)"
R"(    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
)"
R"(        int t = ind % QK5_1;
)"
R"(        int block = ind / QK5_1;
)"
R"(        uchar packed;
)"
R"(        uint qh;
)"
R"(        half d;
)"
R"(        half m;
)"
R"(
)"
R"(        if (transposed) {
)"
R"(            int scalar_index = block * ne01 + r;
)"
R"(            d = src0_d[scalar_index];
)"
R"(            m = src0_m[scalar_index];
)"
R"(
)"
R"(            qh = 0;
)"
R"(            for (int byte = 0; byte < 4; ++byte) {
)"
R"(                qh |= ((uint) src0_qh[(block * 4 + byte) * ne01 + r]) << (8 * byte);
)"
R"(            }
)"
R"(
)"
R"(            int word_col = 8 * block + 4 * (t >> 4) + ((t & 15) >> 2);
)"
R"(            ushort packed_word = ((global ushort *) src0_qs)[word_col * ne01 + r];
)"
R"(            packed = (uchar) (packed_word >> (4 * (t & 3)));
)"
R"(        } else {
)"
R"(            ulong row_byte_offset = (ulong) r * nb01 + (ulong) i11 * nb02 + (ulong) i12 * nb03;
)"
R"(            ulong block_index = src0_block_offset + row_byte_offset / Q5_1_BLOCK_BYTES + block;
)"
R"(            d = src0_d[block_index];
)"
R"(            m = src0_m[block_index];
)"
R"(            qh = ((global uint *) src0_qh)[block_index];
)"
R"(            packed = src0_qs[block_index * (QK5_1 / 2) + (t & 15)];
)"
R"(        }
)"
R"(
)"
R"(        uint q = transposed
)"
R"(            ? packed & 0x0F
)"
R"(            : (packed >> (4 * (t >> 4))) & 0x0F;
)"
R"(        q |= ((qh >> t) & 1) << 4;
)"
R"(        dst_row[ind] = (float) d * (float) q + (float) m;
)"
R"(    }
)"
R"(}
)"
