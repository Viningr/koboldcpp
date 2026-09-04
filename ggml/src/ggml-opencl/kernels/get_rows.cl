#pragma OPENCL EXTENSION cl_khr_fp16 : enable

typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;

#define QK4_0                   32
#define QK5_1                   32
#define Q5_1_BLOCK_BYTES        24

//------------------------------------------------------------------------------
// block_q4_0
//------------------------------------------------------------------------------
struct block_q4_0
{
    half d;
    uint8_t qs[QK4_0 / 2];
};

struct block_q5_1
{
    half d;
    half m;
    uint8_t qh[4];
    uint8_t qs[QK5_1 / 2];
};


//------------------------------------------------------------------------------
// dequantize_q4_0_f32, dequantize_q4_0_f16
//------------------------------------------------------------------------------
void dequantize_q4_0_f32(global struct block_q4_0 * xb, short il, float16 * reg) {
    global ushort * qs = ((global ushort *)xb + 1);
    float d1 = il ? (xb->d / 16.h) : xb->d;
    float d2 = d1 / 256.f;
    float md = -8.h * xb->d;
    ushort mask0 = il ? 0x00F0 : 0x000F;
    ushort mask1 = mask0 << 8;

    reg->s0 = d1 * (qs[0] & mask0) + md;
    reg->s1 = d2 * (qs[0] & mask1) + md;

    reg->s2 = d1 * (qs[1] & mask0) + md;
    reg->s3 = d2 * (qs[1] & mask1) + md;

    reg->s4 = d1 * (qs[2] & mask0) + md;
    reg->s5 = d2 * (qs[2] & mask1) + md;

    reg->s6 = d1 * (qs[3] & mask0) + md;
    reg->s7 = d2 * (qs[3] & mask1) + md;

    reg->s8 = d1 * (qs[4] & mask0) + md;
    reg->s9 = d2 * (qs[4] & mask1) + md;

    reg->sa = d1 * (qs[5] & mask0) + md;
    reg->sb = d2 * (qs[5] & mask1) + md;

    reg->sc = d1 * (qs[6] & mask0) + md;
    reg->sd = d2 * (qs[6] & mask1) + md;

    reg->se = d1 * (qs[7] & mask0) + md;
    reg->sf = d2 * (qs[7] & mask1) + md;
}


//------------------------------------------------------------------------------
// get_rows
//------------------------------------------------------------------------------
kernel void kernel_get_rows_f32(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int nchunks = get_num_groups(0) / ne10;
    int g       = get_group_id(0);
    int i10     = g / nchunks;
    int chunk   = g - i10 * nchunks;
    int i11     = get_group_id(1);
    int i12     = get_group_id(2);

    int r = ((global int *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    global float * dst_row = (global float *) ((global char *) dst  + i12*nb3 + i11*nb2 + i10*nb1);
    global float * src_row = (global float *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03);

    int span  = (ne00 + nchunks - 1) / nchunks;
    int start = chunk * span;
    int end   = min(start + span, ne00);

    for (int ind = start + get_local_id(0); ind < end; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_get_rows_f16(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        if (ind >= ne00) {
            return;
        }
        ((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1))[ind] =
            ((global half *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03))[ind];
    }
}

kernel void kernel_get_rows_q4_0(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    const int NL = 2;

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    int i02 = i11;
    int i03 = i12;

    for (int ind = get_local_id(0); ind < ne00/16; ind += get_local_size(0)) {
        float16 temp;
        if (ind >= ne00) {
            return;
        }
        dequantize_q4_0_f32(
            ((global struct block_q4_0 *) ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03)) + ind/NL, ind%NL, &temp);
        *(((global float16 *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) = temp;
    }
}

kernel void kernel_get_rows_q5_1(
        global void * src0,
        ulong offset0,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        ulong nb10,
        ulong nb11,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global void*)((global char*)src0 + offset0);
    src1 = (global int*)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);

    int r = ((global int *) ((global char *) src1 + i12*nb11 + i11*nb10))[i10];
    int i02 = i11;
    int i03 = i12;

    global struct block_q5_1 * row = (global struct block_q5_1 *)
        ((global char *) src0 + r*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        int block = ind / QK5_1;
        int lane = ind % QK5_1;
        int q_index = lane & 15;
        uchar packed = row[block].qs[q_index];
        uint qh = *((global uint *) row[block].qh);
        uint q = (lane < 16 ? (packed & 0x0F) : (packed >> 4)) |
            (((qh >> lane) & 1u) << 4);
        *(((global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1)) + ind) =
            (float) row[block].d * (float) q + (float) row[block].m;
    }
}

kernel void kernel_get_rows_q5_1_soa(
        global uchar * src0_qs,
        global uchar * src0_qh,
        global half * src0_d,
        global half * src0_m,
        ulong src0_block_offset,
        global int * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne10,
        ulong nb10,
        ulong nb11,
        ulong nb12,
        ulong nb1,
        ulong nb2,
        ulong nb3,
        int ne01,
        int transposed
) {
    src1 = (global int *) ((global char *) src1 + offset1);
    dst = (global float *) ((global char *) dst + offsetd);

    int i10 = get_group_id(0);
    int i11 = get_group_id(1);
    int i12 = get_group_id(2);
    int r = ((global int32_t *) ((global char *) src1 + i12*nb12 + i11*nb11 + i10*nb10))[0];

    global float * dst_row = (global float *) ((global char *) dst + i12*nb3 + i11*nb2 + i10*nb1);

    for (int ind = get_local_id(0); ind < ne00; ind += get_local_size(0)) {
        int t = ind % QK5_1;
        int block = ind / QK5_1;
        uchar packed;
        uint qh;
        half d;
        half m;

        if (transposed) {
            int scalar_index = block * ne01 + r;
            d = src0_d[scalar_index];
            m = src0_m[scalar_index];

            qh = 0;
            for (int byte = 0; byte < 4; ++byte) {
                qh |= ((uint) src0_qh[(block * 4 + byte) * ne01 + r]) << (8 * byte);
            }

            int word_col = 8 * block + 4 * (t >> 4) + ((t & 15) >> 2);
            ushort packed_word = ((global ushort *) src0_qs)[word_col * ne01 + r];
            packed = (uchar) (packed_word >> (4 * (t & 3)));
        } else {
            ulong row_byte_offset = (ulong) r * nb01 + (ulong) i11 * nb02 + (ulong) i12 * nb03;
            ulong block_index = src0_block_offset + row_byte_offset / Q5_1_BLOCK_BYTES + block;
            d = src0_d[block_index];
            m = src0_m[block_index];
            qh = ((global uint *) src0_qh)[block_index];
            packed = src0_qs[block_index * (QK5_1 / 2) + (t & 15)];
        }

        uint q = transposed
            ? packed & 0x0F
            : (packed >> (4 * (t >> 4))) & 0x0F;
        q |= ((qh >> t) & 1) << 4;
        dst_row[ind] = (float) d * (float) q + (float) m;
    }
}
