#include "aes_cbr.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}



uint8_t ctx_key[32];
uint8_t ctx_enckey[32];
uint8_t ctx_deckey[32];
uint8_t expanded_key[176]; // change to an appropriate number

#define AES_BLOCK_SIZE 16
#define THREADS_PER_BLOCK 512


#define F(x)   (((x)<<1) ^ ((((x)>>7) & 1) * 0x1b))
#define FD(x)  (((x) >> 1) ^ (((x) & 1) ? 0x8d : 0))


// The round constant word array, Rcon[i], contains the values given by 
// x to the power (i-1) being powers of x (x is denoted as {02}) in the field GF(2^8)
static const uint8_t Rcon[11] = {
  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

// S table
__constant__ static const uint8_t sbox[256] = {
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
	0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
	0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
	0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
	0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
	0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
	0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
	0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
	0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
	0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
	0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
	0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
	0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
	0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
	0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
	0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
	0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// S table
static const uint8_t sbox1[256] = {
	0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
	0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
	0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
	0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
	0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
	0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
	0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
	0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
	0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
	0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
	0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
	0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
	0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
	0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
	0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
	0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
	0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
	0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
	0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
	0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
	0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
	0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
	0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
	0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
	0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
	0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
	0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
	0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
	0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
	0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
	0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
	0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
};

// inv S table
__constant__ static const uint8_t sboxinv[256] = {
	0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38,
	0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
	0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87,
	0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
	0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d,
	0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
	0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2,
	0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
	0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16,
	0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
	0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda,
	0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
	0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a,
	0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
	0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
	0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
	0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea,
	0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
	0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85,
	0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
	0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89,
	0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
	0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20,
	0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
	0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31,
	0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
	0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d,
	0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
	0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0,
	0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
	0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26,
	0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
};


// x-time operation
__device__ uint8_t rj_xtime(uint8_t x) {
	return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}


// subbyte operation
__device__ void aes_subBytes(uint8_t *buf) {
	register uint8_t i, b;
	for (i = 0; i < 16; ++i) {
		b = buf[i];
		buf[i] = sbox[b];
	}
}

// subbyte operation
__device__ void aes_subBytes_byte(int index,uint8_t *buf) 
{
	uint8_t b;
	b = buf[index];
	buf[index] = sbox[b];
}

// subbyte operation
__device__ void aes_subBytes_inv_byte(int index, uint8_t *buf)
{
	uint8_t b;
	b = buf[index];
	buf[index] = sboxinv[b];
}


// inv subbyte operation
__device__ void aes_subBytes_inv(uint8_t *buf) {
	register uint8_t i, b;
	for (i = 0; i < 16; ++i) {
		b = buf[i];
		buf[i] = sboxinv[b];
	}
}


// add round key operation
__device__ void aes_addRoundKey(uint8_t *buf, uint8_t *key) {
	register uint8_t i = 16;
	while (i--) {
		buf[i] ^= key[i];
	}
}

// add round key operation
__device__ void aes_addRoundKey_byte(int index,uint8_t *buf, uint8_t *key) 
{
		buf[index] ^= key[index];
}


// shift row operation
__device__ void aes_shiftRows(uint8_t *buf) {
	register uint8_t i, j;
	i = buf[1];
	buf[1] = buf[5];
	buf[5] = buf[9];
	buf[9] = buf[13];
	buf[13] = i;
	i = buf[10];
	buf[10] = buf[2];
	buf[2] = i;
	j = buf[3];
	buf[3] = buf[15];
	buf[15] = buf[11];
	buf[11] = buf[7];
	buf[7] = j;
	j = buf[14];
	buf[14] = buf[6];
	buf[6] = j;
}

// shift row operation
__device__ void aes_shiftRows_byte(int index,uint8_t* buf) 
{
	uint8_t i, shift;
	static uint8_t map[16] = { 0,13,10,7,4,1,14,11,8,5,2 ,15,12,9 ,6 ,3 };
	if (index >= 16) return;
	// even row 0 will read but will just right back to same place
	// maybe less optimal but simpler code
	i = buf[index]; // read yours
	shift = map[index];
	__syncthreads();
	// write yours to new row position
	buf[shift] = i;
}

// inv shift row operation
__device__ void aes_shiftRows_inv(uint8_t *buf)
{
	register uint8_t i, j;
	i = buf[1];
	buf[1] = buf[13];
	buf[13] = buf[9];
	buf[9] = buf[5];
	buf[5] = i;
	i = buf[2];
	buf[2] = buf[10];
	buf[10] = i;
	j = buf[3];
	buf[3] = buf[7];
	buf[7] = buf[11];
	buf[11] = buf[15];
	buf[15] = j;
	j = buf[6];
	buf[6] = buf[14];
	buf[14] = j;
}


// shift row operation
__device__ void aes_shiftRows_inv_byte(int index, uint8_t* buf)
{
	uint8_t i, shift;
	// make this 32 OR just subtract 16 from tid
	static uint8_t map[16] = { 0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11 };
	if (index >= 16) return;
	// even row 0 will read but will just right back to same place
	// maybe less optimal but simpler code
	i = buf[index]; // read yours
	shift = map[index];
	__syncthreads();
	// write yours to new row position
	buf[shift] = i;
}



// mix column operation
__device__ void aes_mixColumns(uint8_t *buf) {
	register uint8_t i, a, b, c, d, e;
	for (i = 0; i < 16; i += 4) 
	{
		a = buf[i];
		b = buf[i + 1];
		c = buf[i + 2];
		d = buf[i + 3];
		e = a ^ b ^ c ^ d;
		buf[i] ^= e ^ rj_xtime(a^b);
		buf[i + 1] ^= e ^ rj_xtime(b^c);
		buf[i + 2] ^= e ^ rj_xtime(c^d);
		buf[i + 3] ^= e ^ rj_xtime(d^a);
	}
}

// mix column operation
__device__ void aes_mixColumns_byte(int index, uint8_t *buf) {
	register uint8_t i, a, b, c, d, e;
	//for (i = 0; i < 16; i += 4) {
	if( index == 0 || ((index % 4) == 0) )
	{ // only one thread per column is necessary
		a = buf[index];
		b = buf[index + 1];
		c = buf[index + 2];
		d = buf[index + 3];
		e = a ^ b ^ c ^ d;
		buf[index] ^= e ^ rj_xtime(a^b);
		buf[index + 1] ^= e ^ rj_xtime(b^c);
		buf[index + 2] ^= e ^ rj_xtime(c^d);
		buf[index + 3] ^= e ^ rj_xtime(d^a);
	}
}


// inv mix column operation
__device__ void aes_mixColumns_inv(uint8_t *buf) {
	register uint8_t i, a, b, c, d, e, x, y, z;
	for (i = 0; i < 16; i += 4) 
	{
		a = buf[i];
		b = buf[i + 1];
		c = buf[i + 2];
		d = buf[i + 3];
		e = a ^ b ^ c ^ d;
		z = rj_xtime(e);
		x = e ^ rj_xtime(rj_xtime(z^a^c));
		y = e ^ rj_xtime(rj_xtime(z^b^d));
		buf[i] ^= x ^ rj_xtime(a^b);
		buf[i + 1] ^= y ^ rj_xtime(b^c);
		buf[i + 2] ^= x ^ rj_xtime(c^d);
		buf[i + 3] ^= y ^ rj_xtime(d^a);
	}
}

// inv mix column operation
__device__ void aes_mixColumns_inv_byte(int index,uint8_t *buf) {
	register uint8_t i, a, b, c, d, e, x, y, z;
	if (index == 0 || ((index % 4) == 0))
	{
		a = buf[index];
		b = buf[index + 1];
		c = buf[index + 2];
		d = buf[index + 3];
		e = a ^ b ^ c ^ d;
		z = rj_xtime(e);
		x = e ^ rj_xtime(rj_xtime(z^a^c));
		y = e ^ rj_xtime(rj_xtime(z^b^d));
		buf[index] ^= x ^ rj_xtime(a^b);
		buf[index + 1] ^= y ^ rj_xtime(b^c);
		buf[index + 2] ^= x ^ rj_xtime(c^d);
		buf[index + 3] ^= y ^ rj_xtime(d^a);
	}
}


static void KeyExpansion(uint8_t* RoundKey, const uint8_t* Key)
{
	unsigned i, j, k;
	uint8_t tempa[4]; // Used for the column/row operations

	// The first round key is the key itself.
	for (i = 0; i < 4; ++i)
	{
		RoundKey[(i * 4) + 0] = Key[(i * 4) + 0];
		RoundKey[(i * 4) + 1] = Key[(i * 4) + 1];
		RoundKey[(i * 4) + 2] = Key[(i * 4) + 2];
		RoundKey[(i * 4) + 3] = Key[(i * 4) + 3];
	}

	// All other round keys are found from the previous round keys.
	for (i = 4; i < 4 * (10 + 1); ++i)
	{
		{
			k = (i - 1) * 4;
			tempa[0] = RoundKey[k + 0];
			tempa[1] = RoundKey[k + 1];
			tempa[2] = RoundKey[k + 2];
			tempa[3] = RoundKey[k + 3];

		}

		if (i % 4 == 0)
		{
			// This function shifts the 4 bytes in a word to the left once.
			// [a0,a1,a2,a3] becomes [a1,a2,a3,a0]

			// Function RotWord()
			{
				const uint8_t u8tmp = tempa[0];
				tempa[0] = tempa[1];
				tempa[1] = tempa[2];
				tempa[2] = tempa[3];
				tempa[3] = u8tmp;
			}

			// SubWord() is a function that takes a four-byte input word and 
			// applies the S-box to each of the four bytes to produce an output word.

			// Function Subword()
			{
				tempa[0] = sbox1[0];
				tempa[1] = sbox1[1];
				tempa[2] = sbox1[2];
				tempa[3] = sbox1[3];
			}

			tempa[0] = tempa[0] ^ Rcon[i / 4];
		}
		j = i * 4; k = (i - 4) * 4;
		RoundKey[j + 0] = RoundKey[k + 0] ^ tempa[0];
		RoundKey[j + 1] = RoundKey[k + 1] ^ tempa[1];
		RoundKey[j + 2] = RoundKey[k + 2] ^ tempa[2];
		RoundKey[j + 3] = RoundKey[k + 3] ^ tempa[3];
	}
}



// key initition
void aes256_init(uint8_t *k) {
	uint8_t rcon = 1;
	uint8_t i;
	KeyExpansion(&expanded_key[0], k);

}



// aes encrypt algorithm one thread/one block with AES_BLOCK_SIZE 
__global__ void aes256_encrypt_ecb(uint8_t *buf_d, unsigned long numbytes, uint8_t *ctx_key) {
	uint8_t i, rcon;
	uint8_t buf_t[AES_BLOCK_SIZE]; // thread buffer
	
	__shared__ uint8_t e_key[176]; // thread buffer
	
						//printf("Thread %d\n", threadIdx.x);
	unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * AES_BLOCK_SIZE) + (threadIdx.x * AES_BLOCK_SIZE);
	//int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//printf("offset %d\n", index);
	if (offset >= numbytes) { return; }

	memcpy(buf_t, &buf_d[offset], AES_BLOCK_SIZE);
	//memcpy(e_key, &ctx_key[0], 176);

	aes_addRoundKey(buf_t, &ctx_key[0]);

	for (i = 1; i < 10; ++i) 
	{
		aes_subBytes(buf_t);
		aes_shiftRows(buf_t);
		aes_mixColumns(buf_t);
		aes_addRoundKey(buf_t, &ctx_key[i*16]);
	}
	aes_subBytes(buf_t);
	aes_shiftRows(buf_t);
	aes_addRoundKey(buf_t, &ctx_key[160]);
	/* copy thread buffer back into global memory */
	//memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);
	memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);
	__syncthreads();
}

// aes encrypt algorithm one thread/one block with AES_BLOCK_SIZE 
__global__ void aes256_encrypt_ecb_byte(uint8_t *buf_d, unsigned long numbytes, uint8_t *ctx_key) {
	uint8_t i;
	//uint8_t buf_t[AES_BLOCK_SIZE]; // thread buffer

	__shared__ uint8_t e_key[176]; // thread buffer
	__shared__ uint8_t buf_t[16]; // thread buffer

						//printf("Thread %d\n", threadIdx.x);
	//unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * AES_BLOCK_SIZE) + (threadIdx.x * AES_BLOCK_SIZE);
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	//printf("offset %d\n", index);
	if (index >= numbytes) { return; }

	if(index == 0)
		memcpy(buf_t, &buf_d[0], AES_BLOCK_SIZE);
	__syncthreads();
	//memcpy(e_key, &ctx_key[0], 176);

	aes_addRoundKey_byte(index, buf_d, ctx_key);

	__syncthreads();
	for (i = 1; i < 10; ++i)
	{
		aes_subBytes_byte(index, buf_d);
		__syncthreads();
		aes_shiftRows_byte(index, buf_d);
		__syncthreads();
		aes_mixColumns_byte(index, buf_d);
		__syncthreads();
		aes_addRoundKey_byte(index, buf_d, &ctx_key[i * 16]);
		__syncthreads();
	}
	aes_subBytes_byte(index, buf_d);
	__syncthreads();
	aes_shiftRows_byte(index, buf_d);
	__syncthreads();
	aes_addRoundKey_byte(index, buf_d, &ctx_key[160]);
	__syncthreads();
	/* someone has to write it back */
	//if(index == 0)
	//	memcpy(&buf_d[0], buf_t, AES_BLOCK_SIZE);
	//memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);

}


// aes decrypt algorithm
__global__ void aes256_decrypt_ecb_byte(uint8_t *buf_d, unsigned long numbytes, uint8_t *ctx_key_d) {
	uint8_t i, rcon;
	uint8_t buf_t[AES_BLOCK_SIZE];
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index >= numbytes) { return; }

	//if (index == 0)
		//memcpy(buf_t, &buf_d[0], AES_BLOCK_SIZE);
	__syncthreads();
	
	aes_addRoundKey_byte(index,buf_d, &ctx_key_d[160]);
	__syncthreads();
	for (int round = (10 - 1); round > 0; --round)
	{
		aes_shiftRows_inv_byte(index,buf_d);
		__syncthreads();
		if (index == 0)
		{
			for (int i = 0; i < 16; i++)
				printf("%0x2", buf_d[i]);
			printf("\n");
		}
		aes_subBytes_inv_byte(index,buf_d);
		__syncthreads();
		if (index == 0)
		{
			for (int i = 0; i < 16; i++)
				printf("%0x2", buf_d[i]);
			printf("\n");
		}
		aes_addRoundKey_byte(index,buf_d, &ctx_key_d[round*16]);
		__syncthreads();
		if (index == 0)
		{
			for (int i = 0; i < 16; i++)
				printf("%0x2", buf_d[i]);
			printf("\n");
		}
		aes_mixColumns_inv_byte(index,buf_d);
		__syncthreads();
		if (index == 0)
		{
			for (int i = 0; i < 16; i++)
				printf("%0x2", buf_d[i]);
			printf("\n");
		}
	}

	aes_shiftRows_inv_byte(index,buf_d);
	__syncthreads();
	aes_subBytes_inv_byte(index,buf_d);
	__syncthreads();
	aes_addRoundKey_byte(index,buf_d, &ctx_key_d[0]);

	/* copy thread back into global memory */
	//memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);
	__syncthreads();
}


// aes decrypt algorithm
__global__ void aes256_decrypt_ecb(uint8_t *buf_d, unsigned long numbytes, uint8_t *ctx_key_d) {
	uint8_t i, rcon;
	uint8_t buf_t[AES_BLOCK_SIZE];
	unsigned long offset = (blockIdx.x * THREADS_PER_BLOCK * AES_BLOCK_SIZE) + (threadIdx.x * AES_BLOCK_SIZE);
	if (offset >= numbytes) { return; }
	memcpy(buf_t, &buf_d[offset], AES_BLOCK_SIZE);

	aes_addRoundKey(buf_t, &ctx_key_d[160]);

	for (int round = (10 - 1); round > 0; --round)
	{
		aes_shiftRows_inv(buf_t);
		if (offset == 0)
		{
			for (int i = 0; i < 16; i++)
				printf("%0x2", buf_t[i]);
			printf("\n");
		}
		aes_subBytes_inv(buf_t);
		if (offset == 0)
		{
			for (int i = 0; i < 16; i++)
				printf("%0x2", buf_t[i]);
			printf("\n");
		}
		aes_addRoundKey(buf_t, &ctx_key_d[round * 16]);
		if (offset == 0)
		{
			for (int i = 0; i < 16; i++)
				printf("%0x2", buf_t[i]);
			printf("\n");
		}
		aes_mixColumns_inv(buf_t);
		if (offset == 0)
		{
			for (int i = 0; i < 16; i++)
				printf("%0x2", buf_t[i]);
			printf("\n");
		}
	}

	aes_shiftRows_inv(buf_t);
	aes_subBytes_inv(buf_t);
	aes_addRoundKey(buf_t, &ctx_key_d[0]);

	/* copy thread back into global memory */
	memcpy(&buf_d[offset], buf_t, AES_BLOCK_SIZE);
	__syncthreads();
}

// block level parallelization. Every thread gets one 128 bit block
int aes::Common::aes_encrypt(uint8_t* in_data, uint8_t* out_data, uint8_t* key, int length)
{
	uint8_t *buf_d;
	uint8_t *ctx_key_d, *ctx_enckey_d;

	cudaMemcpyToSymbol(sbox, sbox, sizeof(uint8_t) * 256);
	cudaMemcpyToSymbol(sboxinv, sboxinv, sizeof(uint8_t) * 256);
	//cudaMemcpyToSymbol(map, map, sizeof(uint8_t) * 16);
	memset(expanded_key, 0, sizeof(uint8_t) * 176);
	printf("\nBeginning encryption\n");
	aes256_init(key);

	cudaMalloc((void**)&buf_d, sizeof(uint8_t) * length);
	cudaMalloc((void**)&ctx_key_d,sizeof(uint8_t)*176);

	//cudaMallocPitch gives 2d array
	cudaMemcpy(ctx_key_d, expanded_key, sizeof(uint8_t) * 176, cudaMemcpyHostToDevice);
	cudaMemcpy(buf_d, in_data, sizeof(uint8_t) * length, cudaMemcpyHostToDevice);

	dim3 dimBlock(ceil((double)length / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));

	dim3 dimGrid(THREADS_PER_BLOCK);
	// printf("Creating %d threads over %d blocks\n", dimBlock.x*dimGrid.x, dimBlock.x);
	aes256_encrypt_ecb << <dimBlock, dimGrid >> > (buf_d, length, ctx_key_d);
	//aes256_encrypt_ecb_byte << <dimBlock, dimGrid >> > (buf_d, length, ctx_key_d);
	cudaMemcpy(out_data, buf_d, sizeof(uint8_t) * length, cudaMemcpyDeviceToHost);
	cudaMemcpy(expanded_key, ctx_key_d, sizeof(uint8_t) * 176, cudaMemcpyDeviceToHost);
	std::cout << out_data << std::endl;
	aes256_decrypt_ecb << <dimBlock, dimGrid >> > (buf_d, length, ctx_key_d);

	cudaMemcpy(out_data, buf_d, sizeof(uint8_t) * length, cudaMemcpyDeviceToHost);
	std::cout << out_data << std::endl;

	cudaFree(buf_d);
	cudaFree(ctx_key_d);
	//cudaFree(ctx_enckey_d);


	return EXIT_SUCCESS;
}

// byte level parallelization. Every thread gets one byte into the 128 bit block
int aes::Common::aes_encrypt_byte(uint8_t* in_data, uint8_t* out_data, uint8_t* key, int length)
{
	uint8_t *buf_d;
	uint8_t *ctx_key_d, *ctx_enckey_d;

	cudaMemcpyToSymbol(sbox, sbox, sizeof(uint8_t) * 256);
	cudaMemcpyToSymbol(sboxinv, sboxinv, sizeof(uint8_t) * 256);
	//cudaMemcpyToSymbol(map, map, sizeof(uint8_t) * 16);
	memset(expanded_key, 0, sizeof(uint8_t) * 176);
	printf("\nBeginning encryption\n");
	aes256_init(key);

	cudaMalloc((void**)&buf_d, sizeof(uint8_t) * length);
	cudaMalloc((void**)&ctx_key_d, sizeof(uint8_t) * 176);

	//cudaMallocPitch gives 2d array
	cudaMemcpy(ctx_key_d, expanded_key, sizeof(uint8_t) * 176, cudaMemcpyHostToDevice);
	cudaMemcpy(buf_d, in_data, sizeof(uint8_t) * length, cudaMemcpyHostToDevice);

	dim3 dimBlock(ceil((double)length / (double)(THREADS_PER_BLOCK * AES_BLOCK_SIZE)));

	dim3 dimGrid(AES_BLOCK_SIZE);
	// printf("Creating %d threads over %d blocks\n", dimBlock.x*dimGrid.x, dimBlock.x);
	aes256_encrypt_ecb_byte << <dimBlock, dimGrid >> > (buf_d, length, ctx_key_d);

	cudaMemcpy(out_data, buf_d, sizeof(uint8_t) * length, cudaMemcpyDeviceToHost);
	cudaMemcpy(expanded_key, ctx_key_d, sizeof(uint8_t) * 176, cudaMemcpyDeviceToHost);
	std::cout << out_data << std::endl;

	aes256_decrypt_ecb_byte << <dimBlock, dimGrid >> > (buf_d, length, ctx_key_d);

	cudaMemcpy(out_data, buf_d, sizeof(uint8_t) * length, cudaMemcpyDeviceToHost);
	std::cout << out_data << std::endl;

	cudaFree(buf_d);
	cudaFree(ctx_key_d);
	//cudaFree(ctx_enckey_d);


	return EXIT_SUCCESS;
}