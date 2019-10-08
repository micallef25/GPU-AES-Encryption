#include "aes_ecb_byte.h"
#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>


namespace aes {
	namespace byte_level {
		using aes::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		// S table
		static const uint8_t sbox[256] = {
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
		static const uint8_t sboxinv[256] = {
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
		__device__ void aes_subBytes_byte(uint32_t index, uint8_t *buf,uint8_t* s_sbox)
		{
			uint8_t b;
			b = buf[index];
			buf[index] = s_sbox[b];
		}

		// subbyte operation
		__device__ void aes_subBytes_inv_byte(uint32_t index, uint8_t *buf,uint8_t* s_box)
		{
			uint8_t b;
			b = buf[index];
			buf[index] = s_box[b];
		}

		// add round key operation
		__device__ void aes_addRoundKey_byte(uint32_t map_index, uint32_t index, uint8_t *buf, uint8_t *key)
		{
			// key is only 16,24, or 32 bytes so we need where we would go
			buf[index] ^= key[map_index];
		}

		// shift row operation
		__device__ void aes_shiftRows_byte(uint32_t index, uint8_t* buf, uint8_t* map_buf_s)
		{
			uint8_t i, shift;

			i = buf[index]; // read yours
			shift = map_buf_s[index];

			// a syncthreads is not needed here because a warp is 32 threads and we are operating on 16 bytes of data
			// 
			//__syncthreads();
			// write yours to new row position
			buf[shift] = i;
		}

		// shift row operation
		__device__ void aes_shiftRows_inv_byte(uint32_t index, uint8_t* buf,uint8_t* map_buf_s)
		{
			uint8_t i, shift;
			i = buf[index]; // read yours
			shift = map_buf_s[index];
			//__syncthreads();
			// write yours to new row position
			buf[shift] = i;
		}

		// mix column operation
		__device__ void aes_mixColumns_byte(uint32_t index, uint8_t *buf) 
		{
			uint8_t i, a, b, c, d, e;
			//for (i = 0; i < 16; i += 4) {
			if (index == 0 || ((index % 4) == 0))
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
		__device__ void aes_mixColumns_inv_byte(uint32_t index, uint8_t *buf) {
			uint8_t i, a, b, c, d, e, x, y, z;
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

		// aes encrypt algorithm one thread/one block with AES_BLOCK_SIZE 
		__global__ void kern_aes_encrypt_ecb_byte(uint8_t *buf_d, uint8_t *key_d,uint8_t* row_map, uint8_t* sbox_d,uint32_t numbytes, uint8_t rounds, uint8_t key_length) {
			uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= numbytes) { return; }

			__shared__ uint8_t e_key[MAX_EKEY_LENGTH];
			__shared__ uint8_t buf_t[blockSize1d]; // thread buffer
			__shared__ uint64_t row_map_s[blockSize1d];
			__shared__ uint8_t s_sbox[256];

			uint8_t copy_bytes = 0;
			uint8_t aes_index = index % AES_BLOCK_SIZE;
			uint8_t block_index = index % blockSize1d;
			
			// someones gotta copy
			if (block_index == 0)
			{
				// since shared mem is per block we copy in what well need
				copy_bytes = (numbytes < index + 128) ? (numbytes - index) : blockSize1d;
				memcpy(buf_t, &buf_d[index], copy_bytes);
				memcpy(e_key, &key_d[0], key_length);
				memcpy(row_map_s, row_map, blockSize1d);
				memcpy(s_sbox, sbox_d, sizeof(uint8_t) * 256);
//#pragma unroll
				//for (uint8_t b = 0; b < blockSize1d; b += sizeof(uint64_t)) {
				//	uint64_t *val = (uint64_t*)row_map[b * sizeof(uint64_t)];
				//	row_map_s[b] = *val;
				//}
			}
		
			__syncthreads();

			aes_addRoundKey_byte(aes_index, block_index, buf_t, &e_key[0]);

			for (uint8_t i = 1; i < rounds; ++i)
			{
				aes_subBytes_byte(block_index, buf_t,s_sbox);
				//
				aes_shiftRows_byte(block_index, buf_t, (uint8_t*)row_map_s);
				//
				aes_mixColumns_byte(block_index, buf_t);
				//
				aes_addRoundKey_byte(aes_index, block_index, buf_t, &e_key[i * AES_BLOCK_SIZE]);
			}
			aes_subBytes_byte(block_index, buf_t,s_sbox);
			//
			aes_shiftRows_byte(block_index, buf_t, (uint8_t*)row_map_s);
			//
			aes_addRoundKey_byte(aes_index, block_index, buf_t, &e_key[key_length-AES_BLOCK_SIZE]);
			__syncthreads();

			/* someone has to write it back */
			if (block_index == 0)
			{
				memcpy(&buf_d[index], buf_t, copy_bytes);
			}

		}


		// aes decrypt algorithm
		__global__ void kern_aes_decrypt_ecb_byte(uint8_t *buf_d, uint8_t *key_d, uint8_t* row_map, uint8_t* sbox_d, uint32_t numbytes, uint8_t rounds, uint8_t key_length) {

			uint32_t index = (blockIdx.x * blockDim.x) + threadIdx.x;
			if (index >= numbytes) { return; }
			
			uint8_t copy_bytes = 0;
			uint8_t aes_index = index % AES_BLOCK_SIZE;
			uint8_t block_index = index % blockSize1d;
			// TODO this needs to be per blocksize and EKEY should maybe just be set
			// to highest
			__shared__ uint8_t e_key[MAX_EKEY_LENGTH];
			__shared__ uint8_t buf_t[blockSize1d]; // thread buffer
			__shared__ uint8_t row_map_s[blockSize1d];
			__shared__ uint8_t s_sbox[256];

			// someones gotta copy
			if (block_index == 0)
			{
				// since shared mem is per block we copy in what well need
				copy_bytes = (numbytes < index + 128) ? (numbytes - index) : blockSize1d;
				memcpy(buf_t, &buf_d[index], copy_bytes);
				memcpy(e_key, &key_d[0], key_length);
				memcpy(row_map_s, row_map, blockSize1d);
				memcpy(s_sbox, sbox_d, sizeof(uint8_t) * 256);
			}
			__syncthreads();

			aes_addRoundKey_byte(aes_index, block_index,buf_t, &e_key[key_length-AES_BLOCK_SIZE]);
			
			for (uint8_t round = (rounds - 1); round > 0; --round)
			{
				aes_shiftRows_inv_byte(block_index,buf_t, row_map_s);

				aes_subBytes_inv_byte(block_index, buf_t, s_sbox);

				aes_addRoundKey_byte(aes_index, block_index,buf_t, &e_key[round * AES_BLOCK_SIZE]);

				aes_mixColumns_inv_byte(block_index, buf_t);

			}

			aes_shiftRows_inv_byte(block_index, buf_t, row_map_s);

			aes_subBytes_inv_byte(block_index, buf_t,s_sbox);

			aes_addRoundKey_byte(aes_index, block_index,buf_t, &e_key[0]);

			__syncthreads();

			/* someone has to write it back */
			if (block_index == 0)
			{
				memcpy(&buf_d[index], buf_t, copy_bytes);
			}
		}

		// byte level parallelization. Every thread gets one byte into the 128 bit block
		int aes::byte_level::aes_encrypt_byte(aes_info* aes)
		{
			uint8_t* buf_d;
			uint8_t* key_d;
			uint8_t* row_map_d;
			uint8_t* sbox_d;
			// each thread gets one byte!
			const int active_threads = aes->padded_length;
			dim3 dimBlock = (active_threads + blockSize1d - 1) / blockSize1d;
			int map_copies = blockSize1d / AES_BLOCK_SIZE;
			static uint8_t map[16] = { 0,13,10,7,4,1,14,11,8,5,2 ,15,12,9 ,6 ,3 };

			//printf("\nBeginning byte level parralelization encryption...\n");

			// get our space
			cudaMalloc((void**)&buf_d, sizeof(uint8_t) * aes->padded_length);
			cudaMalloc((void**)&key_d, sizeof(uint8_t) * aes->expand_length);
			cudaMalloc((void**)&row_map_d, sizeof(uint8_t) * blockSize1d);
			cudaMalloc((void**)&sbox_d, sizeof(uint8_t) * 256);
			checkCUDAError("cudaMalloc");
			// copy data to device
			cudaMemcpy(key_d, aes->key_expand, sizeof(uint8_t) * aes->expand_length, cudaMemcpyHostToDevice);
			cudaMemcpy(buf_d, aes->data,sizeof(uint8_t) * aes->padded_length, cudaMemcpyHostToDevice);
			cudaMemcpy(sbox_d, sbox, sizeof(uint8_t) * 256, cudaMemcpyHostToDevice);
			checkCUDAError("cudacopy");

			// I know what you are thinking .... wtf is this eric and I would tend to agree with you
			for (int i = 0; i < map_copies; i++)
			{
				cudaMemcpy(&row_map_d[i*AES_BLOCK_SIZE], map, sizeof(uint8_t) * AES_BLOCK_SIZE, cudaMemcpyHostToDevice);
				for (int j = 0; j < AES_BLOCK_SIZE; j++) map[j] += 16;
			}

			//start timer
			timer().startGpuTimer();

			// decryption kernel
			kern_aes_encrypt_ecb_byte << <dimBlock, blockSize1d >> > (buf_d, key_d,row_map_d,sbox_d,aes->padded_length, aes->rounds, aes->expand_length);
			checkCUDAError("kernel");
			
			//end timer
			timer().endGpuTimer();

			// copy back results
			cudaMemcpy(aes->data, buf_d, sizeof(uint8_t) * aes->padded_length, cudaMemcpyDeviceToHost);
			checkCUDAError("cudacopy");
			// clean up buffers
			cudaFree(buf_d);
			cudaFree(key_d);
			cudaFree(row_map_d);
			cudaFree(sbox_d);

			return EXIT_SUCCESS;
		}


		// byte level parallelization. Every thread gets one byte into the 128 bit block
		int aes::byte_level::aes_decrypt_byte(aes_info* aes)
		{
			uint8_t* buf_d;
			uint8_t* key_d;
			uint8_t* row_map_d;
			uint8_t* sbox_d;
			// each thread gets one byte!
			const uint32_t active_threads = aes->padded_length;
			dim3 dimBlock = (active_threads + blockSize1d - 1) / blockSize1d;
			uint32_t map_copies = blockSize1d / AES_BLOCK_SIZE;
			static uint8_t map[16] = { 0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11 };

			//printf("\nBeginning byte level parralelization encryption...\n");

			// get our space
			cudaMalloc((void**)&buf_d, sizeof(uint8_t) * aes->padded_length);
			cudaMalloc((void**)&key_d, sizeof(uint8_t) * aes->expand_length);
			cudaMalloc((void**)&row_map_d, sizeof(uint8_t) * blockSize1d);
			cudaMalloc((void**)&sbox_d, sizeof(uint8_t) * 256);
			checkCUDAError("cudaMalloc");
			// copy data to device_
			cudaMemcpy(key_d, aes->key_expand, sizeof(uint8_t) * aes->expand_length, cudaMemcpyHostToDevice);
			cudaMemcpy(buf_d, aes->data, sizeof(uint8_t) * aes->padded_length, cudaMemcpyHostToDevice);
			cudaMemcpy(sbox_d, sboxinv, sizeof(uint8_t) * 256, cudaMemcpyHostToDevice);
			checkCUDAError("cudaMalloc");

			// I know what you are thinking .... wtf is this eric and I would tend to agree with you
			for (uint32_t i = 0; i < map_copies; i++)
			{
				cudaMemcpy(&row_map_d[i*AES_BLOCK_SIZE], map, sizeof(uint8_t) * AES_BLOCK_SIZE, cudaMemcpyHostToDevice);
				for (uint8_t j = 0; j < AES_BLOCK_SIZE; j++) map[j] += 16;
			}
			//start timer
			timer().startGpuTimer();

			// decryption kernel
			kern_aes_decrypt_ecb_byte << <dimBlock, blockSize1d >> > (buf_d, key_d, row_map_d,sbox_d, aes->padded_length, aes->rounds, aes->expand_length);
			checkCUDAError("cudakern");
			
			//end timer
			timer().endGpuTimer();

			// copy back results
			cudaMemcpy(aes->data, buf_d, sizeof(uint8_t) *  aes->padded_length, cudaMemcpyDeviceToHost);
			checkCUDAError("cudacopy");
			// clean up buffers
			cudaFree(buf_d);
			cudaFree(key_d);
			cudaFree(row_map_d);
			cudaFree(sbox_d);

			return EXIT_SUCCESS;
		}
	}
}