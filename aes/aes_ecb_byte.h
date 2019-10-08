#pragma once

#include "common.h"


namespace aes {

	namespace byte_level {
		aes::Common::PerformanceTimer& timer();
		int aes_encrypt_byte(aes_info* ae);
		int aes_decrypt_byte(aes_info* aes);

		__device__ void aes_subBytes_byte(uint32_t index, uint8_t *buf,uint8_t* s_sbox);
		__device__ void aes_subBytes_inv_byte(uint32_t index, uint8_t *buf, uint8_t* s_sbox);
		
		__device__ void aes_addRoundKey_byte(uint32_t map_index, uint32_t index, uint8_t *buf, uint8_t *key);
		
		__device__ void aes_shiftRows_byte(uint32_t index, uint8_t* buf, uint8_t* map_buf_s);
		__device__ void aes_shiftRows_inv_byte(uint32_t index, uint8_t* buf, uint8_t* map_buf_s);
		
		__device__ void aes_mixColumns_byte(uint32_t index, uint8_t *buf);
		__device__ void aes_mixColumns_inv_byte(uint32_t index, uint8_t *buf);
	}
}