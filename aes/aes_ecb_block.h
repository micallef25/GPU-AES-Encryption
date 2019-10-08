#include "common.h"


namespace aes {

	//__device__ void aes_mixColumns(uint8_t *buf);

	namespace block_level {
		aes::Common::PerformanceTimer& timer();
		int aes_encrypt_block(aes_info* aes);
		int aes_decrypt_block(aes_info* aes);

		
		__device__ void aes_subBytes(uint8_t *buf);
		__device__ void aes_subBytes_inv(uint8_t *buf);
		
		__device__ void aes_mixColumns_inv(uint8_t *buf);
		__device__ void aes_mixColumns(uint8_t *buf);
		
		__device__ void aes_addRoundKey(uint8_t *buf, uint8_t *key);
		
		__device__ void aes_shiftRows(uint8_t *buf);
		__device__ void aes_shiftRows_inv(uint8_t *buf);
	}
}