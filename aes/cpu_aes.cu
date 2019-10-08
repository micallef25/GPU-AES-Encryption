#include "common.h"
#include "tiny-AES-c/aes.hpp"
#include <aes/cpu_aes.h>

namespace aes {
	namespace CPU {
		using aes::Common::PerformanceTimer;
		PerformanceTimer& timer()
		{
			static PerformanceTimer timer;
			return timer;
		}

		// call library functions for baseline time
		// TODO change types
		void aes::CPU::cpu_encrypt(AES_ctx* ctx, uint8_t* buffer, uint32_t padded_length)
		{
			timer().startCpuTimer();

			uint32_t iterations = padded_length / AES_BLOCKLEN;

			// library just does 128 bits at a time .... 
			for (uint32_t i = 0; i < iterations; i++) {
				AES_ECB_encrypt(ctx, &buffer[i*AES_BLOCKLEN]);
			}

			timer().endCpuTimer();

		}
		void aes::CPU::cpu_decrypt(AES_ctx* ctx, uint8_t* buffer, uint32_t padded_length)
		{
			timer().startCpuTimer();
			
			uint32_t iterations = padded_length / AES_BLOCKLEN;

			// library just does 128 bits at a time .... 
			for (uint32_t i = 0; i < iterations; i++) {
				AES_ECB_decrypt(ctx, &buffer[i*AES_BLOCKLEN]);
			}

			timer().endCpuTimer();
		}

		// call library functions for baseline time
		void aes::CPU::cpu_encrypt_ctr(AES_ctx* ctx, uint8_t* buffer, uint32_t padded_length)
		{
			timer().startCpuTimer();

			// ctr encrypt and decrpyt are ampedextrious.
			AES_CTR_xcrypt_buffer(ctx, buffer,padded_length);

			timer().endCpuTimer();

		}
		void aes::CPU::cpu_decrypt_ctr(AES_ctx* ctx, uint8_t* buffer, uint32_t padded_length)
		{
			timer().startCpuTimer();

			// ctr encrypt and decrpyt are ampedextrious.
			AES_CTR_xcrypt_buffer(ctx, buffer, padded_length);

			timer().endCpuTimer();
		}


	}
}