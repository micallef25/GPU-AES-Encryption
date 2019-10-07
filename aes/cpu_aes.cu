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


		void aes::CPU::cpu_encrypt(AES_ctx* ctx, uint8_t* outbuff)
		{
			timer().startCpuTimer();

			AES_ECB_encrypt(ctx,outbuff);

			timer().endCpuTimer();

		}
		void aes::CPU::cpu_decrypt(AES_ctx* ctx, uint8_t* buffer)
		{
			timer().startCpuTimer();
			
			AES_ECB_decrypt(ctx,buffer);

			timer().endCpuTimer();
		}


	}
}