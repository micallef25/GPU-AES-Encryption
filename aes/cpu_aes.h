#pragma once

#include "common.h"
#include "tiny-AES-c/aes.hpp"


namespace aes {
	namespace CPU {
		aes::Common::PerformanceTimer& timer();
		void cpu_encrypt(AES_ctx* ctx, uint8_t* outbuff);
		void cpu_decrypt(AES_ctx* ctx, uint8_t* outbuff);


	}
}