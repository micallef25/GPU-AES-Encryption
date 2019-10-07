#pragma once

#include "common.h"


namespace aes {

	namespace byte_level {
		aes::Common::PerformanceTimer& timer();
		int aes_encrypt_byte(aes_info* ae);
		int aes_decrypt_byte(aes_info* aes);
	}
}