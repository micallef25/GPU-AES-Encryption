#pragma once

#include "common.h"


namespace aes {

	namespace ctr {
		aes::Common::PerformanceTimer& timer();
		int aes_ctr_encrypt(aes_info* aes);
		int aes_ctr_decrypt(aes_info* aes);
	}
}