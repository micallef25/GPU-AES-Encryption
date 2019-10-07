#include "common.h"


namespace aes {

	namespace block_level {
		aes::Common::PerformanceTimer& timer();
		int aes_encrypt_block(aes_info* aes);
		int aes_decrypt_block(aes_info* aes);
	}
}