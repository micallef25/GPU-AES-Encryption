set(SOURCE_FILES
    "aes_ecb_block.h"
    "aes_ecb_block.cu"
    "aes_ecb_byte.h"
    "aes_ecb_byte.cu"
    "common.h"
    "common.cu"
    "cpu_aes.cu"
    "cpu_aes.h"
    )

cuda_add_library(aes
    ${SOURCE_FILES}
    OPTIONS -arch=sm_70
    )
