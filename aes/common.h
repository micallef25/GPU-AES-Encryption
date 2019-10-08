#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <stdint.h>


#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

#define AES_BLOCK_SIZE 16
#define blockSize1d 128
#define MAX_EKEY_LENGTH 240

#define F(x)   (((x)<<1) ^ ((((x)>>7) & 1) * 0x1b))
#define FD(x)  (((x) >> 1) ^ (((x) & 1) ? 0x8d : 0))

// make enum later
#define _AES128_ 128
#define _AES192_ 192
#define _AES256_ 256

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);
//void memory_debug_float(int elements, float* cuda_mem, float* cpu_mem);


inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return x == 1 ? 0 : ilog2(x - 1) + 1;
}

//https://tools.ietf.org/html/rfc3686#section-2.1
//page 6 tells the in depth desription
typedef struct IV_struct
{
	uint32_t nonce; // a unique 32 bit field
	uint64_t uiv; // a unique IV field
	uint32_t ctr; // counter
}IV;


typedef struct aes_info
{
	uint8_t* data; // data to decrypt/encrypt
	uint8_t* keys;
	uint8_t* key_expand;
	int rounds; // how many rounds to generate encryption/decryption
	int words; // hoe many words in a key
	int type; // 128,196 or 256?
	int file_length; // excludes padding
	int padded_length;
	int expand_length;
	IV* ctr_iv;
}aes_info;


namespace aes {
	namespace Common {

		aes_info* create_aes_struct(std::string File, int type);
		void destroy_aes_struct(aes_info* aes);
		void KeyExpansion(uint8_t* RoundKey, const uint8_t* Key, int type);
		/**
		* This class is used for timing the performance
		* Uncopyable and unmovable
		*
		* Adapted from WindyDarian(https://github.com/WindyDarian)
		*/
		class PerformanceTimer
		{
		public:

			PerformanceTimer()
			{
				cudaEventCreate(&event_start);
				cudaEventCreate(&event_end);
			}

			~PerformanceTimer()
			{
				cudaEventDestroy(event_start);
				cudaEventDestroy(event_end);
			}

			void startCpuTimer()
			{
				if (cpu_timer_started) { throw std::runtime_error("CPU timer already started"); }
				cpu_timer_started = true;

				time_start_cpu = std::chrono::high_resolution_clock::now();
			}

			void endCpuTimer()
			{
				time_end_cpu = std::chrono::high_resolution_clock::now();

				if (!cpu_timer_started) { throw std::runtime_error("CPU timer not started"); }

				std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
				prev_elapsed_time_cpu_milliseconds =
					static_cast<decltype(prev_elapsed_time_cpu_milliseconds)>(duro.count());

				cpu_timer_started = false;
			}

			void startGpuTimer()
			{
				if (gpu_timer_started) { throw std::runtime_error("GPU timer already started"); }
				gpu_timer_started = true;

				cudaEventRecord(event_start);
			}

			void endGpuTimer()
			{
				cudaEventRecord(event_end);
				cudaEventSynchronize(event_end);

				if (!gpu_timer_started) { throw std::runtime_error("GPU timer not started"); }

				cudaEventElapsedTime(&prev_elapsed_time_gpu_milliseconds, event_start, event_end);
				gpu_timer_started = false;
			}

			float getCpuElapsedTimeForPreviousOperation() //noexcept //(damn I need VS 2015
			{
				return prev_elapsed_time_cpu_milliseconds;
			}

			float getGpuElapsedTimeForPreviousOperation() //noexcept
			{
				return prev_elapsed_time_gpu_milliseconds;
			}

			// remove copy and move functions
			PerformanceTimer(const PerformanceTimer&) = delete;
			PerformanceTimer(PerformanceTimer&&) = delete;
			PerformanceTimer& operator=(const PerformanceTimer&) = delete;
			PerformanceTimer& operator=(PerformanceTimer&&) = delete;

		private:
			cudaEvent_t event_start = nullptr;
			cudaEvent_t event_end = nullptr;

			using time_point_t = std::chrono::high_resolution_clock::time_point;
			time_point_t time_start_cpu;
			time_point_t time_end_cpu;

			bool cpu_timer_started = false;
			bool gpu_timer_started = false;

			float prev_elapsed_time_cpu_milliseconds = 0.f;
			float prev_elapsed_time_gpu_milliseconds = 0.f;
		};
	}
}