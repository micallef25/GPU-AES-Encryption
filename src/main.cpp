/**
 * @file      main.cpp
 * @brief     Stream compaction test program
 * @authors   Kai Ninomiya
 * @date      2015
 * @copyright University of Pennsylvania
 */

#include <cstdio>
#include "testing_helpers.hpp"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include "tiny-AES-c/aes.hpp"
#include <cxxopts.hpp>
#include <aes/aes_ecb_byte.h>
#include <aes/aes_ecb_block.h>
#include <aes/common.h>
#include <aes/cpu_aes.h>
#include <aes/aes_ctr.h>

int main(int argc, char** argv) {

	bool benchmark = false;
	int aes_type = -1;

	//TODO add user input of their own key and maybe a nice feature of encryption / decryption
	cxxopts::Options options(argv[0], " - this program aes encryption on the CPU, and GPU, benchmark speeds for parallelizing at a block level and byte level");
	options.add_options()
		("b,benchmark", "Benchmark", cxxopts::value<bool>(benchmark))
		("i,input", "Input file for decryption encryption", cxxopts::value<std::string>())
		("o,output", "Output file to write to", cxxopts::value<std::string>())
		("k,keysize", "level of encryption, 128,196,256 supported", cxxopts::value<int>()->default_value("128"));

	auto result = options.parse(argc, argv);
	auto arguments = result.arguments();

	const std::string File = result["input"].as<std::string>();
	aes_type = result["keysize"].as<int>();

	struct AES_ctx ctx;
	struct AES_ctx ctx_ctr;
	struct AES_ctx ctx_ctr2; // apparently we need one for a reset? dumb library

	aes_info* aes_byte = aes::Common::create_aes_struct(File, aes_type);
	aes_info* aes_block = aes::Common::create_aes_struct(File, aes_type);
	aes_info* aes_original = aes::Common::create_aes_struct(File, aes_type);
	aes_info* aes_cpu = aes::Common::create_aes_struct(File, aes_type);
	aes_info* aes_ctr = aes::Common::create_aes_struct(File, aes_type);
	aes_info* aes_cpu_ctr = aes::Common::create_aes_struct(File, aes_type);
	AES_init_ctx(&ctx, aes_cpu->keys);
	AES_init_ctx_iv(&ctx_ctr, aes_cpu_ctr->keys,(uint8_t*)aes_cpu_ctr->ctr_iv);
	AES_init_ctx_iv(&ctx_ctr2, aes_cpu_ctr->keys, (uint8_t*)aes_cpu_ctr->ctr_iv);

	aes::block_level::aes_encrypt_block(aes_block);
	printElapsedTime(aes::block_level::timer().getGpuElapsedTimeForPreviousOperation(), "Encrypt Block level (std::chrono Measured)");
	aes::byte_level::aes_encrypt_byte(aes_byte);
	printElapsedTime(aes::byte_level::timer().getGpuElapsedTimeForPreviousOperation(), "Encrypt Byte level (std::chrono Measured)");
	aes::CPU::cpu_encrypt(&ctx,aes_cpu->data,aes_cpu->padded_length);
	printElapsedTime(aes::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "Encrypt CPU (std::chrono Measured)");
	aes::CPU::cpu_encrypt_ctr(&ctx_ctr, aes_cpu_ctr->data, aes_cpu_ctr->padded_length);
	printElapsedTime(aes::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "Encrypt CPU (std::chrono Measured)");
	aes::ctr::aes_ctr_encrypt(aes_ctr);
	printElapsedTime(aes::ctr::timer().getGpuElapsedTimeForPreviousOperation(), "Encrypt CTR Mode (std::chrono Measured)");

	aes::block_level::aes_decrypt_block(aes_block);
	printElapsedTime(aes::block_level::timer().getGpuElapsedTimeForPreviousOperation(), "Decrypt Block level (std::chrono Measured)");
	aes::byte_level::aes_decrypt_byte(aes_byte);
	printElapsedTime(aes::byte_level::timer().getGpuElapsedTimeForPreviousOperation(), "Decrypt Byte level (std::chrono Measured)");
	aes::CPU::cpu_decrypt(&ctx, aes_cpu->data, aes_cpu->padded_length);
	printElapsedTime(aes::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "Decrypt CPU (std::chrono Measured)");
	
	aes::ctr::aes_ctr_decrypt(aes_ctr);
	printElapsedTime(aes::ctr::timer().getGpuElapsedTimeForPreviousOperation(), "Decrypt CTR Mode (std::chrono Measured)");
	aes::CPU::cpu_decrypt_ctr(&ctx_ctr2, aes_cpu_ctr->data, aes_cpu_ctr->padded_length);
	printElapsedTime(aes::CPU::timer().getCpuElapsedTimeForPreviousOperation(), "Decrypt CPU CTR Mode (std::chrono Measured)");

	int padded_length = aes_byte->padded_length;
	int pass = memcmp(aes_original->data, aes_block->data,padded_length);
	if (pass == 0)
	{
		std::cout << "decrypted at a blocklevel successfully" << std::endl;
	}
	else {
		std::cout << "AES failed " << std::endl;
	}
	int bytepass = memcmp(aes_original->data, aes_byte->data,padded_length);
	if (bytepass == 0)
	{
		std::cout << "decrypted at a byte level successfully" << std::endl;
	}
	else {
		std::cout << "AES at a byte level failed " << std::endl;
	}
    bytepass = memcmp(aes_original->data, aes_cpu->data, padded_length);
	if (bytepass == 0)
	{
		std::cout << "decrypted CPU ECB successfully" << std::endl;
	}
	else {
		std::cout << "AES lib failed " << std::endl;
		//std::cout << 
	}
	bytepass = memcmp(aes_original->data, aes_ctr->data, padded_length);
	if (bytepass == 0)
	{
		std::cout << "decrypted CTR successfully" << std::endl;
	}
	else {
		std::cout << "AES lib failed " << std::endl;
		//std::cout << 
	}
	bytepass = memcmp(aes_original->data, aes_cpu_ctr->data, padded_length);
	if (bytepass == 0)
	{
		std::cout << "decrypted CPU CTR successfully" << std::endl;
	}
	else {
		std::cout << "AES lib failed " << std::endl;
		for (int i = 0; i < aes_original->padded_length; i++)
		{
			std::cout << aes_original->data[i];
		}
		std::cout << std::endl;
		for (int i = 0; i < aes_cpu_ctr->padded_length; i++)
		{
			std::cout << aes_cpu_ctr->data[i];
		}
		std::cout << std::endl;
	}

	aes::Common::destroy_aes_struct(aes_original);
	aes::Common::destroy_aes_struct(aes_byte);
	aes::Common::destroy_aes_struct(aes_block);
	aes::Common::destroy_aes_struct(aes_cpu);
	aes::Common::destroy_aes_struct(aes_ctr);
	aes::Common::destroy_aes_struct(aes_cpu_ctr);

    system("pause"); // stop Win32 console from closing on exit

}
