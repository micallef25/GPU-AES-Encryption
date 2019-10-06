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
#include <aes/aes_cbr.h>
//using namespace std;

#define ECB 1

static int test_decrypt_ecb(void)
{
#if defined(AES256)
	uint8_t key[] = { 0x60, 0x3d, 0xeb, 0x10, 0x15, 0xca, 0x71, 0xbe, 0x2b, 0x73, 0xae, 0xf0, 0x85, 0x7d, 0x77, 0x81,
					  0x1f, 0x35, 0x2c, 0x07, 0x3b, 0x61, 0x08, 0xd7, 0x2d, 0x98, 0x10, 0xa3, 0x09, 0x14, 0xdf, 0xf4 };
	uint8_t in[] = { 0xf3, 0xee, 0xd1, 0xbd, 0xb5, 0xd2, 0xa0, 0x3c, 0x06, 0x4b, 0x5a, 0x7e, 0x3d, 0xb1, 0x81, 0xf8 };
#elif defined(AES192)
	uint8_t key[] = { 0x8e, 0x73, 0xb0, 0xf7, 0xda, 0x0e, 0x64, 0x52, 0xc8, 0x10, 0xf3, 0x2b, 0x80, 0x90, 0x79, 0xe5,
					  0x62, 0xf8, 0xea, 0xd2, 0x52, 0x2c, 0x6b, 0x7b };
	uint8_t in[] = { 0xbd, 0x33, 0x4f, 0x1d, 0x6e, 0x45, 0xf2, 0x5f, 0xf7, 0x12, 0xa2, 0x14, 0x57, 0x1f, 0xa5, 0xcc };
#elif defined(AES128)
	uint8_t key[] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c };
	uint8_t in[] = { 0x3a, 0xd7, 0x7b, 0xb4, 0x0d, 0x7a, 0x36, 0x60, 0xa8, 0x9e, 0xca, 0xf3, 0x24, 0x66, 0xef, 0x97 };
#endif

	uint8_t out[] = { 0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96, 0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a };
	struct AES_ctx ctx;

	AES_init_ctx(&ctx, key);
	AES_ECB_decrypt(&ctx, in);

	printf("ECB decrypt: ");

	if (0 == memcmp((char*)out, (char*)in, 16)) {
		printf("SUCCESS!\n");
		return(0);
	}
	else {
		printf("FAILURE!\n");
		return(1);
	}
}


int main(int argc, char** argv) {

	if (argc < 2) {
		printf("Usage: %s filename", argv[0]);
		return 1;
	}

	const char *File = argv[1];
	printf("File: %s\n", File);

	std::ifstream inFile;

	inFile.open(File);
	if (!inFile) {
		std::cout << "Unable to open file" << File << std::endl;
		exit(1); // terminate with error
	}
	// seek to end
	inFile.seekg(0, std::ios::end);
	int length = inFile.tellg();
	int padded_length = length + (length % 16);
	std::cout << length << " : " << padded_length << std::endl;
	inFile.seekg(0, std::ios::beg); // go back 
	uint8_t* buffer = new uint8_t[padded_length];
	uint8_t* outbuff = new uint8_t[padded_length];
	uint8_t* outbuff2 = new uint8_t[padded_length];
	uint8_t* outbuff3 = new uint8_t[padded_length];
	inFile.read((char*)&buffer[0], length);
	memcpy(outbuff, buffer, length);

	// might be best to do this in aes? 
	// can move later inside 
	// maybe we just pass 
	for (int i = length; i < padded_length+1; i++)
	{
		buffer[i] = 0; // pad with zeros
		outbuff[i] = 0;
	}

	struct AES_ctx ctx;
	uint8_t key[32];
	uint8_t key2[32];
	for (int i = 0; i < 32; i++) key[i] = i;
	for (int i = 0; i < 32; i++) key2[i] = i;
	// read file from input
	aes::Common::aes_encrypt(buffer,outbuff2,key,padded_length);
	aes::Common::aes_encrypt_byte(outbuff, outbuff3, key2, padded_length);

//	AES_init_ctx(&ctx, key2);
	//AES_ECB_encrypt(&ctx, (uint8_t*)outbuff);
	//AES_ECB_decrypt(&ctx, (uint8_t*)buffer);

	for (int i = 0; i < padded_length; i++)
	{
		std::cout << outbuff[i]; // pad with zeros
	}
	std::cout << std::endl;

    system("pause"); // stop Win32 console from closing on exit

}
