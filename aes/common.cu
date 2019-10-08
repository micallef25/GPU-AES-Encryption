#include "aes_ecb_byte.h"
#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

void checkCUDAErrorFn(const char *msg, const char *file, int line) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
	exit(EXIT_FAILURE);
}


namespace aes {
	namespace Common {

		// The round constant word array, Rcon[i], contains the values given by 
		// x to the power (i-1) being powers of x (x is denoted as {02}) in the field GF(2^8)
		static const uint8_t Rcon[11] = {
		  0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36 };

		// S table
		static const uint8_t sbox1[256] = {
			0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5,
			0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
			0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0,
			0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
			0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc,
			0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
			0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a,
			0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
			0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0,
			0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
			0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b,
			0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
			0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85,
			0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
			0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
			0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
			0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17,
			0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
			0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88,
			0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
			0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c,
			0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
			0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9,
			0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
			0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6,
			0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
			0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e,
			0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
			0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94,
			0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
			0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68,
			0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
		};

		// create our AES instance
		aes_info* aes::Common::create_aes_struct(std::string File,int type)
		{
			std::ifstream inFile;
			int key_bytes;
			int rounds;
			int expand_size;

			if (type == _AES128_)
			{
				key_bytes = 16;
				rounds = 10;
				expand_size = 176;
			}
			else if(type == _AES196_)
			{
				key_bytes = 24;
				rounds = 12;
				expand_size = 208;
			}
			else if(type == _AES256_)
			{
				key_bytes = 32;
				rounds = 14;
				expand_size = 240;
			}
			else
			{
				printf("invalid aes \n");
				exit(1); // terminate with error
			}


			inFile.open(File);
			if (!inFile) {
				std::cout << "Unable to open file" << File << std::endl;
				exit(1); // terminate with error
			}

			// get file and padding length
			inFile.seekg(0, std::ios::end);
			int length = inFile.tellg(); // get length
			int padded_length = ((length % 16) == 0) ? length : length + (16 - (length % 16));
			std::cout << length << " : " << padded_length << std::endl;
			inFile.seekg(0, std::ios::beg); // go back 

			// make buffers and such
			IV* iv = new IV;
			uint8_t* buffer = new uint8_t[padded_length];
			aes_info* aes = new aes_info;
			uint8_t* keys = new uint8_t[key_bytes];
			uint8_t* key_expand = new uint8_t[expand_size];
			memset(key_expand, 0, expand_size*sizeof(uint8_t));
			
			// create a generic key need to reinvestigate this?
			for (int i = 0; i < key_bytes; i++) keys[i] = i;
			
			iv->ctr = 0;
			iv->uiv = 0xDECAFBAD; // classic
			iv->nonce - 0xB00B; // (;

			// finally read data
			inFile.read((char*)&buffer[0], length);
			
			// be a good boy
			inFile.close();

			for (int i = length; i < padded_length; i++)
			{
				buffer[i] = 0; // pad with zeros
			}
			
			// seems sensical to just expand your key here
			KeyExpansion(key_expand, keys,type);
			aes->ctr_iv = iv;
			aes->file_length = length;
			aes->padded_length = padded_length;
			aes->expand_length = expand_size;
			aes->data = buffer;
			aes->rounds = rounds;
			aes->keys = keys;
			aes->key_expand = key_expand;
			aes->type = type;
			return aes;
		}

		// destroy it ruthlessly
		void aes::Common::destroy_aes_struct(aes_info* aes)
		{
			// delete all our internal data
			delete[] aes->data;
			delete[] aes->keys;
			delete[] aes->key_expand;
			delete aes->ctr_iv;
			// finally delete yourself... you lived a good life...
			delete aes; 
		}

		// change to pass in the instance?
		void aes::Common::KeyExpansion(uint8_t* RoundKey, const uint8_t* Key,int type)
		{
			unsigned i,j,k;
			uint8_t tempa[4]; // Used for the column/row operations
			int key_words; // how many 32 bit words in a key
			int rounds; // how many rounds for generating encrpytion decryption

			if (type == _AES128_)
			{
				key_words = 4;
				rounds = 10;
			}
			else if (type == _AES196_)
			{
				key_words = 6;
				rounds = 12;
			}
			else if (type == _AES256_)
			{
				key_words = 8;
				rounds = 14;
			}
			else
			{
				printf("invalid aes \n");
				exit(1); // terminate with error
			}


			// The first round key is the key itself.
			for (i = 0; i < key_words; ++i)
			{
				RoundKey[(i * 4) + 0] = Key[(i * 4) + 0];
				RoundKey[(i * 4) + 1] = Key[(i * 4) + 1];
				RoundKey[(i * 4) + 2] = Key[(i * 4) + 2];
				RoundKey[(i * 4) + 3] = Key[(i * 4) + 3];
			}

			// All other round keys are found from the previous round keys.
			for (i = key_words; i < 4 * (rounds + 1); ++i)
			{
				{
					k = (i - 1) * 4;
					tempa[0] = RoundKey[k + 0];
					tempa[1] = RoundKey[k + 1];
					tempa[2] = RoundKey[k + 2];
					tempa[3] = RoundKey[k + 3];

				}

				if ( i % key_words == 0)
				{
					// This function shifts the 4 bytes in a word to the left once.
					// [a0,a1,a2,a3] becomes [a1,a2,a3,a0]

					// Function RotWord()
					{
						const uint8_t u8tmp = tempa[0];
						tempa[0] = tempa[1];
						tempa[1] = tempa[2];
						tempa[2] = tempa[3];
						tempa[3] = u8tmp;
					}

					// SubWord() is a function that takes a four-byte input word and 
					// applies the S-box to each of the four bytes to produce an output word.

					// Function Subword()
					{
						tempa[0] = sbox1[tempa[0]];
						tempa[1] = sbox1[tempa[1]];
						tempa[2] = sbox1[tempa[2]];
						tempa[3] = sbox1[tempa[3]];
					}

					tempa[0] = tempa[0] ^ Rcon[i / key_words];
				}
				if (type == _AES256_)
				{
					if (i % key_words == 4)
					{
						// Function Subword()
						{
							tempa[0] = sbox1[tempa[0]];
							tempa[1] = sbox1[tempa[1]];
							tempa[2] = sbox1[tempa[2]];
							tempa[3] = sbox1[tempa[3]];
						}
					}
				}
				j = i * 4; k = (i - key_words) * 4;
				RoundKey[j + 0] = RoundKey[k + 0] ^ tempa[0];
				RoundKey[j + 1] = RoundKey[k + 1] ^ tempa[1];
				RoundKey[j + 2] = RoundKey[k + 2] ^ tempa[2];
				RoundKey[j + 3] = RoundKey[k + 3] ^ tempa[3];
			}
		}
	}
}