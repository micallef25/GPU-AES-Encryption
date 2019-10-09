CUDA AES encryption
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Eric Micallef
  * https://www.linkedin.com/in/eric-micallef-99291714b/
  
* Tested on: Windows 10, i5, Nvidia GTX 1660 (Personal)

- [Repo Contents](#Repo-Contents)

- [AES Overview](#AES-Overview)
  - [ECB Mode](#ECB-Mode)
  - [CTR Mode](#CTR-Mode)

- [Algorithm Overview](#Algorithm-Overview)
  - [Mix Columns](#Mix-Columns)
  - [Shift Rows](#Shift-Rows)
  - [Sub Bytes](#Sub-Bytes)
  - [Add Round Key](#Add-Round-Key)

- [Performance Analysis](#Performance-Analysis)
  - [Block Level](#Block-Level)
  - [Byte Level](#Byte-Level)
  - [Compiler Exploits](#Compiler-Exploits)

- [Resources](#Resources)

# Repo Contents
This repo contains code that benchmarks the well known cryptography AES algorithms.
We run ECB and CTR modes on both the CPU and GPU to see the benefits that can be obtained with highly parallel algorithms on highly parallel hardware.

The CPU benchmark is written in C, The GPU benchmark is written in CUDA.

The GPU has two different methods for encrypting and decrypting. One can encrypt and decypt at the granularity of a byte or a block. Performance and details are discussed below.

# AES Overview

AES is a highly popular cryptography algorithm. This algorithm allows the user to encrpyt and decrypt files using a key. There are modes ECB, CBC, OFB, CFB, CTR each offering a different way of encrypting and decrypting files. 

## ECB Mode

ECB mode is one mode implemented for benchmarking in this repo. It is highly parallelizable this mode is a perfect fit for the GPU. From the diagram above we can see how the algorithm works. We

![](img/ecb.PNG)

## CTR Mode
Lorem ipsum... (the added text)

![](img/ctr.PNG)

# Algorithm Overview

AES128, AES192 and AES256 all follow the same schema where we continually transform our input data over a set of rounds and a fixed size key.

For AES 128 we have a fixed size key of 16bytes, an expanded key of 176bytes and 10 rounds of transformations that each 16 bytes of data mmust go through.

For AES 192 we have a fixed size key of 24 bytes, an expanded key of 204 bytes and 12 rounds of transformations that the 16 bytes of data must go through

For AES 256 we have a fixed size key of 32 bytes, an expanded key of 240 bytes and 14 rounds of transformations that the 16 bytes of data must go through.

Each round goes through 4 transformations. Column inverse, row shift, sub bytes and adding the round key. Each transformation is briefly discussed below.

## Mix Columns

The most confusing... each column is combined using an inertible linear transformation.

![](img/mixcolumns.PNG)

## Shift Rows 

bytes are shifted by row accordign to which row they are in.

![](img/shiftrows.PNG)

## Sub Bytes

Each byte is substituted with the value in the pre computed look up table

![](img/subbytes.PNG)

## Add Round Key

Each byte of the current state transformation is XOR'd with the roundkey

![](img/addroundkey.PNG)

# Performance Analysis
Since AES can be parrallelized in a few different ways I chose to investigate the question how much parrallelism is too much? 

From analyzing the algorithm you can split it at a block level granularity where each thread operates on one 16 byte block. 

We can also split it up where each thread works on a single byte in the 16 byte block. 

So for block level parallelism if we have a text file of length 128 bytes we will need 8 threads to finsih the work. 

For byte level parallelism with the same text file length of 128 bytes we will need 128 threads to finish the work.

operating at a byte level seems a bit of overkill but you never really know until you do it.

## Block Level

![](img/blocklevel.png)

At a block level each thread reads its respective 16 bytes and begins its transformations.

Each round we must perform 16 reads from our look up table and 16 reads from our roundkey per thread. To help reduce latency we move these into shared memory. Now upon launch one thread can read in all of our keys and look up table and every thread in the block can read from shared memory as opposed to main memory. Barring any bank conflicts a shared memory read costs around two cycles. This removes the stereotypical memory bottleneck.


## Byte Level

![](img/bytelevel.png)

At a byte level we utilize shared memory much more than block level.

When operating at a byte granularity we need to move the data we want to transform, our key, and our look up table. So the shared memory cost is a bit higher. We get the same benefit of exploiting the use of shared memory for transforming our text.

## Compiler Exploits

I spent some time using the unroll pragma on some of the my loops. As well as trying to utilize my memory bus by making my own memcopy by typecasting to 64 bits to send bigger chunks of data. This inspiration came from talking to my classmate Taylor Nelms after he had mentioned he did something similar.

I did see some speed up by doing this but then realized an optimized compiler will just do this stuff for me... so... really it was not the most efficient use of my time. 

# Resources

## Understanding AES encryption/decryption

* https://tools.ietf.org/html/rfc3686#section-2.1
* https://www.comparitech.com/blog/information-security/what-is-aes-encryption/
* https://en.wikipedia.org/wiki/Block_cipher_mode_of_operation#Counter_(CTR)
* https://developer.nvidia.com/gpugems/GPUGems3/gpugems3_ch36.html
* https://engineering.purdue.edu/kak/compsec/NewLectures/Lecture8.pdf

## Text Generation

* http://loremfuckingipsum.com/text.php

## Libraries

* https://github.com/kokke/tiny-AES-c
* https://github.com/jarro2783/cxxopts

## Pictures 

* ECB / CTR / AES algorith pictures taken from wikipedia