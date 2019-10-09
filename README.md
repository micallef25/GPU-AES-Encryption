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

- [Performance Analysis](#Performance-Analysis)
  - [Block Level](#Block-Level)
  - [Byte Level](#Byte-Level)

- [Resources](#Resources)

# Repo Contents
This repo contains code that benchmarks the well known cryptography AES algorithms.
We run ECB and CTR modes on both the CPU and GPU to see the benefits that can be obtained with highly parallel algorithms on highly parallel hardware.

The CPU benchmark is written in C, The GPU benchmark is written in CUDA.

The GPU has two different methods for encrypting and decrypting. One can encrypt and decypt at the granularity of a byte or a block. Performance and details are discussed below.

# AES Overview

<iframe src="https://giphy.com/embed/Z543HuFdQAmkg" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/hacker-Z543HuFdQAmkg">via GIPHY</a></p>

AES is a highly popular cryptography algorithm. This algorithm allows the user to encrpyt and decrypt files using a key. There are modes ECB, CBC, OFB, CFB, CTR each offering a different way of encrypting and decrypting files. 

## ECB Mode
Lorem ipsum...

![](img/ecb.PNG)

## CTR Mode
Lorem ipsum... (the added text)

![](img/ctr.PNG)

# Performance Analysis
Lorem ipsum...

## Block Level
Lorem ipsum...

![](img/blocklevel.png)

## Byte Level
Lorem ipsum... (the added text)

![](img/bytelevel.png)

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
