#!/usr/bin/env python3
import sys
import os
import re
import subprocess


# FILE FOR LAUNCHING THE EXECUTABLE ALOT AND GATHERING DATA

Executable = "../build/Release/cis565_aes_encryption.exe"


def launch():



    #text_file = open("tmp.txt", "w")

    print('\nLAUNCHING 128 BIT ENCRYPTION TESTS\n')
    args = [Executable, "--keysize","128", "--input", "../encryption_files/random1k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","128", "--input", "../encryption_files/random10k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","128", "--input", "../encryption_files/random50k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","128", "--input", "../encryption_files/random100k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","128", "--input", "../encryption_files/random1000k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","128", "--input", "../encryption_files/random10000k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    print("\nLAUNCHING 192 BIT ENCRYPTION TESTS\n")
    args = [Executable, "--keysize","192", "--input", "../encryption_files/random1k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","192", "--input", "../encryption_files/random10k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","192", "--input", "../encryption_files/random50k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","192", "--input", "../encryption_files/random100k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","192", "--input", "../encryption_files/random1000k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","192", "--input", "../encryption_files/random10000k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    print("\nLAUNCHING 256 BIT ENCRYPTION TESTS\n")
    args = [Executable, "--keysize","256", "--input", "../encryption_files/random1k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","256", "--input", "../encryption_files/random10k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","256", "--input", "../encryption_files/random50k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","256", "--input", "../encryption_files/random100k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","256", "--input", "../encryption_files/random1000k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    args = [Executable, "--keysize","256", "--input", "../encryption_files/random10000k.txt"]
    results = subprocess.run(args, capture_output=True)
    resultString = results.stdout
    print(str(resultString))
    #text_file.write(str(resultString))
    #text_file.close()



def main():
    launch()



if __name__ == "__main__":
    main()