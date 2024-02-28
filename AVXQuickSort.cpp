#include <iostream>
#include "QuickSort.h"

std::pair<std::vector<double>, std::vector<double>> benchmark(QuickSort& q, size_t nBenchmarks) {
	std::pair<std::vector<double>, std::vector<double>> ret;
	ret.first = std::vector<double>(nBenchmarks);
	ret.second = std::vector<double>(nBenchmarks);
	// create test data
	std::vector<std::vector<int>> testDataNaive(nBenchmarks);
	std::vector<std::vector<int>> testDataAvx(nBenchmarks);
	for (size_t i = 0; i < nBenchmarks; i++) {
		std::vector<int> seqData = q.createRandomData(std::pow(2, i));
		std::vector<int> avxData(seqData);
		testDataNaive[i] = std::move(seqData);
		testDataAvx[i] = std::move(avxData);
	}

	// benchmark naive
	std::cout << "naive benchmark" << std::endl;
	for (int i = 0; i < nBenchmarks; i++) {
		std::cout << i << " ";
		ret.first[i] = q.quickSortNaive(testDataNaive[i]);
	}
	std::cout << " naive finished!" << std::endl;

	// benchmark avx
	std::cout << "avx benchmark" << std::endl;
	for (int i = 0; i < nBenchmarks; i++) {
		std::cout << i << " ";
		ret.second[i] = q.quickSortAVX(testDataAvx[i]);
	}
	std::cout << " avx finished!" << std::endl;
	
	return ret;
}

int main() {

	QuickSort q;
	const uint64_t size = 50;// *1024 * 1024; // 50M * 4Byte = 200MByte

	auto seqData = q.createRandomData(size);			
	auto avxData(seqData);

	double seqTime = q.quickSortNaive(seqData);
	if (!q.prove(seqData)) {
		std::cout << "seq sort did not sort..." << std::endl;
		exit(-42);
	}

	std::cout << "sorted seq in " << seqTime << "ms" << std::endl;

	double parTime = q.quickSortAVX(avxData);
	if (!q.prove(avxData)) {
		std::cout << "avx sort did not sort..." << std::endl;
		exit(-42);
	}

	// compare if both sorts did the same thing
	if (memcmp(avxData.data(), seqData.data(), sizeof(int) * seqData.size())) {
		std::cout << "sorts do not have the same result..." << std::endl;
		exit(-42);
	}

	std::cout << "sorting worked!" << std::endl;

	std::cout << "sorted AVX in " << parTime << "ms" << std::endl;
	std::cout << "speedup is " << seqTime / parTime << std::endl;

	auto a = benchmark(q, 28);
	for (int i = 0; i < 28; i++) {
		std::cout << i << "\t" << a.first[i] << "\t" << a.second[i] << std::endl;
	}
}
