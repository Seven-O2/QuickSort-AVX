#pragma once
#include <vector>
#include <random>
#include <chrono>

class QuickSort
{
	void qSortNaive(std::vector<int>& a) { // facade 
		qSortNaive(a, 0, a.size() - 1);
	}
	void qSortNaive(std::vector<int>& a, int64_t beg, int64_t end); // implementation

	void qSortAVX(std::vector<int>& a) { // facade 
		qSortAVX(a, 0, a.size() - 1);
	}
	void qSortAVX(std::vector<int>& a, int64_t beg, int64_t end); // implementation

	double meassuredSort(void(QuickSort::* sortFunc)(std::vector<int>&), std::vector<int>& a);

public:

	double quickSortNaive(std::vector<int>& a)
	{
		return meassuredSort(&QuickSort::qSortNaive, a);
	}

	double quickSortAVX(std::vector<int>& a)
	{
		return meassuredSort(&QuickSort::qSortAVX, a);
	}

	bool prove(std::vector<int>& a);
	std::vector<int> createRandomData(int64_t size);
};
