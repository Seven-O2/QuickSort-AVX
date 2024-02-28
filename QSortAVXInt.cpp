#include "QuickSort.h"
#include "PermLookupTable32Bit.h"
#include <assert.h>
#include <immintrin.h>

constexpr int intIn256 = (sizeof(__m256i) / sizeof(int));

// naive quicksort without tail recursion elimination
void QuickSort::qSortNaive(std::vector<int>& a, int64_t start, int64_t stop)
{
	assert((start >= 0) && "start is negative");
	if (stop >= 0) {
		assert((stop >= 0) && "stop is negative");
		if (start < stop) {
			int pivot = a[(start + stop) >> 1];
			int64_t beg = start;
			int64_t end = stop;
			while (beg <= end) {
				while (a[beg] < pivot)
					beg++;
				while (a[end] > pivot)
					end--;
				if (beg <= end) {
					int t = a[beg];
					a[beg] = a[end];
					a[end] = t;
					beg++;
					end--;
				}
			}
			qSortNaive(a, start, end);
			qSortNaive(a, beg, stop);
		}
	}
	
}

void AvxPartition(int* origData, int*& writeLeft, int*& writeRight, __m256i& pivotVector) {
	// load data into vector
	__m256i data = _mm256_loadu_si256((__m256i*)(origData));
	// create mask
	__m256i compared = _mm256_cmpgt_epi32(data, pivotVector);
	int mask = _mm256_movemask_ps(*(__m256*)&compared);
	// permute data with permutation table
	data = _mm256_permutevar8x32_epi32(data, _mm256_loadu_si256((__m256i*)PermLookupTable32Bit::permTable[mask]));
	// store into respective places
	_mm256_storeu_si256((__m256i*)writeLeft,  data);
	_mm256_storeu_si256((__m256i*)writeRight, data);
	// count the amount of values > pivot
	int nLargerPivot = _mm_popcnt_u32(mask);
	writeRight -= nLargerPivot;
	writeLeft  += 8 - nLargerPivot;

}

// assumes that pivot element is at a[right]
int* VecotrizedPartition(int* left, int* right) {
	int piv = *right;
	__m256i P = _mm256_set1_epi32(piv);

	// Prepare the tmp array
	int tmpArray[24];
	int* const tmpStart = &tmpArray[0];			// start address of the tmp array
	int* const tmpEnd = &tmpArray[24];			// last + 1 addres of the tmp array
	int* tmpLeft = tmpStart;					// where to start writing on the left side (<= pivot)
	int* tmpRight = tmpEnd - intIn256;		// where to start writing on the right side (> pivot)
	// Read most left and most right block into the tmp array, so we can implace correctly
	AvxPartition(left              , tmpLeft, tmpRight, P);
	AvxPartition(right - intIn256, tmpLeft, tmpRight, P);
	tmpRight += intIn256;	// reset the offset from the beginning

	int* writeLeft  = left;						// write values <= pivot
	int* writeRight = right - intIn256;		// write values  > pivot
	int* readLeft   = left + intIn256;		// pop N ints from <= pivot side
	int* readRight  = right - 2 * intIn256;	// pop N ints from  > pivot side

	// as long as there are at least N unread, unsorted entries
	while (readRight >= readLeft) {
		int* nextPtr;
		// decide wether to read the left or the right next block of data
		if ((readLeft - writeLeft) <= (writeRight - readRight)) {
			nextPtr = readLeft;
			readLeft += intIn256;
		} else {
			nextPtr = readRight;
			readRight -= intIn256;
		}
		AvxPartition(nextPtr, writeLeft, writeRight, P);
	}

	// for each element not yet sorted by avx, put into tmp array
	for (readLeft; readLeft < readRight + intIn256; readLeft++){
		*readLeft <= piv ? *tmpLeft++ = *readLeft : *--tmpRight = *readLeft;
	}
	
	// copy the contents of the tmp array to the actual data
	// left side
	size_t leftTmpSize = tmpLeft - tmpStart;
	memcpy(writeLeft, tmpStart, leftTmpSize * sizeof(int));
	writeLeft += leftTmpSize; // advance pointer
	// right side
	size_t rightTmpSize = tmpEnd - tmpRight;
	memcpy(writeLeft, tmpRight, rightTmpSize * sizeof(int));
	
	// lastly, swap the pivot with the first element in the data that is > pivot
	*right = *writeLeft;
	*writeLeft = piv;

	return writeLeft;
}

void insertionSort(std::vector<int>& a, int64_t start, int64_t stop) {
	for (size_t i = start + 1; i <= stop; i++) {
		int key = a[i];
		size_t j = i;
		while (j > start && a[j - 1] > key) {
			a[j] = a[j - 1];
			j = j - 1;
		}
		a[j] = key;
	}
}

void QuickSort::qSortAVX(std::vector<int>& a, int64_t beg, int64_t end)
{
	int64_t len = end - beg + 1;
	// Only use quicksort when problem set is larger than 16
	if (len > 16) {
		int64_t bound = (size_t)(VecotrizedPartition(&a[beg], &a[end]) - &a[0]);
		qSortAVX(a, beg, bound - 1); // sort left side
		qSortAVX(a, bound + 1, end); // sort right side
	} else if (len > 1) {
		insertionSort(a, beg, end);
	}
}

double QuickSort::meassuredSort(void(QuickSort::* sortFunc)(std::vector<int>&), std::vector<int>& a)
{
	auto start = std::chrono::high_resolution_clock::now();

	(this->*sortFunc)(a);

	auto stop = std::chrono::high_resolution_clock::now();

	return (double)std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
}

bool QuickSort::prove(std::vector<int>& a)
{
	for (uint64_t i = 0; i < a.size() - 1; ++i) {
		if (a[i] > a[i + 1])
			return false;
	}

	return true;
}

std::vector<int> QuickSort::createRandomData(int64_t size)
{
	assert((size >= 0) && "negative size");
	std::vector<int> data;

	// random gen
	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, INT_MAX);
	generator.seed((unsigned int)std::chrono::system_clock::now().time_since_epoch().count());

	for (int i = 0; i < size; ++i) {
		data.push_back(distribution(generator));
	}

	return data;
}
