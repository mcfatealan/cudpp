#include <cudpp_hash.h>
#include <cuda_util.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <map>
#include <set>
#include <vector>
#include <string.h>             // memcpy
#include <stdint.h>            
#include <iostream>

#define CUDPP_APP_COMMON_IMPL
#include "stopwatch.h"


//using namespace cudpp_app;

inline int hash6432shift(uint64_t key)
{
  key = (~key) + (key << 18); // key = (key << 18) - key - 1;
  key = key ^ (key >> 31);
  key = key * 21; // key = (key + (key << 2)) + (key << 4);
  key = key ^ (key >> 11);
  key = key + (key << 6);
  key = key ^ (key >> 22);
  return (int) key;
}

int createHashTable(CUDPPHandle theCudpp,
        CUDPPHandle &hash_table_handle,
        unsigned int kInputSize)
{

    float space_usage = 1.5f;
    printf("\tSpace usage: %f\n", space_usage);

    /// -------------------- Create and build the basic hash table.
    CUDPPHashTableConfig config;
    config.type = CUDPP_MULTIVALUE_HASH_TABLE;
    config.kInputSize = kInputSize;
    config.space_usage = space_usage;
    CUDPPResult result;
    result = cudppHashTable(theCudpp, &hash_table_handle, &config);
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error in cudppHashTable call in"
                "testHashTable (make sure your device is at"
                "least compute version 2.0\n");
    }


}

int insertHashTable(CUDPPHandle hash_table_handle,
        unsigned int kInputSize,
        unsigned int * input_vals,
        unsigned int * input_keys,
        unsigned int * d_test_vals,
        unsigned int * d_test_keys,
        unsigned *&sorted_values,
        unsigned int * d_all_values)
{
   
    CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, input_keys,
                sizeof(unsigned) * kInputSize,
                cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(d_test_vals, input_vals,
                sizeof(unsigned) * kInputSize,
                cudaMemcpyHostToDevice));

    cudpp_app::StopWatch timer;
    timer.reset();
    timer.start();

    CUDPPResult result;
    result = cudppHashInsert(hash_table_handle, d_test_keys,
            d_test_vals, kInputSize);
    cudaThreadSynchronize();
    timer.stop();
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error in cudppHashInsert call in"
                "testHashTable\n");
    }
    printf("\tHash table build: %f ms\n", timer.getTime());
    /// ----------------------------------------------------------- 
    //
    unsigned int values_size;
    if (cudppMultivalueHashGetValuesSize(hash_table_handle,
                                     &values_size) !=
        CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error: "
            "cudppMultivalueHashGetValuesSize()\n");
    }
    sorted_values = new unsigned[values_size];
    if (cudppMultivalueHashGetAllValues(hash_table_handle,
                                    &d_all_values) !=
        CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error: "
            "cudppMultivalueHashGetAllValues()\n");
    }

    CUDA_SAFE_CALL(cudaMemcpy(sorted_values,
                          d_all_values,
                          sizeof(unsigned) * values_size,
                          cudaMemcpyDeviceToHost));

}


int getHashTable(CUDPPHandle hash_table_handle,
        unsigned int kInputSize,
        uint2 *d_test_vals_multivalue,
        unsigned int * d_test_keys,
        uint2 *query_vals_multivalue,
        unsigned int * query_keys)
{
    CUDA_SAFE_CALL(cudaMemcpy(d_test_keys, query_keys,
                sizeof(unsigned) * kInputSize,
                cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemset(d_test_vals_multivalue, 0,
                sizeof(uint2) * kInputSize));

    /// --------------------------------------- Query the table.
    cudpp_app::StopWatch timer;
    timer.reset();
    timer.start();
    /// hash_table.Retrieve(kInputSize, d_test_keys,
    //                      d_test_vals);

    unsigned int errors = 0;
    CUDPPResult result;
    result = cudppHashRetrieve(hash_table_handle,
        d_test_keys,
        d_test_vals_multivalue,
        kInputSize);

    timer.stop();

    printf("\tHash table retrieve in  %f ms\n", timer.getTime());

    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error in cudppHashRetrieve call in"
                "testHashTable\n");
    }

    CUDA_SAFE_CALL(cudaMemcpy(query_vals_multivalue,
        d_test_vals_multivalue,
        sizeof(uint2) * kInputSize,
        cudaMemcpyDeviceToHost));
    
    return 0;    
}


int destroyHashTable(CUDPPHandle theCudpp,
    CUDPPHandle hash_table_handle)
{
    /// -------------------------------------------- Free the table.
    CUDPPResult result;
    result = cudppDestroyHashTable(theCudpp, hash_table_handle);
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error in cudppDestroyHashTable call in"
                "testHashTable\n");
    }

    /// hash_table.Release();
    /// ------------------------------------------------------------
    return 0;
}




int main(int argc, const char **argv)
{
    unsigned kInputSize = 6;
    unsigned kOutputSize = 3;

    unsigned *input_keys = new unsigned[kInputSize];
    unsigned *input_vals = new unsigned[kInputSize];
    unsigned *query_keys = new unsigned[kOutputSize];
    uint2 *query_vals_multivalue = new uint2[kOutputSize];

    unsigned *sorted_values = NULL;

    // Allocate the GPU memory.
    unsigned *d_test_keys = NULL, *d_test_vals = NULL;
    uint2 *d_test_vals_multivalue = NULL;
    unsigned int * d_all_values = NULL;
    
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_keys,
                sizeof(unsigned) * kInputSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_vals,
                              sizeof(unsigned) * kInputSize));
    CUDA_SAFE_CALL(cudaMalloc((void**) &d_test_vals_multivalue,
                              sizeof(uint2) * kInputSize));


    CUDPPHandle theCudpp;
    CUDPPHandle hash_table_handle;
    CUDPPResult result = cudppCreate(&theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        fprintf(stderr, "Error initializing CUDPP Library.\n");
    }
    
    for (unsigned i = 0; i < kInputSize; ++i)
    {
        std::cout << "Put > ";
        std::cin >> input_keys[i];
        std::cin >> input_vals[i];
    }

    createHashTable(theCudpp,
            hash_table_handle,
            1000);

    insertHashTable(hash_table_handle,
            kInputSize,
            input_vals,
            input_keys,
            d_test_vals,
            d_test_keys,
            sorted_values,
            d_all_values);
    
    std::cout << "Get "<<kOutputSize<<"> ";
    for (unsigned i = 0; i < kOutputSize; ++i)
    {
        std::cin >> query_keys[i];
    }

    getHashTable(hash_table_handle,
            kOutputSize,
            d_test_vals_multivalue,
            d_test_keys,
            query_vals_multivalue,
            query_keys);

    std::cout << "Retrieved:\n";
    for (unsigned i = 0; i < kOutputSize; ++i)
    {
        for (unsigned j = 0; j < query_vals_multivalue[i].y; ++j) {
            std::cout << sorted_values[query_vals_multivalue[i].x + j]<<" ";
        }
    }

    destroyHashTable(theCudpp,hash_table_handle);

    CUDA_SAFE_CALL(cudaFree(d_test_keys));
    CUDA_SAFE_CALL(cudaFree(d_test_vals));
    CUDA_SAFE_CALL(cudaFree(d_test_vals_multivalue));
    delete [] input_keys;
    delete [] input_vals;
    delete [] query_keys;
    delete [] query_vals_multivalue;
    delete [] sorted_values;


    result = cudppDestroy(theCudpp);
    if (result != CUDPP_SUCCESS)
    {
        printf("Error shutting down CUDPP Library.\n");
    }

    return 0;
}

