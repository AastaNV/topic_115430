#include <cassert>
#include <cstdio>
#include <vector>

#include <cooperative_groups.h>
#include <cub/cub.cuh>

#define WORKING 1
using data_type = float;

constexpr size_t num { 82944 };
constexpr size_t size_bytes { num * sizeof( data_type ) };
constexpr size_t loops { 20 };

constexpr data_type tol2 { 1e-2f * 1e-2f };
constexpr data_type floatone { 1.0f };
constexpr data_type r0_kernel { 2.0f };
constexpr data_type r0_cublas { 2.0f };

// If you are only to print for debugging 
// Consider storing as register 
__managed__ data_type result1 {};
__managed__ data_type result2 {};
__managed__ data_type beta {};

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void sdot( const data_type *__restrict__ d_a, 
                      const data_type *__restrict__ d_b,
                      const data_type *__restrict__ d_c, 
                      const data_type *__restrict__ d_d ) {

    auto block = cooperative_groups::this_thread_block( );
    auto grid = cooperative_groups::this_grid( );

    typedef cub::BlockLoad<data_type, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;
    typedef cub::BlockReduce<data_type, BLOCK_THREADS, cub::BLOCK_REDUCE_WARP_REDUCTIONS>              BlockReduce;

    __shared__ union TempStorage {
        typename BlockLoad::TempStorage   load;
        typename BlockReduce::TempStorage reduce;

    } temp_storage;
    
    const unsigned int block_offset { blockIdx.x * blockDim.x * ITEMS_PER_THREAD };

    data_type a[ITEMS_PER_THREAD] {};
    data_type b[ITEMS_PER_THREAD] {};

    BlockLoad( temp_storage.load ).Load( d_a + block_offset, a );
    block.sync( );

    BlockLoad( temp_storage.load ).Load( d_b + block_offset, b );
    block.sync( );

    data_type sum {};

    // Preform 1st sdot
#pragma unroll ITEMS_PER_THREAD
    for ( int i = 0; i < ITEMS_PER_THREAD; i++ ) {
        sum += a[i] * b[i];
    }

    data_type aggregate = BlockReduce( temp_storage.reduce ).Sum( sum );
    block.sync( );

    if ( block.thread_rank( ) == 0 ) {
        atomicAdd( &result1, aggregate );
    }

    // Make sure all blocks are finished before continuing
    grid.sync( );
    
    data_type c[ITEMS_PER_THREAD] {};
    data_type d[ITEMS_PER_THREAD] {};
    
    BlockLoad( temp_storage.load ).Load( d_c + block_offset, c );
    block.sync( );
    
    BlockLoad( temp_storage.load ).Load( d_d + block_offset, d );
    block.sync( );
    
    const data_type alpha { __fdividef( r0_kernel, result1 ) }; // fast approximation to save registers
    
    // Perform 1st and 2nd saxpy
#pragma unroll ITEMS_PER_THREAD
    for ( int i = 0; i < ITEMS_PER_THREAD; i++ ) {
        c[i] += alpha * a[i];
        d[i] += -alpha * b[i];
    }
        
    sum = 0;
    
    // Preform 2nd sdot
#pragma unroll ITEMS_PER_THREAD
    for ( int i = 0; i < ITEMS_PER_THREAD; i++ ) {
        sum += d[i] * d[i];
    }

    aggregate = BlockReduce( temp_storage.reduce ).Sum( sum );
    block.sync( );
    
    if ( block.thread_rank( ) == 0 ) {
        atomicAdd( &result2, aggregate );
    }
    
    // Make sure all blocks are finished before continuing
    grid.sync( );
    
    // Check tolerance: All threads will return or not
    if ( result2 < tol2 ) {
        return;
    }
    
    // Calculate beta
    beta = __fdividef(result2, r0_cublas); // fast approximation to save registers
    
    // Scalar multiply
#pragma unroll ITEMS_PER_THREAD
    for ( int i = 0; i < ITEMS_PER_THREAD; i++ ) {
        a[i] *= beta;
    }
    
    // Preform 3rd saxpy
#pragma unroll ITEMS_PER_THREAD
    for ( int i = 0; i < ITEMS_PER_THREAD; i++ ) {
        a[i] += -floatone * d[i];
    }
}

template<int BLOCK_THREADS, int ITEMS_PER_THREAD>
void Test( const data_type *a, 
           const data_type *b, 
           const data_type *c, 
           const data_type *d ) {
    
    cudaError_t error;

    cudaMemset( &result1, 0, sizeof( data_type ) );

    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    void *args[] { &a, &b, &c, &d };

    const int blocksPerGrid { (num/ITEMS_PER_THREAD + BLOCK_THREADS -1) / BLOCK_THREADS };
    
    printf("blocksPerGrid = %d\n", blocksPerGrid);

    cudaEventRecord( start );

    int numblocks;

    for ( int i = 0; i < loops; i++ ) {
        cudaMemset( &result1, 0, sizeof( data_type ) );
        cudaMemset( &result2, 0, sizeof( data_type ) );
        error  = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numblocks, sdot<BLOCK_THREADS, ITEMS_PER_THREAD>, BLOCK_THREADS, 0);

#if WORKING
        int n_blocks = min(numblocks*8, blocksPerGrid);
        std::printf("n_blocks: %d\n", n_blocks);
        error = cudaLaunchCooperativeKernel(reinterpret_cast<void *>( &sdot<BLOCK_THREADS, ITEMS_PER_THREAD> ),
                                    n_blocks,
                                    BLOCK_THREADS,
                                    args);
        std::printf("numblocks: %d\n", numblocks);
#else
        error = cudaLaunchCooperativeKernel(reinterpret_cast<void *>( &sdot<BLOCK_THREADS, ITEMS_PER_THREAD> ),
                                    blocksPerGrid,
                                    BLOCK_THREADS,
                                    args);
#endif
/*
        int n_blocks = min(numblocks*8, blocksPerGrid);
        std::printf("n_blocks: %d\n", n_blocks);
        error = cudaLaunchCooperativeKernel(reinterpret_cast<void *>( &sdot<BLOCK_THREADS, ITEMS_PER_THREAD> ),
                                    n_blocks, 
                                    BLOCK_THREADS, 
                                    args);
        std::printf("numblocks: %d\n", numblocks);
*/
        if(error != cudaSuccess){
            std::printf("errorcode: %d\n", error);
        }
    }

    cudaEventRecord( stop );
    cudaEventSynchronize( stop );

    float milliseconds {};
    cudaEventElapsedTime( &milliseconds, start, stop );

    std::printf( "B: %d | T: %d | I: %d @ %f\n", blocksPerGrid, BLOCK_THREADS, ITEMS_PER_THREAD, ( milliseconds / loops ) );

	// Check results
    printf("beta = %f| r0 = %f| result1 = %f| result2 = %f\n", beta, r0_kernel, result1, result2);
}

int main( ) {
    
    cudaSetDevice(1);
    
    data_type *a {};
    data_type *b {};
    data_type *c {};
    data_type *d {};
    
    cudaMallocManaged( &a, size_bytes );
    cudaMallocManaged( &b, size_bytes );
    cudaMallocManaged( &c, size_bytes );
    cudaMallocManaged( &d, size_bytes );

    for ( int i = 0; i < num/2; i++ ) {
        a[i] = 1;
        b[i] = 16;
        c[i] = 1;
        d[i] = 1;
    }

    for ( int i = num/2; i < num; i++ ) {
        a[i] = 2;
        b[i] = 3;
        c[i] = 1;
        d[i] = 1;
    }
    
    cudaMemPrefetchAsync( a, num, 0 );
    cudaMemPrefetchAsync( b, num, 0 );
    cudaMemPrefetchAsync( c, num, 0 );
    cudaMemPrefetchAsync( d, num, 0 );

	// Warm up kernel
    // Test<32, 2>( a, b, c, d );

    // std::printf( "---- Disregard warm-up above ----- \n" );
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, 0);
    int num_sms = deviceProp.multiProcessorCount;
    std::printf("num_sms: %d\n", num_sms);
    Test<256, 6>( a, b, c, d );
}
