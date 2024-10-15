# WebGPU Matrix Multiplication Benchmark

Try it on: https://sharmarithik.github.io/mm_bench/

## Matrix multiplication methods

Naive Implementation:
This approach performs matrix multiplication by assigning each thread to compute a single element in the output matrix C. Each thread iterates through a row of matrix A and a column of matrix B, accumulating the result. While this method is straightforward, it can become inefficient for large matrices due to frequent global memory accesses, which slow down performance by constantly reading and writing data without caching or optimization techniques.

```wgsl
function getNaiveWGSL(workgroupSize) {
    const [x, y] = workgroupSize.split(',').map(Number);
    return `
        @group(0) @binding(0) var<storage, read> matrixA : array<f32>;
        @group(0) @binding(1) var<storage, read> matrixB : array<f32>;
        @group(0) @binding(2) var<storage, read_write> matrixC : array<f32>;
        @group(0) @binding(3) var<uniform> uniforms : Uniforms;

        struct Uniforms {
            N : u32,
            alpha : f32,
            beta : f32,
        }

        @compute @workgroup_size(${x}, ${y})
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let row = global_id.y;
            let col = global_id.x;
            if (row < uniforms.N && col < uniforms.N) {
                var sum = 0.0;
                for (var i = 0u; i < uniforms.N; i = i + 1u) {
                    sum = sum + matrixA[row * uniforms.N + i] * matrixB[i * uniforms.N + col];
                }
	matrixC[row * uniforms.N + col] = uniforms.alpha * sum + uniforms.beta * matrixC[row * uniforms.N + col];
            }
        }
    `;
}
```
GMEM Coalescing (tiled implementation with global memory coalescing):
This approach optimizes matrix multiplication by dividing the matrices into smaller tiles of size TILE_SIZE x TILE_SIZE (set by the user). Threads process tiles of matrix A and matrix B, calculating partial sums across the tiles. The use of coalesced memory access improves efficiency by reading consecutive elements from global memory. This reduces the number of expensive global memory accesses, making the method more suitable for large matrices compared to the naive approach. After processing all tiles, the final result is written directly to matrix C.

```wgsl
function getGMEMCoalescingWGSL(workgroupSize, tileSize) {
    const [x, y] = workgroupSize.split(',').map(Number);
    return `
        @group(0) @binding(0) var<storage, read> matrixA : array<f32>;
        @group(0) @binding(1) var<storage, read> matrixB : array<f32>;
        @group(0) @binding(2) var<storage, read_write> matrixC : array<f32>;
        @group(0) @binding(3) var<uniform> uniforms : Uniforms;
        
        struct Uniforms {
            N : u32,
            alpha : f32,
            beta : f32,
        }
        const TILE_SIZE : u32 = ${tileSize}u;
        @compute @workgroup_size(${x}, ${y})
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let row = global_id.y;
            let col = global_id.x;
            let N = uniforms.N;
            // Out of bounds check
            if (row >= N || col >= N) {
                return;
            }
            var sum = 0.0;
            // Loop over tiles of size TILE_SIZE
            for (var t = 0u; t < N; t += TILE_SIZE) {
                // Process TILE_SIZE elements from matrixA and matrixB
                for (var i = 0u; i < TILE_SIZE; i++) {
                    let k = t + i;
                    if (k < N) {
                        // Coalesced read from matrixA (row-major)
                        let aElement = matrixA[row * N + k];
                        // Coalesced read from matrixB (column-major)
                        let bElement = matrixB[k * N + col];
                        sum += aElement * bElement;
                    }
                }
            }
            // Write the result back to matrixC with scaling
            let index = row * N + col;
            matrixC[index] = uniforms.alpha * sum + uniforms.beta * matrixC[index];
        }
    `;
}
```
SMEM Caching (tiled implementation with shared memory caching):
This approach uses shared memory (SMEM) to cache tiles of matrix A and B, reducing global memory accesses. Threads in a workgroup collaboratively load small tiles (of size TILE_SIZE x TILE_SIZE, set by the user) into shared memory, which is faster than global memory. Each thread then computes a partial result using the tiles, with synchronization between threads via workgroupBarrier() to ensure proper data loading. This method significantly boosts performance by minimizing slow global memory access, leveraging fast shared memory for frequently reused data. It is well-suited for large matrix multiplications.

```wgsl
function getSMEMCacheBlockingWGSL(workgroupSize, tileSize) {
    const [x, y] = workgroupSize.split(',').map(Number);
    return `
        @group(0) @binding(0) var<storage, read> matrixA : array<f32>;
        @group(0) @binding(1) var<storage, read> matrixB : array<f32>;
        @group(0) @binding(2) var<storage, read_write> matrixC : array<f32>;
        @group(0) @binding(3) var<uniform> uniforms : Uniforms;

        struct Uniforms {
            N : u32,
            alpha : f32,
            beta : f32,
        }

        const TILE_SIZE : u32 = ${tileSize}u;

        var<workgroup> tileA : array<array<f32, ${tileSize}>, ${tileSize}>;
        var<workgroup> tileB : array<array<f32, ${tileSize}>, ${tileSize}>;

        @compute @workgroup_size(${x}, ${y})
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) {
            let row = global_id.y;
            let col = global_id.x;
            let tileRow = local_id.y;
            let tileCol = local_id.x;

            var sum = 0.0;

            for (var t = 0u; t < uniforms.N; t += TILE_SIZE) {
                // Collaborative loading of tiles into shared memory
                if (t + tileCol < uniforms.N && row < uniforms.N) {
                    tileA[tileRow][tileCol] = matrixA[row * uniforms.N + (t + tileCol)];
                }
                if (t + tileRow < uniforms.N && col < uniforms.N) {
                    tileB[tileRow][tileCol] = matrixB[(t + tileRow) * uniforms.N + col];
                }

                workgroupBarrier();

                // Compute using tiles
                for (var k = 0u; k < TILE_SIZE; k++) {
                    if (t + k < uniforms.N) {
                        sum += tileA[tileRow][k] * tileB[k][tileCol];
                    }
                }

                workgroupBarrier();
            }

            if (row < uniforms.N && col < uniforms.N) {
                let index = row * uniforms.N + col;
                matrixC[index] = uniforms.alpha * sum + uniforms.beta * matrixC[index];
            }
        }
    `;
}
```
Vectorized Implementation:
This approach optimizes matrix multiplication by processing four elements at a time using vectorized operations. Instead of handling single elements from matrix A and B, it loads four consecutive elements from each matrix into vec4 variables. These vectors are then processed using a dot product, reducing the number of iterations and improving computational efficiency. This vectorized approach takes advantage of hardware SIMD (Single Instruction, Multiple Data) capabilities, speeding up large matrix computations compared to scalar processing. It is more efficient for handling large data sets but requires matrices to be a multiple of 4 for optimal performance.

```wgsl
function getVectorizedWGSL(workgroupSize) {
    const [x, y] = workgroupSize.split(',').map(Number);
    return `
        @group(0) @binding(0) var<storage, read> matrixA : array<f32>;
        @group(0) @binding(1) var<storage, read> matrixB : array<f32>;
        @group(0) @binding(2) var<storage, read_write> matrixC : array<f32>;
        @group(0) @binding(3) var<uniform> uniforms : Uniforms;
        struct Uniforms {
            N : u32,
            alpha : f32,
            beta : f32,
        }

        @compute @workgroup_size(${x}, ${y})
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let row = global_id.y;
            let col = global_id.x;
            if (row < uniforms.N && col < uniforms.N) {
                var sum = 0.0;
                for (var i = 0u; i < uniforms.N; i += 4u) {
                    let aVec = vec4<f32>(
                        matrixA[row * uniforms.N + i],
                        matrixA[row * uniforms.N + i + 1],
                        matrixA[row * uniforms.N + i + 2],
                        matrixA[row * uniforms.N + i + 3]
                    );
                    let bVec = vec4<f32>(
                        matrixB[i * uniforms.N + col],
                        matrixB[(i + 1) * uniforms.N + col],
                        matrixB[(i + 2) * uniforms.N + col],
                        matrixB[(i + 3) * uniforms.N + col]
                    );
                    sum += dot(aVec, bVec);
                }
                matrixC[row * uniforms.N + col] = uniforms.alpha * sum + uniforms.beta * matrixC[row * uniforms.N + col];
            }
        }
    `;
}
```
Global Tiling (8x8) Implementation:
This method uses an 8x8 tiling strategy for matrix multiplication without relying on shared memory. Each thread computes one element of the result by iterating over tiles of matrix A and B. It processes tiles of size 8x8, accumulating partial sums across multiple tiles. The approach efficiently checks matrix bounds only once per thread, minimizing overhead. While not using shared memory, it still reduces the number of global memory accesses by handling multiple elements per thread in each tile.

```wgsl
function getGlobalTiledWGSL8x8(workgroupSize) {
    const [x, y] = workgroupSize.split(',').map(Number);
    return `
        @group(0) @binding(0) var<storage, read> matrixA : array<f32>;
        @group(0) @binding(1) var<storage, read> matrixB : array<f32>;
        @group(0) @binding(2) var<storage, read_write> matrixC : array<f32>;
        @group(0) @binding(3) var<uniform> uniforms : Uniforms;
        
        struct Uniforms {
            N : u32,
            alpha : f32,
            beta : f32,
        }

        @compute @workgroup_size(${x}, ${y})
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) {
            let row = global_id.y;
            let col = global_id.x;
            let N = uniforms.N;

            var acc = 0.0;
            let numTiles = (N + 8u - 1u) / 8u;

            // Check row and column bounds only once
            let validRow = row < N;
            let validCol = col < N;

            // Perform tiling without using shared memory
            if (validRow && validCol) {
                for (var t = 0u; t < numTiles; t = t + 1u) {
                    for (var k = 0u; k < 8u; k = k + 1u) {
                        let aIndex = row * N + (t * 8u + k);
                        let bIndex = (t * 8u + k) * N + col;

                        if (aIndex < N * N && bIndex < N * N) {
                            acc = acc + matrixA[aIndex] * matrixB[bIndex];
                        }
                    }
                }

                // Write the result to matrixC
                matrixC[row * N + col] = uniforms.alpha * acc + uniforms.beta * matrixC[row * N + col];
            }
        }
    `;
}
``` 
Shared Memory Tiling (8x8) Implementation:
This method enhances matrix multiplication by using shared memory to cache tiles of size 8x8 for both matrix A and matrix B. Threads in a workgroup collaboratively load these tiles into shared memory, reducing global memory access. The threads then compute partial sums by multiplying the cached tiles and accumulating the result. Synchronization (workgroupBarrier()) ensures all data is correctly loaded before computation.

```wgsl
function getSharedTiledWGSL8x8(workgroupSize) {
    const [x, y] = workgroupSize.split(',').map(Number);
    return `
        @group(0) @binding(0) var<storage, read> matrixA : array<f32>;
        @group(0) @binding(1) var<storage, read> matrixB : array<f32>;
        @group(0) @binding(2) var<storage, read_write> matrixC : array<f32>;
        @group(0) @binding(3) var<uniform> uniforms : Uniforms;
        
        struct Uniforms {
            N : u32,
            alpha : f32,
            beta : f32,
        }

        // Shared memory for tiles (shared across the workgroup)
        var<workgroup> tileA : array<array<f32, 8>, 8>;
        var<workgroup> tileB : array<array<f32, 8>, 8>;

        @compute @workgroup_size(${x}, ${y})
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) {
            let row = global_id.y;
            let col = global_id.x;
            let N = uniforms.N;

            var acc = 0.0;
            let numTiles = (N + 8u - 1u) / 8u;

            // Check row and column bounds only once
            let validRow = row < N;
            let validCol = col < N;

            // Loop over the number of tiles
            for (var t = 0u; t < numTiles; t = t + 1u) {
                // Load tileA from matrixA into shared memory only if row is valid
                if (validRow && (t * 8u + local_id.x) < N) {
                    tileA[local_id.y][local_id.x] = matrixA[row * N + t * 8u + local_id.x];
                } else {
                    tileA[local_id.y][local_id.x] = 0.0;
                }

                // Load tileB from matrixB into shared memory only if col is valid
                if (validCol && (t * 8u + local_id.y) < N) {
                    tileB[local_id.y][local_id.x] = matrixB[(t * 8u + local_id.y) * N + col];
                } else {
                    tileB[local_id.y][local_id.x] = 0.0;
                }

                // Synchronize to ensure all threads have loaded their tile data
                workgroupBarrier();

                // Multiply the two tiles and accumulate the result
                for (var k = 0u; k < 8u; k = k + 1u) {
                    acc = acc + tileA[local_id.y][k] * tileB[k][local_id.x];
                }

                // Synchronize before loading the next tile
                workgroupBarrier();
            }

            // Write the result to matrixC only if row and col are valid
            if (validRow && validCol) {
                matrixC[row * N + col] = uniforms.alpha * acc + uniforms.beta * matrixC[row * N + col];
            }
        }
    `;
}
```
