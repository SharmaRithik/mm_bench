# WebGPU Matrix Multiplication Benchmark

## Matrix multiplication methods

Naive Implementation:
Simple approach using nested loops. Each thread computes one output element by iterating through a row of matrix A and a column of matrix B. Straightforward but may be inefficient for large matrices due to repeated global memory accesses.

```wgsl
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let N = ${N}u;
    let row = global_id.y;
    let col = global_id.x;

    if (row < N && col < N) {
        var sum = 0.0;
        for (var i = 0u; i < N; i = i + 1u) {
            sum = sum + matrixA[row * N + i] * matrixB[i * N + col];
        }
        matrixC[row * N + col] = sum;
    }
}

Tiled Implementation:
Uses tiling to improve memory access patterns. Loads subsets of input matrices into shared memory tiles. Computes partial sums within tiles, reducing global memory accesses. Uses workgroup barriers for synchronization. More efficient than naive for larger matrices.

```wgsl
@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) {
    let N = ${N}u;
    let row = global_id.y;
    let col = global_id.x;
    let localRow = local_id.y;
    let localCol = local_id.x;

    var sum = 0.0;
    let numTiles = (N + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tiles into shared memory
        // ...

        workgroupBarrier();

        // Compute partial sums
        for (var k = 0u; k < ${TILE_SIZE}u; k = k + 1u) {
            sum = sum + tileA[localRow][k] * tileB[k][localCol];
        }

        workgroupBarrier();
    }

    if (row < N && col < N) {
        matrixC[row * N + col] = sum;
    }
}

Vectorized Implementation:
Utilizes vector operations to process multiple elements simultaneously. Loads 4 elements at a time into vec4 variables, performing element-wise operations. Reduces main loop iterations by a factor of 4. Can improve performance through better utilization of SIMD-like capabilities.

```wgsl
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    let N = ${N}u;
    let row = global_id.y;
    let col = global_id.x;

    if (row < N && col < N) {
        var sum = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        for (var i = 0u; i < N; i = i + 4u) {
            let aRow = vec4<f32>(
                matrixA[row * N + i],
                matrixA[row * N + i + 1u],
                matrixA[row * N + i + 2u],
                matrixA[row * N + i + 3u]
            );
            let bCol = vec4<f32>(
                matrixB[i * N + col],
                matrixB[(i + 1u) * N + col],
                matrixB[(i + 2u) * N + col],
                matrixB[(i + 3u) * N + col]
            );
            sum = sum + aRow * bCol;
        }
        matrixC[row * N + col] = sum.x + sum.y + sum.z + sum.w;
    }
}

Shared Memory Implementation:
Similar to tiled, but explicitly uses workgroup shared memory. Loads data into shared arrays, computes using these faster memory tiles. Employs workgroup barriers for synchronization. Can significantly improve performance for large matrices by reducing memory latency and improving data locality.

```wgsl
var<workgroup> tileA : array<array<f32, ${TILE_SIZE}>, ${TILE_SIZE}>;
var<workgroup> tileB : array<array<f32, ${TILE_SIZE}>, ${TILE_SIZE}>;

@compute @workgroup_size(${TILE_SIZE}, ${TILE_SIZE})
fn main(@builtin(global_invocation_id) global_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) {
    let N = ${N}u;
    let row = global_id.y;
    let col = global_id.x;
    let localRow = local_id.y;
    let localCol = local_id.x;

    var sum = 0.0;
    let numTiles = (N + ${TILE_SIZE - 1}u) / ${TILE_SIZE}u;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        let tileARow = row;
        let tileACol = t * ${TILE_SIZE}u + localCol;
        if (tileARow < N && tileACol < N) {
            tileA[localRow][localCol] = matrixA[tileARow * N + tileACol];
        } else {
            tileA[localRow][localCol] = 0.0;
        }

        let tileBRow = t * ${TILE_SIZE}u + localRow;
        let tileBCol = col;
        if (tileBRow < N && tileBCol < N) {
            tileB[localRow][localCol] = matrixB[tileBRow * N + tileBCol];
        } else {
            tileB[localRow][localCol] = 0.0;
        }

        workgroupBarrier();

        for (var k = 0u; k < ${TILE_SIZE}u; k = k + 1u) {
            sum = sum + tileA[localRow][k] * tileB[k][localCol];
        }

        workgroupBarrier();
    }

    if (row < N && col < N) {
        matrixC[row * N + col] = sum;
    }
}
