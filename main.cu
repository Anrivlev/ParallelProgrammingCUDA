// Ivlev Andrey B19 - 511
// Variant generation:
const int ID = 4; // Student ID
const int G = 511; // Group
const int X = G * 2 + ID; // X = 1026
const int A = X % 4; // A = 2
const int B = 5 + X % 5; // B = 6

#include <iostream>

__device__ int isSuitable(int R, int G, int B)
{
    if (R * G * B < 1000) return 1;
    else return 0;
}

__global__ void _cuda_parallel_pixels_counting(int size, int* data, int* d_number_of_pixels)
{
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = globalIdx; i < size; i+=3*B)
    {
        atomicAdd(&d_number_of_pixels[blockIdx.x], isSuitable((int)data[i], (int)data[i + 1], (int)data[i + 2]));
        // globalIdx += blockDim.x * gridDim.x;
        __syncthreads();
    }
}

int main() {
    // BMP file reading
    const char* filename = (char*) R"(C:\Users\warcr\CLionProjects\ParallelProgrammingCUDA\images\img01.bmp)";

    int i;
    FILE* f = fopen(filename, "rb");
    unsigned char info[54];

    // read the 54-byte header
    fread(info, sizeof(unsigned char), 54, f);

    // extract image height and width from header
    int width = *(int*)&info[18];
    int height = *(int*)&info[22];
    std::cout << "width: " << width << std::endl;
    std::cout << "height: " << height << std::endl;

    // allocate 3 bytes per pixel
    int size = 3 * width * height;
    auto* data = new unsigned char[size];

    // read the rest of the data at once
    fread(data, sizeof(unsigned char), size, f);
    fclose(f);
    for (i = 0; i < size; i += 3)
    {
        // flip the order of every 3 bytes in order to get RGB instead of BGR
        unsigned char tmp = data[i];
        data[i] = data[i + 2];
        data[i + 2] = tmp;
    }
    // data is read

    int* number_of_pixels = new int[B];
    int* d_data;
    int* d_number_of_pixels;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_number_of_pixels, B * sizeof(int));
    cudaMemcpy(d_data, data, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridSize = dim3(1, 1, 1);
    dim3 blockSize = dim3(B, 1, 1);

    _cuda_parallel_pixels_counting<<<gridSize, blockSize>>>(size, d_data, d_number_of_pixels);

    cudaDeviceSynchronize();
    cudaMemcpy(number_of_pixels, d_number_of_pixels, B * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_number_of_pixels);

    int total_number_of_pixels = 0;
    std::cout << "Thread results: [";
    for (int i = 0; i < B; i++)
    {
        total_number_of_pixels += number_of_pixels[i];
        std::cout << number_of_pixels[i];
        if (i != B-1)  std::cout << " ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Total number of suitable pixels (CUDA): " << total_number_of_pixels << std::endl;
    std::cout << std::endl;
    return 0;
}
