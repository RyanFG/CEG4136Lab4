#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// Taille de la matrice (modifiable)
#define N 4

// Kernel de transposition avec mémoire partagée et padding
__global__ void matrixTransposeNoPadding(float* input, float* floatoutput) {
    __shared__ float tile[32][32]; // Shared memory without padding

    int x = blockIdx.x * 32 + threadIdx.x; // Global index x
    int y = blockIdx.y * 32 + threadIdx.y; // Global index y

    // Load data into shared memory
    if (x < N && y < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }

    __syncthreads(); // Synchronize threads to ensure data is loaded

    // Transpose indices for output
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;

    // Write transposed data to global memory
    if (x < N && y < N) {
        floatoutput[y * N + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Fonction principale
int main() {
    // Allocation de la mémoire hôte
    size_t size = N * N * sizeof(float);
    float* h_input = (float*)malloc(size);
    float* h_output = (float*)malloc(size);

    // Initialisation de la matrice d'entrée
    cout << "Initial: " << endl;
    for (int i = 0; i < N * N; ++i) {
        h_input[i] = static_cast<float>(i);
        cout << h_input[i] << endl;
    }

    // Allocation de la mémoire device
    float* d_input, * d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copie de la matrice d'entrée vers le device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Dimensions de la grille et des blocs
    dim3 block(32, 32); // Taille du bloc
    dim3 grid((N + 31) / 32, (N + 31) / 32); // Taille de la grille

    // Exécution du kernel
    matrixTransposeShared << <grid, block >> > (d_input, d_output);
    cudaDeviceSynchronize();

    // Copie des résultats vers l'hôte
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Vérification du résultat
    bool success = true;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (h_output[j * N + i] != h_input[i * N + j]) {
                success = false;
                break;
            }
        }
    }

    if (success) {
        cout << "Transposition correcte !" << endl;

        cout << "Final: " << endl;
        for (int i = 0; i < N * N; ++i) {
            cout << h_output[i] << endl;
        }
    }
    else {
        cout << "Erreur dans la transposition." << endl;
    }

    // Libération de la mémoire
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
