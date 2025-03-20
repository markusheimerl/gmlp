#ifndef GPU_GMLP_H
#define GPU_GMLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

typedef struct {
    // Network dimensions
    int input_dim;
    int hidden_dim;
    int ffn_dim;     // Dimension of feed-forward network
    int output_dim;
    int batch_size;
    
    // Weights
    float* proj_in_weight;   // hidden_dim x input_dim
    float* sgu_gate_weight;  // ffn_dim x (hidden_dim/2)
    float* sgu_proj_weight;  // ffn_dim x (hidden_dim/2)
    float* sgu_out_weight;   // hidden_dim x ffn_dim
    float* proj_out_weight;  // output_dim x hidden_dim
    
    // Device (GPU) weights
    float* d_proj_in_weight;   
    float* d_sgu_gate_weight;  
    float* d_sgu_proj_weight;  
    float* d_sgu_out_weight;   
    float* d_proj_out_weight;  
    
    // Weight gradients
    float* d_proj_in_weight_grad;
    float* d_sgu_gate_weight_grad;
    float* d_sgu_proj_weight_grad;
    float* d_sgu_out_weight_grad;
    float* d_proj_out_weight_grad;
    
    // Host (CPU) weight gradients for updates
    float* proj_in_weight_grad;
    float* sgu_gate_weight_grad;
    float* sgu_proj_weight_grad;
    float* sgu_out_weight_grad;
    float* proj_out_weight_grad;
    
    // Adam optimizer parameters (on host)
    float* proj_in_m;
    float* proj_in_v;
    float* sgu_gate_m;
    float* sgu_gate_v;
    float* sgu_proj_m;
    float* sgu_proj_v;
    float* sgu_out_m;
    float* sgu_out_v;
    float* proj_out_m;
    float* proj_out_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;
    
    // Intermediate activations for forward/backward pass
    float* d_input;           // batch_size x input_dim
    float* d_proj_in_output;  // batch_size x hidden_dim
    float* d_u_part;          // batch_size x (hidden_dim/2) - First half for gate
    float* d_v_part;          // batch_size x (hidden_dim/2) - Second half for projection
    float* d_sgu_gate_output; // batch_size x ffn_dim
    float* d_gate_activated;  // batch_size x ffn_dim
    float* d_sgu_proj_output; // batch_size x ffn_dim
    float* d_gated_output;    // batch_size x ffn_dim - After applying gate
    float* d_sgu_output;      // batch_size x hidden_dim
    float* d_predictions;     // batch_size x output_dim
    
    // Input and output on device
    float* d_X;               // batch_size x input_dim
    float* d_y;               // batch_size x output_dim
    
    // Host predictions for evaluation
    float* predictions;       // batch_size x output_dim
    
    // Intermediate gradients for backward pass
    float* d_error;              // batch_size x output_dim
    float* d_sgu_output_grad;    // batch_size x hidden_dim
    float* d_gated_output_grad;  // batch_size x ffn_dim
    float* d_sgu_proj_grad;      // batch_size x ffn_dim
    float* d_gate_activated_grad;// batch_size x ffn_dim
    float* d_sgu_gate_grad;      // batch_size x ffn_dim
    float* d_u_part_grad;        // batch_size x (hidden_dim/2)
    float* d_v_part_grad;        // batch_size x (hidden_dim/2)
    float* d_proj_in_grad;       // batch_size x hidden_dim
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
} GPU_GMLP;

// CUDA kernel implementations
__global__ void gelu_activation_kernel(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        input[idx] = x * 0.5f * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
    }
}

__global__ void gelu_backward_kernel(float* grad_out, float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Approximate GELU derivative
        float cdf = 0.5f * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
        float pdf = 0.797885f * (1.0f + 0.134145f * x * x) * 
                    (1.0f - tanhf(0.797885f * (x + 0.044715f * x * x * x)) * 
                            tanhf(0.797885f * (x + 0.044715f * x * x * x)));
        float gelu_grad = cdf + x * pdf;
        output[idx] = grad_out[idx] * gelu_grad;
    }
}

__global__ void sigmoid_activation_kernel(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

__global__ void multiply_elements_kernel(float* a, float* b, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] * b[idx];
    }
}

__global__ void error_computation_kernel(float* predictions, float* targets, float* error, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - targets[idx];
    }
}

__global__ void sigmoid_backward_kernel(float* grad_out, float* sigmoid_output, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid = sigmoid_output[idx];
        output[idx] = grad_out[idx] * sigmoid * (1.0f - sigmoid);
    }
}

// Utility function to check CUDA errors
void checkCudaErrors(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

// Utility function to check cuBLAS errors
void checkCublasErrors(cublasStatus_t status, const char* message) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error: %s: %d\n", message, status);
        exit(EXIT_FAILURE);
    }
}

// Initialize gMLP network
GPU_GMLP* init_gpu_gmlp(int input_dim, int hidden_dim, int ffn_dim, int output_dim, int batch_size) {
    GPU_GMLP* gmlp = (GPU_GMLP*)malloc(sizeof(GPU_GMLP));
    if (!gmlp) {
        printf("Failed to allocate memory for GPU_GMLP\n");
        return NULL;
    }
    
    // Store dimensions
    gmlp->input_dim = input_dim;
    gmlp->hidden_dim = hidden_dim;
    gmlp->ffn_dim = ffn_dim;
    gmlp->output_dim = output_dim;
    gmlp->batch_size = batch_size;
    
    // Initialize Adam parameters
    gmlp->beta1 = 0.9f;
    gmlp->beta2 = 0.999f;
    gmlp->epsilon = 1e-8f;
    gmlp->t = 0;
    gmlp->weight_decay = 0.01f;
    
    // Calculate half hidden dimension for SGU
    int half_hidden = hidden_dim / 2;
    
    // Initialize cuBLAS
    checkCublasErrors(cublasCreate(&gmlp->cublas_handle), "cublasCreate failed");
    
    // Allocate weights on host
    gmlp->proj_in_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    gmlp->sgu_gate_weight = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    gmlp->sgu_proj_weight = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    gmlp->sgu_out_weight = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    gmlp->proj_out_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate weight gradients on host for updates
    gmlp->proj_in_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    gmlp->sgu_gate_weight_grad = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    gmlp->sgu_proj_weight_grad = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    gmlp->sgu_out_weight_grad = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    gmlp->proj_out_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate Adam buffers
    gmlp->proj_in_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    gmlp->proj_in_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    gmlp->sgu_gate_m = (float*)calloc(ffn_dim * half_hidden, sizeof(float));
    gmlp->sgu_gate_v = (float*)calloc(ffn_dim * half_hidden, sizeof(float));
    gmlp->sgu_proj_m = (float*)calloc(ffn_dim * half_hidden, sizeof(float));
    gmlp->sgu_proj_v = (float*)calloc(ffn_dim * half_hidden, sizeof(float));
    gmlp->sgu_out_m = (float*)calloc(hidden_dim * ffn_dim, sizeof(float));
    gmlp->sgu_out_v = (float*)calloc(hidden_dim * ffn_dim, sizeof(float));
    gmlp->proj_out_m = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    gmlp->proj_out_v = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    
    // Allocate predictions on host for evaluation
    gmlp->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Allocate memory on device (GPU)
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_proj_in_weight, hidden_dim * input_dim * sizeof(float)),
                  "Allocate d_proj_in_weight");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_gate_weight, ffn_dim * half_hidden * sizeof(float)),
                  "Allocate d_sgu_gate_weight");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_proj_weight, ffn_dim * half_hidden * sizeof(float)),
                  "Allocate d_sgu_proj_weight");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_out_weight, hidden_dim * ffn_dim * sizeof(float)),
                  "Allocate d_sgu_out_weight");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_proj_out_weight, output_dim * hidden_dim * sizeof(float)),
                  "Allocate d_proj_out_weight");
    
    // Allocate weight gradients on device
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_proj_in_weight_grad, hidden_dim * input_dim * sizeof(float)),
                  "Allocate d_proj_in_weight_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_gate_weight_grad, ffn_dim * half_hidden * sizeof(float)),
                  "Allocate d_sgu_gate_weight_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_proj_weight_grad, ffn_dim * half_hidden * sizeof(float)),
                  "Allocate d_sgu_proj_weight_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_out_weight_grad, hidden_dim * ffn_dim * sizeof(float)),
                  "Allocate d_sgu_out_weight_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_proj_out_weight_grad, output_dim * hidden_dim * sizeof(float)),
                  "Allocate d_proj_out_weight_grad");
    
    // Allocate input and output on device
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_X, batch_size * input_dim * sizeof(float)),
                  "Allocate d_X");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_y, batch_size * output_dim * sizeof(float)),
                  "Allocate d_y");
    
    // Allocate intermediate activations on device
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_proj_in_output, batch_size * hidden_dim * sizeof(float)),
                  "Allocate d_proj_in_output");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_gate_output, batch_size * ffn_dim * sizeof(float)),
                  "Allocate d_sgu_gate_output");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_gate_activated, batch_size * ffn_dim * sizeof(float)),
                  "Allocate d_gate_activated");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_proj_output, batch_size * ffn_dim * sizeof(float)),
                  "Allocate d_sgu_proj_output");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_gated_output, batch_size * ffn_dim * sizeof(float)),
                  "Allocate d_gated_output");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_output, batch_size * hidden_dim * sizeof(float)),
                  "Allocate d_sgu_output");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_predictions, batch_size * output_dim * sizeof(float)),
                  "Allocate d_predictions");
    
    // Allocate intermediate gradients on device
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_error, batch_size * output_dim * sizeof(float)),
                  "Allocate d_error");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_output_grad, batch_size * hidden_dim * sizeof(float)),
                  "Allocate d_sgu_output_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_gated_output_grad, batch_size * ffn_dim * sizeof(float)),
                  "Allocate d_gated_output_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_proj_grad, batch_size * ffn_dim * sizeof(float)),
                  "Allocate d_sgu_proj_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_gate_activated_grad, batch_size * ffn_dim * sizeof(float)),
                  "Allocate d_gate_activated_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_sgu_gate_grad, batch_size * ffn_dim * sizeof(float)),
                  "Allocate d_sgu_gate_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_u_part_grad, batch_size * half_hidden * sizeof(float)),
                  "Allocate d_u_part_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_v_part_grad, batch_size * half_hidden * sizeof(float)),
                  "Allocate d_v_part_grad");
    checkCudaErrors(cudaMalloc((void**)&gmlp->d_proj_in_grad, batch_size * hidden_dim * sizeof(float)),
                  "Allocate d_proj_in_grad");
    
    // Set up u_part and v_part pointers
    gmlp->d_u_part = gmlp->d_proj_in_output;
    gmlp->d_v_part = gmlp->d_proj_in_output + batch_size * half_hidden;
    
    // Initialize weights with He initialization
    float scale_in = sqrtf(2.0f / input_dim);
    float scale_half_hidden = sqrtf(2.0f / half_hidden);
    float scale_ffn = sqrtf(2.0f / ffn_dim);
    float scale_hidden = sqrtf(2.0f / hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        gmlp->proj_in_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
    }
    
    for (int i = 0; i < ffn_dim * half_hidden; i++) {
        gmlp->sgu_gate_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_half_hidden;
        gmlp->sgu_proj_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_half_hidden;
    }
    
    for (int i = 0; i < hidden_dim * ffn_dim; i++) {
        gmlp->sgu_out_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_ffn;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        gmlp->proj_out_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_hidden;
    }
    
    // Copy initialized weights to device
    checkCudaErrors(cudaMemcpy(gmlp->d_proj_in_weight, gmlp->proj_in_weight, 
                   hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy proj_in_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_gate_weight, gmlp->sgu_gate_weight, 
                   ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy sgu_gate_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_proj_weight, gmlp->sgu_proj_weight, 
                   ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy sgu_proj_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_out_weight, gmlp->sgu_out_weight, 
                   hidden_dim * ffn_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy sgu_out_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_proj_out_weight, gmlp->proj_out_weight, 
                   output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy proj_out_weight to device");
    
    return gmlp;
}

// Free network memory
void free_gpu_gmlp(GPU_GMLP* gmlp) {
    if (!gmlp) return;
    
    // Free weights on host
    if (gmlp->proj_in_weight) free(gmlp->proj_in_weight);
    if (gmlp->sgu_gate_weight) free(gmlp->sgu_gate_weight);
    if (gmlp->sgu_proj_weight) free(gmlp->sgu_proj_weight);
    if (gmlp->sgu_out_weight) free(gmlp->sgu_out_weight);
    if (gmlp->proj_out_weight) free(gmlp->proj_out_weight);
    
    // Free gradients on host
    if (gmlp->proj_in_weight_grad) free(gmlp->proj_in_weight_grad);
    if (gmlp->sgu_gate_weight_grad) free(gmlp->sgu_gate_weight_grad);
    if (gmlp->sgu_proj_weight_grad) free(gmlp->sgu_proj_weight_grad);
    if (gmlp->sgu_out_weight_grad) free(gmlp->sgu_out_weight_grad);
    if (gmlp->proj_out_weight_grad) free(gmlp->proj_out_weight_grad);
    
    // Free Adam buffers
    if (gmlp->proj_in_m) free(gmlp->proj_in_m);
    if (gmlp->proj_in_v) free(gmlp->proj_in_v);
    if (gmlp->sgu_gate_m) free(gmlp->sgu_gate_m);
    if (gmlp->sgu_gate_v) free(gmlp->sgu_gate_v);
    if (gmlp->sgu_proj_m) free(gmlp->sgu_proj_m);
    if (gmlp->sgu_proj_v) free(gmlp->sgu_proj_v);
    if (gmlp->sgu_out_m) free(gmlp->sgu_out_m);
    if (gmlp->sgu_out_v) free(gmlp->sgu_out_v);
    if (gmlp->proj_out_m) free(gmlp->proj_out_m);
    if (gmlp->proj_out_v) free(gmlp->proj_out_v);
    
    // Free predictions on host
    if (gmlp->predictions) free(gmlp->predictions);
    
    // Free device memory
    cudaFree(gmlp->d_proj_in_weight);
    cudaFree(gmlp->d_sgu_gate_weight);
    cudaFree(gmlp->d_sgu_proj_weight);
    cudaFree(gmlp->d_sgu_out_weight);
    cudaFree(gmlp->d_proj_out_weight);
    
    cudaFree(gmlp->d_proj_in_weight_grad);
    cudaFree(gmlp->d_sgu_gate_weight_grad);
    cudaFree(gmlp->d_sgu_proj_weight_grad);
    cudaFree(gmlp->d_sgu_out_weight_grad);
    cudaFree(gmlp->d_proj_out_weight_grad);
    
    cudaFree(gmlp->d_X);
    cudaFree(gmlp->d_y);
    
    cudaFree(gmlp->d_proj_in_output);
    cudaFree(gmlp->d_sgu_gate_output);
    cudaFree(gmlp->d_gate_activated);
    cudaFree(gmlp->d_sgu_proj_output);
    cudaFree(gmlp->d_gated_output);
    cudaFree(gmlp->d_sgu_output);
    cudaFree(gmlp->d_predictions);
    
    cudaFree(gmlp->d_error);
    cudaFree(gmlp->d_sgu_output_grad);
    cudaFree(gmlp->d_gated_output_grad);
    cudaFree(gmlp->d_sgu_proj_grad);
    cudaFree(gmlp->d_gate_activated_grad);
    cudaFree(gmlp->d_sgu_gate_grad);
    cudaFree(gmlp->d_u_part_grad);
    cudaFree(gmlp->d_v_part_grad);
    cudaFree(gmlp->d_proj_in_grad);
    
    // Destroy cuBLAS handle
    cublasDestroy(gmlp->cublas_handle);
    
    // Free the struct
    free(gmlp);
}

// Forward pass
void forward_pass_gpu_gmlp(GPU_GMLP* gmlp, float* X) {
    int half_hidden = gmlp->hidden_dim / 2;
    
    // Copy input data to device
    checkCudaErrors(cudaMemcpy(gmlp->d_X, X, gmlp->batch_size * gmlp->input_dim * sizeof(float), 
                   cudaMemcpyHostToDevice), "Copy X to device");
    
    // Prepare constants for cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 1. Input projection: X → hidden_dim
    // Note: cuBLAS uses column-major order, transpose the operation
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 gmlp->hidden_dim, gmlp->batch_size, gmlp->input_dim,
                                 &alpha,
                                 gmlp->d_proj_in_weight, gmlp->hidden_dim,
                                 gmlp->d_X, gmlp->input_dim,
                                 &beta,
                                 gmlp->d_proj_in_output, gmlp->hidden_dim), 
                      "Input projection forward");
    
    // Apply GELU activation to the input projection
    int block_size = 256;
    int num_blocks = (gmlp->batch_size * gmlp->hidden_dim + block_size - 1) / block_size;
    gelu_activation_kernel<<<num_blocks, block_size>>>(gmlp->d_proj_in_output, 
                                                      gmlp->batch_size * gmlp->hidden_dim);
    checkCudaErrors(cudaGetLastError(), "GELU activation kernel launch");
    
    // 2. Spatial Gating Unit (SGU)
    // 2a. Compute gate values from first half of hidden states (u)
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 gmlp->ffn_dim, gmlp->batch_size, half_hidden,
                                 &alpha,
                                 gmlp->d_sgu_gate_weight, gmlp->ffn_dim,
                                 gmlp->d_u_part, half_hidden,
                                 &beta,
                                 gmlp->d_sgu_gate_output, gmlp->ffn_dim),
                      "SGU gate forward");
    
    // Apply sigmoid activation to gate values
    num_blocks = (gmlp->batch_size * gmlp->ffn_dim + block_size - 1) / block_size;
    sigmoid_activation_kernel<<<num_blocks, block_size>>>(gmlp->d_sgu_gate_output, gmlp->d_gate_activated, 
                                                         gmlp->batch_size * gmlp->ffn_dim);
    checkCudaErrors(cudaGetLastError(), "Sigmoid activation kernel launch");
    
    // 2b. Project second half of hidden states (v) to FFN dimension
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 gmlp->ffn_dim, gmlp->batch_size, half_hidden,
                                 &alpha,
                                 gmlp->d_sgu_proj_weight, gmlp->ffn_dim,
                                 gmlp->d_v_part, half_hidden,
                                 &beta,
                                 gmlp->d_sgu_proj_output, gmlp->ffn_dim),
                      "SGU projection forward");
    
    // 2c. Apply gating (element-wise multiplication)
    num_blocks = (gmlp->batch_size * gmlp->ffn_dim + block_size - 1) / block_size;
    multiply_elements_kernel<<<num_blocks, block_size>>>(gmlp->d_sgu_proj_output, gmlp->d_gate_activated, 
                                                        gmlp->d_gated_output, gmlp->batch_size * gmlp->ffn_dim);
    checkCudaErrors(cudaGetLastError(), "Element-wise multiplication kernel launch");
    
    // 2d. Project back to hidden dimension
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 gmlp->hidden_dim, gmlp->batch_size, gmlp->ffn_dim,
                                 &alpha,
                                 gmlp->d_sgu_out_weight, gmlp->hidden_dim,
                                 gmlp->d_gated_output, gmlp->ffn_dim,
                                 &beta,
                                 gmlp->d_sgu_output, gmlp->hidden_dim),
                      "SGU output forward");
    
    // 3. Output projection: hidden_dim → output_dim
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                 gmlp->output_dim, gmlp->batch_size, gmlp->hidden_dim,
                                 &alpha,
                                 gmlp->d_proj_out_weight, gmlp->output_dim,
                                 gmlp->d_sgu_output, gmlp->hidden_dim,
                                 &beta,
                                 gmlp->d_predictions, gmlp->output_dim),
                      "Output projection forward");
    
    // Copy predictions back to host for evaluation
    checkCudaErrors(cudaMemcpy(gmlp->predictions, gmlp->d_predictions, 
                   gmlp->batch_size * gmlp->output_dim * sizeof(float), cudaMemcpyDeviceToHost),
                   "Copy predictions to host");
}

// Calculate loss (Mean Squared Error)
float calculate_loss_gpu_gmlp(GPU_GMLP* gmlp, float* y) {
    // Copy targets to device
    checkCudaErrors(cudaMemcpy(gmlp->d_y, y, gmlp->batch_size * gmlp->output_dim * sizeof(float), 
                   cudaMemcpyHostToDevice), "Copy y to device");
    
    // Compute error
    int block_size = 256;
    int num_blocks = (gmlp->batch_size * gmlp->output_dim + block_size - 1) / block_size;
    error_computation_kernel<<<num_blocks, block_size>>>(gmlp->d_predictions, gmlp->d_y, 
                                                        gmlp->d_error, gmlp->batch_size * gmlp->output_dim);
    checkCudaErrors(cudaGetLastError(), "Error computation kernel launch");
    
    // Copy error back to host to compute loss
    float* host_error = (float*)malloc(gmlp->batch_size * gmlp->output_dim * sizeof(float));
    checkCudaErrors(cudaMemcpy(host_error, gmlp->d_error, gmlp->batch_size * gmlp->output_dim * sizeof(float), 
                   cudaMemcpyDeviceToHost), "Copy error to host");
    
    // Compute mean squared error
    float loss = 0.0f;
    for (int i = 0; i < gmlp->batch_size * gmlp->output_dim; i++) {
        loss += host_error[i] * host_error[i];
    }
    loss /= (gmlp->batch_size * gmlp->output_dim);
    
    free(host_error);
    return loss;
}

// Zero out all gradients
void zero_gradients_gpu_gmlp(GPU_GMLP* gmlp) {
    checkCudaErrors(cudaMemset(gmlp->d_proj_in_weight_grad, 0, gmlp->hidden_dim * gmlp->input_dim * sizeof(float)),
                  "Zero d_proj_in_weight_grad");
    checkCudaErrors(cudaMemset(gmlp->d_sgu_gate_weight_grad, 0, gmlp->ffn_dim * (gmlp->hidden_dim/2) * sizeof(float)),
                  "Zero d_sgu_gate_weight_grad");
    checkCudaErrors(cudaMemset(gmlp->d_sgu_proj_weight_grad, 0, gmlp->ffn_dim * (gmlp->hidden_dim/2) * sizeof(float)),
                  "Zero d_sgu_proj_weight_grad");
    checkCudaErrors(cudaMemset(gmlp->d_sgu_out_weight_grad, 0, gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float)),
                  "Zero d_sgu_out_weight_grad");
    checkCudaErrors(cudaMemset(gmlp->d_proj_out_weight_grad, 0, gmlp->output_dim * gmlp->hidden_dim * sizeof(float)),
                  "Zero d_proj_out_weight_grad");
}

// Backward pass
void backward_pass_gpu_gmlp(GPU_GMLP* gmlp, float* X) {
    int half_hidden = gmlp->hidden_dim / 2;
    
    // Prepare constants for cuBLAS
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 1. Gradient of output projection
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 gmlp->output_dim, gmlp->hidden_dim, gmlp->batch_size,
                                 &alpha,
                                 gmlp->d_error, gmlp->output_dim,
                                 gmlp->d_sgu_output, gmlp->hidden_dim,
                                 &beta,
                                 gmlp->d_proj_out_weight_grad, gmlp->output_dim),
                      "Output projection weight gradient");
    
    // 2. Gradient flowing back to SGU output
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 gmlp->hidden_dim, gmlp->batch_size, gmlp->output_dim,
                                 &alpha,
                                 gmlp->d_proj_out_weight, gmlp->output_dim,
                                 gmlp->d_error, gmlp->output_dim,
                                 &beta,
                                 gmlp->d_sgu_output_grad, gmlp->hidden_dim),
                      "Gradient to SGU output");
    
    // 3. Gradient of SGU output weight
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 gmlp->hidden_dim, gmlp->ffn_dim, gmlp->batch_size,
                                 &alpha,
                                 gmlp->d_sgu_output_grad, gmlp->hidden_dim,
                                 gmlp->d_gated_output, gmlp->ffn_dim,
                                 &beta,
                                 gmlp->d_sgu_out_weight_grad, gmlp->hidden_dim),
                      "SGU output weight gradient");
    
    // 4. Gradient flowing back to gated_output
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 gmlp->ffn_dim, gmlp->batch_size, gmlp->hidden_dim,
                                 &alpha,
                                 gmlp->d_sgu_out_weight, gmlp->hidden_dim,
                                 gmlp->d_sgu_output_grad, gmlp->hidden_dim,
                                 &beta,
                                 gmlp->d_gated_output_grad, gmlp->ffn_dim),
                      "Gradient to gated output");
    
    // 5. Gradient through the gating mechanism
    // 5a. Gradient to sgu_proj_output
    int block_size = 256;
    int num_blocks = (gmlp->batch_size * gmlp->ffn_dim + block_size - 1) / block_size;
    multiply_elements_kernel<<<num_blocks, block_size>>>(gmlp->d_gated_output_grad, gmlp->d_gate_activated, 
                                                        gmlp->d_sgu_proj_grad, gmlp->batch_size * gmlp->ffn_dim);
    checkCudaErrors(cudaGetLastError(), "Gradient to SGU proj output kernel launch");
    
    // 5b. Gradient to gate_activated
    multiply_elements_kernel<<<num_blocks, block_size>>>(gmlp->d_gated_output_grad, gmlp->d_sgu_proj_output, 
                                                        gmlp->d_gate_activated_grad, gmlp->batch_size * gmlp->ffn_dim);
    checkCudaErrors(cudaGetLastError(), "Gradient to gate activated kernel launch");
    
    // 5c. Gradient through sigmoid for gate
    sigmoid_backward_kernel<<<num_blocks, block_size>>>(gmlp->d_gate_activated_grad, gmlp->d_gate_activated, 
                                                       gmlp->d_sgu_gate_grad, gmlp->batch_size * gmlp->ffn_dim);
    checkCudaErrors(cudaGetLastError(), "Sigmoid backward kernel launch");
    
    // 6. Gradient of sgu_proj_weight
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 gmlp->ffn_dim, half_hidden, gmlp->batch_size,
                                 &alpha,
                                 gmlp->d_sgu_proj_grad, gmlp->ffn_dim,
                                 gmlp->d_v_part, half_hidden,
                                 &beta,
                                 gmlp->d_sgu_proj_weight_grad, gmlp->ffn_dim),
                      "SGU projection weight gradient");
    
    // 7. Gradient of sgu_gate_weight
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 gmlp->ffn_dim, half_hidden, gmlp->batch_size,
                                 &alpha,
                                 gmlp->d_sgu_gate_grad, gmlp->ffn_dim,
                                 gmlp->d_u_part, half_hidden,
                                 &beta,
                                 gmlp->d_sgu_gate_weight_grad, gmlp->ffn_dim),
                      "SGU gate weight gradient");
    
    // 8. Gradient flowing back to v_part
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 half_hidden, gmlp->batch_size, gmlp->ffn_dim,
                                 &alpha,
                                 gmlp->d_sgu_proj_weight, gmlp->ffn_dim,
                                 gmlp->d_sgu_proj_grad, gmlp->ffn_dim,
                                 &beta,
                                 gmlp->d_v_part_grad, half_hidden),
                      "Gradient to v_part");
    
    // 9. Gradient flowing back to u_part
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                 half_hidden, gmlp->batch_size, gmlp->ffn_dim,
                                 &alpha,
                                 gmlp->d_sgu_gate_weight, gmlp->ffn_dim,
                                 gmlp->d_sgu_gate_grad, gmlp->ffn_dim,
                                 &beta,
                                 gmlp->d_u_part_grad, half_hidden),
                      "Gradient to u_part");
    
    // 10. Combine u_part_grad and v_part_grad to form proj_in_grad
    checkCudaErrors(cudaMemcpy(gmlp->d_proj_in_grad, gmlp->d_u_part_grad, 
                   gmlp->batch_size * half_hidden * sizeof(float), cudaMemcpyDeviceToDevice),
                   "Copy u_part_grad to proj_in_grad");
    checkCudaErrors(cudaMemcpy(gmlp->d_proj_in_grad + gmlp->batch_size * half_hidden, gmlp->d_v_part_grad, 
                   gmlp->batch_size * half_hidden * sizeof(float), cudaMemcpyDeviceToDevice),
                   "Copy v_part_grad to proj_in_grad");
    
    // 11. Gradient through GELU activation
    num_blocks = (gmlp->batch_size * gmlp->hidden_dim + block_size - 1) / block_size;
    gelu_backward_kernel<<<num_blocks, block_size>>>(gmlp->d_proj_in_grad, gmlp->d_proj_in_output, 
                                                   gmlp->d_proj_in_grad, gmlp->batch_size * gmlp->hidden_dim);
    checkCudaErrors(cudaGetLastError(), "GELU backward kernel launch");
    
    // 12. Gradient to input projection weights
    // Copy input data to device again if it's not already there
    checkCudaErrors(cudaMemcpy(gmlp->d_X, X, gmlp->batch_size * gmlp->input_dim * sizeof(float), 
                   cudaMemcpyHostToDevice), "Copy X to device for backward pass");
    
    checkCublasErrors(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                                 gmlp->hidden_dim, gmlp->input_dim, gmlp->batch_size,
                                 &alpha,
                                 gmlp->d_proj_in_grad, gmlp->hidden_dim,
                                 gmlp->d_X, gmlp->input_dim,
                                 &beta,
                                 gmlp->d_proj_in_weight_grad, gmlp->hidden_dim),
                      "Input projection weight gradient");
    
    // Copy gradients back to host for weight updates
    checkCudaErrors(cudaMemcpy(gmlp->proj_in_weight_grad, gmlp->d_proj_in_weight_grad, 
                   gmlp->hidden_dim * gmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost),
                   "Copy proj_in_weight_grad to host");
    checkCudaErrors(cudaMemcpy(gmlp->sgu_gate_weight_grad, gmlp->d_sgu_gate_weight_grad, 
                   gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyDeviceToHost),
                   "Copy sgu_gate_weight_grad to host");
    checkCudaErrors(cudaMemcpy(gmlp->sgu_proj_weight_grad, gmlp->d_sgu_proj_weight_grad, 
                   gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyDeviceToHost),
                   "Copy sgu_proj_weight_grad to host");
    checkCudaErrors(cudaMemcpy(gmlp->sgu_out_weight_grad, gmlp->d_sgu_out_weight_grad, 
                   gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float), cudaMemcpyDeviceToHost),
                   "Copy sgu_out_weight_grad to host");
    checkCudaErrors(cudaMemcpy(gmlp->proj_out_weight_grad, gmlp->d_proj_out_weight_grad, 
                   gmlp->output_dim * gmlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost),
                   "Copy proj_out_weight_grad to host");
}

// Update weights using AdamW
void update_weights_gpu_gmlp(GPU_GMLP* gmlp, float learning_rate) {
    gmlp->t++;  // Increment time step
    int half_hidden = gmlp->hidden_dim / 2;
    
    float beta1_t = powf(gmlp->beta1, gmlp->t);
    float beta2_t = powf(gmlp->beta2, gmlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Update projection in weights
    for (int i = 0; i < gmlp->hidden_dim * gmlp->input_dim; i++) {
        float grad = gmlp->proj_in_weight_grad[i] / gmlp->batch_size;
        
        gmlp->proj_in_m[i] = gmlp->beta1 * gmlp->proj_in_m[i] + (1.0f - gmlp->beta1) * grad;
        gmlp->proj_in_v[i] = gmlp->beta2 * gmlp->proj_in_v[i] + (1.0f - gmlp->beta2) * grad * grad;
        
        float update = alpha_t * gmlp->proj_in_m[i] / (sqrtf(gmlp->proj_in_v[i]) + gmlp->epsilon);
        gmlp->proj_in_weight[i] = gmlp->proj_in_weight[i] * (1.0f - learning_rate * gmlp->weight_decay) - update;
    }
    
    // Update SGU gate weights
    for (int i = 0; i < gmlp->ffn_dim * half_hidden; i++) {
        float grad = gmlp->sgu_gate_weight_grad[i] / gmlp->batch_size;
        
        gmlp->sgu_gate_m[i] = gmlp->beta1 * gmlp->sgu_gate_m[i] + (1.0f - gmlp->beta1) * grad;
        gmlp->sgu_gate_v[i] = gmlp->beta2 * gmlp->sgu_gate_v[i] + (1.0f - gmlp->beta2) * grad * grad;
        
        float update = alpha_t * gmlp->sgu_gate_m[i] / (sqrtf(gmlp->sgu_gate_v[i]) + gmlp->epsilon);
        gmlp->sgu_gate_weight[i] = gmlp->sgu_gate_weight[i] * (1.0f - learning_rate * gmlp->weight_decay) - update;
    }
    
    // Update SGU projection weights
    for (int i = 0; i < gmlp->ffn_dim * half_hidden; i++) {
        float grad = gmlp->sgu_proj_weight_grad[i] / gmlp->batch_size;
        
        gmlp->sgu_proj_m[i] = gmlp->beta1 * gmlp->sgu_proj_m[i] + (1.0f - gmlp->beta1) * grad;
        gmlp->sgu_proj_v[i] = gmlp->beta2 * gmlp->sgu_proj_v[i] + (1.0f - gmlp->beta2) * grad * grad;
        
        float update = alpha_t * gmlp->sgu_proj_m[i] / (sqrtf(gmlp->sgu_proj_v[i]) + gmlp->epsilon);
        gmlp->sgu_proj_weight[i] = gmlp->sgu_proj_weight[i] * (1.0f - learning_rate * gmlp->weight_decay) - update;
    }
    
    // Update SGU output weights
    for (int i = 0; i < gmlp->hidden_dim * gmlp->ffn_dim; i++) {
        float grad = gmlp->sgu_out_weight_grad[i] / gmlp->batch_size;
        
        gmlp->sgu_out_m[i] = gmlp->beta1 * gmlp->sgu_out_m[i] + (1.0f - gmlp->beta1) * grad;
        gmlp->sgu_out_v[i] = gmlp->beta2 * gmlp->sgu_out_v[i] + (1.0f - gmlp->beta2) * grad * grad;
        
        float update = alpha_t * gmlp->sgu_out_m[i] / (sqrtf(gmlp->sgu_out_v[i]) + gmlp->epsilon);
        gmlp->sgu_out_weight[i] = gmlp->sgu_out_weight[i] * (1.0f - learning_rate * gmlp->weight_decay) - update;
    }
    
    // Update projection out weights
    for (int i = 0; i < gmlp->output_dim * gmlp->hidden_dim; i++) {
        float grad = gmlp->proj_out_weight_grad[i] / gmlp->batch_size;
        
        gmlp->proj_out_m[i] = gmlp->beta1 * gmlp->proj_out_m[i] + (1.0f - gmlp->beta1) * grad;
        gmlp->proj_out_v[i] = gmlp->beta2 * gmlp->proj_out_v[i] + (1.0f - gmlp->beta2) * grad * grad;
        
        float update = alpha_t * gmlp->proj_out_m[i] / (sqrtf(gmlp->proj_out_v[i]) + gmlp->epsilon);
        gmlp->proj_out_weight[i] = gmlp->proj_out_weight[i] * (1.0f - learning_rate * gmlp->weight_decay) - update;
    }
    
    // Copy updated weights back to device
    checkCudaErrors(cudaMemcpy(gmlp->d_proj_in_weight, gmlp->proj_in_weight, 
                   gmlp->hidden_dim * gmlp->input_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy updated proj_in_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_gate_weight, gmlp->sgu_gate_weight, 
                   gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy updated sgu_gate_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_proj_weight, gmlp->sgu_proj_weight, 
                   gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy updated sgu_proj_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_out_weight, gmlp->sgu_out_weight, 
                   gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy updated sgu_out_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_proj_out_weight, gmlp->proj_out_weight, 
                   gmlp->output_dim * gmlp->hidden_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy updated proj_out_weight to device");
}

// Save model weights to binary file
void save_gpu_gmlp(GPU_GMLP* gmlp, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    int half_hidden = gmlp->hidden_dim / 2;
    
    // Save dimensions
    fwrite(&gmlp->input_dim, sizeof(int), 1, file);
    fwrite(&gmlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&gmlp->ffn_dim, sizeof(int), 1, file);
    fwrite(&gmlp->output_dim, sizeof(int), 1, file);
    fwrite(&gmlp->batch_size, sizeof(int), 1, file);
    
    // Save weights
    fwrite(gmlp->proj_in_weight, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(gmlp->sgu_gate_weight, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(gmlp->sgu_proj_weight, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(gmlp->sgu_out_weight, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(gmlp->proj_out_weight, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    
    // Save Adam state
    fwrite(&gmlp->t, sizeof(int), 1, file);
    fwrite(gmlp->proj_in_m, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(gmlp->proj_in_v, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(gmlp->sgu_gate_m, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(gmlp->sgu_gate_v, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(gmlp->sgu_proj_m, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(gmlp->sgu_proj_v, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(gmlp->sgu_out_m, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(gmlp->sgu_out_v, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(gmlp->proj_out_m, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    fwrite(gmlp->proj_out_v, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
GPU_GMLP* load_gpu_gmlp(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening file for reading: %s\n", filename);
        return NULL;
    }
    
    // Read dimensions
    int input_dim, hidden_dim, ffn_dim, output_dim, stored_batch_size;
    fread(&input_dim, sizeof(int), 1, file);
    fread(&hidden_dim, sizeof(int), 1, file);
    fread(&ffn_dim, sizeof(int), 1, file);
    fread(&output_dim, sizeof(int), 1, file);
    fread(&stored_batch_size, sizeof(int), 1, file);
    
    // Use custom_batch_size if provided, otherwise use stored value
    int batch_size = (custom_batch_size > 0) ? custom_batch_size : stored_batch_size;
    
    // Initialize network
    GPU_GMLP* gmlp = init_gpu_gmlp(input_dim, hidden_dim, ffn_dim, output_dim, batch_size);
    if (!gmlp) {
        fclose(file);
        return NULL;
    }
    
    int half_hidden = hidden_dim / 2;
    
    // Load weights
    fread(gmlp->proj_in_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(gmlp->sgu_gate_weight, sizeof(float), ffn_dim * half_hidden, file);
    fread(gmlp->sgu_proj_weight, sizeof(float), ffn_dim * half_hidden, file);
    fread(gmlp->sgu_out_weight, sizeof(float), hidden_dim * ffn_dim, file);
    fread(gmlp->proj_out_weight, sizeof(float), output_dim * hidden_dim, file);
    
    // Load Adam state
    fread(&gmlp->t, sizeof(int), 1, file);
    fread(gmlp->proj_in_m, sizeof(float), hidden_dim * input_dim, file);
    fread(gmlp->proj_in_v, sizeof(float), hidden_dim * input_dim, file);
    fread(gmlp->sgu_gate_m, sizeof(float), ffn_dim * half_hidden, file);
    fread(gmlp->sgu_gate_v, sizeof(float), ffn_dim * half_hidden, file);
    fread(gmlp->sgu_proj_m, sizeof(float), ffn_dim * half_hidden, file);
    fread(gmlp->sgu_proj_v, sizeof(float), ffn_dim * half_hidden, file);
    fread(gmlp->sgu_out_m, sizeof(float), hidden_dim * ffn_dim, file);
    fread(gmlp->sgu_out_v, sizeof(float), hidden_dim * ffn_dim, file);
    fread(gmlp->proj_out_m, sizeof(float), output_dim * hidden_dim, file);
    fread(gmlp->proj_out_v, sizeof(float), output_dim * hidden_dim, file);
    
    fclose(file);
    
    // Copy loaded weights to device
    checkCudaErrors(cudaMemcpy(gmlp->d_proj_in_weight, gmlp->proj_in_weight, 
                   hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy loaded proj_in_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_gate_weight, gmlp->sgu_gate_weight, 
                   ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy loaded sgu_gate_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_proj_weight, gmlp->sgu_proj_weight, 
                   ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy loaded sgu_proj_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_sgu_out_weight, gmlp->sgu_out_weight, 
                   hidden_dim * ffn_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy loaded sgu_out_weight to device");
    checkCudaErrors(cudaMemcpy(gmlp->d_proj_out_weight, gmlp->proj_out_weight, 
                   output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice),
                   "Copy loaded proj_out_weight to device");
    
    printf("Model loaded from %s\n", filename);
    
    return gmlp;
}

#endif