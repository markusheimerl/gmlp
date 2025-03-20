#ifndef GMLP_H
#define GMLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// CUDA Error checking macro
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// cuBLAS Error checking macro
#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, \
                (int)status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

typedef struct {
    // Network dimensions
    int input_dim;
    int hidden_dim;
    int ffn_dim;     // Dimension of feed-forward network
    int output_dim;
    int batch_size;
    
    // Weights
    float* proj_in_weight;   // hidden_dim x input_dim
    float* sgu_gate_weight;  // ffn_dim x hidden_dim
    float* sgu_proj_weight;  // hidden_dim x ffn_dim
    float* proj_out_weight;  // output_dim x hidden_dim
    
    // Weight gradients
    float* proj_in_weight_grad;
    float* sgu_gate_weight_grad;
    float* sgu_proj_weight_grad;
    float* proj_out_weight_grad;
    
    // Adam optimizer parameters
    float* proj_in_m;
    float* proj_in_v;
    float* sgu_gate_m;
    float* sgu_gate_v;
    float* sgu_proj_m;
    float* sgu_proj_v;
    float* proj_out_m;
    float* proj_out_v;
    float beta1;
    float beta2;
    float epsilon;
    int t;
    float weight_decay;
    
    // Intermediate activations for forward/backward pass
    float* proj_in_output;    // batch_size x hidden_dim
    float* sgu_gate_output;   // batch_size x ffn_dim
    float* gate_activated;    // batch_size x ffn_dim
    float* sgu_proj_input;    // batch_size x ffn_dim
    float* sgu_output;        // batch_size x hidden_dim
    float* predictions;       // batch_size x output_dim
    
    // Intermediate gradients for backward pass
    float* error;             // batch_size x output_dim
    float* sgu_output_grad;   // batch_size x hidden_dim
    float* sgu_proj_grad;     // batch_size x ffn_dim
    float* gate_activated_grad;  // batch_size x ffn_dim
    float* sgu_gate_grad;     // batch_size x ffn_dim
    float* proj_in_grad;      // batch_size x hidden_dim
    
    // Device copies
    float* d_proj_in_weight;
    float* d_sgu_gate_weight;
    float* d_sgu_proj_weight;
    float* d_proj_out_weight;
    
    float* d_proj_in_weight_grad;
    float* d_sgu_gate_weight_grad;
    float* d_sgu_proj_weight_grad;
    float* d_proj_out_weight_grad;
    
    float* d_proj_in_m;
    float* d_proj_in_v;
    float* d_sgu_gate_m;
    float* d_sgu_gate_v;
    float* d_sgu_proj_m;
    float* d_sgu_proj_v;
    float* d_proj_out_m;
    float* d_proj_out_v;
    
    float* d_proj_in_output;
    float* d_sgu_gate_output;
    float* d_gate_activated;
    float* d_sgu_proj_input;
    float* d_sgu_output;
    float* d_predictions;
    
    float* d_error;
    float* d_sgu_output_grad;
    float* d_sgu_proj_grad;
    float* d_gate_activated_grad;
    float* d_sgu_gate_grad;
    float* d_proj_in_grad;
    
    float* d_X;
    float* d_y;
    
    // cuBLAS handle
    cublasHandle_t cublas_handle;
} GMLP;

// CUDA kernel for GELU activation
__global__ void gelu_kernel(float* output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        output[idx] = x * 0.5f * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
    }
}

// CUDA kernel for GELU derivative
__global__ void gelu_derivative_kernel(float* grad_input, float* x, float* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = x[idx];
        // Approximate GELU derivative
        float cdf = 0.5f * (1.0f + tanhf(0.797885f * (val + 0.044715f * val * val * val)));
        float pdf = 0.797885f * (1.0f + 0.134145f * val * val) * 
                   (1.0f - tanhf(0.797885f * (val + 0.044715f * val * val * val)) * 
                          tanhf(0.797885f * (val + 0.044715f * val * val * val)));
        float gelu_grad = cdf + val * pdf;
        grad_input[idx] = grad_output[idx] * gelu_grad;
    }
}

// CUDA kernel for sigmoid activation
__global__ void sigmoid_kernel(float* output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// CUDA kernel for element-wise multiplication
__global__ void element_wise_multiply_kernel(float* output, float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] *= input[idx];
    }
}

// CUDA kernel for sigmoid derivative and gradient calculation
__global__ void sigmoid_derivative_kernel(float* grad, float* sigmoid_output, float* grad_output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid = sigmoid_output[idx];
        grad[idx] = grad_output[idx] * sigmoid * (1.0f - sigmoid);
    }
}

// CUDA kernel for gate activation gradient
__global__ void gate_activation_grad_kernel(float* gate_activated_grad, float* sgu_proj_grad, 
                                          float* sgu_proj_input, float* gate_activated, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        gate_activated_grad[idx] = sgu_proj_grad[idx] * sgu_proj_input[idx] / gate_activated[idx];
    }
}

// CUDA kernel for error calculation
__global__ void calc_error_kernel(float* error, float* predictions, float* y, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - y[idx];
    }
}

// CUDA kernel for AdamW update
__global__ void adamw_update_kernel(float* weight, float* grad, float* m, float* v,
                                  float beta1, float beta2, float epsilon, 
                                  float learning_rate, float weight_decay, float alpha_t,
                                  int batch_size, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / batch_size;
        
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        
        float update = alpha_t * m[idx] / (sqrtf(v[idx]) + epsilon);
        weight[idx] = weight[idx] * (1.0f - learning_rate * weight_decay) - update;
    }
}

// Initialize gMLP network
GMLP* init_gmlp(int input_dim, int hidden_dim, int ffn_dim, int output_dim, int batch_size) {
    GMLP* gmlp = (GMLP*)malloc(sizeof(GMLP));
    
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
    
    // Initialize cuBLAS
    CHECK_CUBLAS(cublasCreate(&gmlp->cublas_handle));
    
    // Allocate host memory
    gmlp->proj_in_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    gmlp->sgu_gate_weight = (float*)malloc(ffn_dim * hidden_dim * sizeof(float));
    gmlp->sgu_proj_weight = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    gmlp->proj_out_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    gmlp->proj_in_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    gmlp->sgu_gate_weight_grad = (float*)malloc(ffn_dim * hidden_dim * sizeof(float));
    gmlp->sgu_proj_weight_grad = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    gmlp->proj_out_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    gmlp->proj_in_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    gmlp->proj_in_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    gmlp->sgu_gate_m = (float*)calloc(ffn_dim * hidden_dim, sizeof(float));
    gmlp->sgu_gate_v = (float*)calloc(ffn_dim * hidden_dim, sizeof(float));
    gmlp->sgu_proj_m = (float*)calloc(hidden_dim * ffn_dim, sizeof(float));
    gmlp->sgu_proj_v = (float*)calloc(hidden_dim * ffn_dim, sizeof(float));
    gmlp->proj_out_m = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    gmlp->proj_out_v = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    
    gmlp->proj_in_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    gmlp->sgu_gate_output = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->gate_activated = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->sgu_proj_input = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->sgu_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    gmlp->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    gmlp->error = (float*)malloc(batch_size * output_dim * sizeof(float));
    gmlp->sgu_output_grad = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    gmlp->sgu_proj_grad = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->gate_activated_grad = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->sgu_gate_grad = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->proj_in_grad = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    
    // Initialize weights with He initialization
    float scale_in = sqrtf(2.0f / input_dim);
    float scale_hidden = sqrtf(2.0f / hidden_dim);
    float scale_ffn = sqrtf(2.0f / ffn_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        gmlp->proj_in_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
    }
    
    for (int i = 0; i < ffn_dim * hidden_dim; i++) {
        gmlp->sgu_gate_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_hidden;
    }
    
    for (int i = 0; i < hidden_dim * ffn_dim; i++) {
        gmlp->sgu_proj_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_ffn;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        gmlp->proj_out_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_hidden;
    }
    
    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_weight, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_weight, ffn_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_weight, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_out_weight, output_dim * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_weight_grad, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_weight_grad, ffn_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_weight_grad, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_out_weight_grad, output_dim * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_m, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_v, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_m, ffn_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_v, ffn_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_m, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_v, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_out_m, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_out_v, output_dim * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_output, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_gate_activated, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_input, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_predictions, batch_size * output_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&gmlp->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_output_grad, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_grad, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_gate_activated_grad, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_grad, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_grad, batch_size * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMalloc(&gmlp->d_X, batch_size * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_y, batch_size * output_dim * sizeof(float)));
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_weight, gmlp->proj_in_weight, 
                         hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_weight, gmlp->sgu_gate_weight, 
                         ffn_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_weight, gmlp->sgu_proj_weight, 
                         hidden_dim * ffn_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_weight, gmlp->proj_out_weight, 
                         output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Initialize Adam buffers to zero
    CHECK_CUDA(cudaMemset(gmlp->d_proj_in_m, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_proj_in_v, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_gate_m, 0, ffn_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_gate_v, 0, ffn_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_proj_m, 0, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_proj_v, 0, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_proj_out_m, 0, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_proj_out_v, 0, output_dim * hidden_dim * sizeof(float)));
    
    return gmlp;
}

// Free network memory
void free_gmlp(GMLP* gmlp) {
    // Free weights
    free(gmlp->proj_in_weight);
    free(gmlp->sgu_gate_weight);
    free(gmlp->sgu_proj_weight);
    free(gmlp->proj_out_weight);
    
    // Free gradients
    free(gmlp->proj_in_weight_grad);
    free(gmlp->sgu_gate_weight_grad);
    free(gmlp->sgu_proj_weight_grad);
    free(gmlp->proj_out_weight_grad);
    
    // Free Adam buffers
    free(gmlp->proj_in_m);
    free(gmlp->proj_in_v);
    free(gmlp->sgu_gate_m);
    free(gmlp->sgu_gate_v);
    free(gmlp->sgu_proj_m);
    free(gmlp->sgu_proj_v);
    free(gmlp->proj_out_m);
    free(gmlp->proj_out_v);
    
    // Free intermediate activations
    free(gmlp->proj_in_output);
    free(gmlp->sgu_gate_output);
    free(gmlp->gate_activated);
    free(gmlp->sgu_proj_input);
    free(gmlp->sgu_output);
    free(gmlp->predictions);
    
    // Free intermediate gradients
    free(gmlp->error);
    free(gmlp->sgu_output_grad);
    free(gmlp->sgu_proj_grad);
    free(gmlp->gate_activated_grad);
    free(gmlp->sgu_gate_grad);
    free(gmlp->proj_in_grad);
    
    // Free device memory
    cudaFree(gmlp->d_proj_in_weight);
    cudaFree(gmlp->d_sgu_gate_weight);
    cudaFree(gmlp->d_sgu_proj_weight);
    cudaFree(gmlp->d_proj_out_weight);
    
    cudaFree(gmlp->d_proj_in_weight_grad);
    cudaFree(gmlp->d_sgu_gate_weight_grad);
    cudaFree(gmlp->d_sgu_proj_weight_grad);
    cudaFree(gmlp->d_proj_out_weight_grad);
    
    cudaFree(gmlp->d_proj_in_m);
    cudaFree(gmlp->d_proj_in_v);
    cudaFree(gmlp->d_sgu_gate_m);
    cudaFree(gmlp->d_sgu_gate_v);
    cudaFree(gmlp->d_sgu_proj_m);
    cudaFree(gmlp->d_sgu_proj_v);
    cudaFree(gmlp->d_proj_out_m);
    cudaFree(gmlp->d_proj_out_v);
    
    cudaFree(gmlp->d_proj_in_output);
    cudaFree(gmlp->d_sgu_gate_output);
    cudaFree(gmlp->d_gate_activated);
    cudaFree(gmlp->d_sgu_proj_input);
    cudaFree(gmlp->d_sgu_output);
    cudaFree(gmlp->d_predictions);
    
    cudaFree(gmlp->d_error);
    cudaFree(gmlp->d_sgu_output_grad);
    cudaFree(gmlp->d_sgu_proj_grad);
    cudaFree(gmlp->d_gate_activated_grad);
    cudaFree(gmlp->d_sgu_gate_grad);
    cudaFree(gmlp->d_proj_in_grad);
    
    cudaFree(gmlp->d_X);
    cudaFree(gmlp->d_y);
    
    // Destroy cuBLAS handle
    cublasDestroy(gmlp->cublas_handle);
    
    // Free the struct
    free(gmlp);
}

// Forward pass
void forward_pass_gmlp(GMLP* gmlp, float* X) {
    // Copy input data to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_X, X, gmlp->batch_size * gmlp->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    int block_size = 256;
    int num_blocks;
    
    // 1. Input projection: X → hidden_dim
    // Note: cuBLAS uses column-major format, but we work with row-major data
    // So we compute: (gmlp->proj_in_weight)^T * X^T = (X * gmlp->proj_in_weight)^T
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            gmlp->hidden_dim, gmlp->batch_size, gmlp->input_dim,
                            &alpha,
                            gmlp->d_proj_in_weight, gmlp->input_dim,
                            gmlp->d_X, gmlp->input_dim,
                            &beta,
                            gmlp->d_proj_in_output, gmlp->hidden_dim));
    
    // Store a copy before applying GELU
    float* d_pre_activation;
    CHECK_CUDA(cudaMalloc(&d_pre_activation, gmlp->batch_size * gmlp->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_pre_activation, gmlp->d_proj_in_output, 
                         gmlp->batch_size * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToDevice));
    
    // Apply GELU activation to the input projection
    num_blocks = (gmlp->batch_size * gmlp->hidden_dim + block_size - 1) / block_size;
    gelu_kernel<<<num_blocks, block_size>>>(
        gmlp->d_proj_in_output,
        d_pre_activation,
        gmlp->batch_size * gmlp->hidden_dim
    );
    
    // 2. Spatial Gating Unit (SGU)
    // 2a. Compute gate values
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            gmlp->ffn_dim, gmlp->batch_size, gmlp->hidden_dim,
                            &alpha,
                            gmlp->d_sgu_gate_weight, gmlp->hidden_dim,
                            gmlp->d_proj_in_output, gmlp->hidden_dim,
                            &beta,
                            gmlp->d_sgu_gate_output, gmlp->ffn_dim));
    
    // Apply sigmoid activation to gate values
    num_blocks = (gmlp->batch_size * gmlp->ffn_dim + block_size - 1) / block_size;
    sigmoid_kernel<<<num_blocks, block_size>>>(
        gmlp->d_gate_activated,
        gmlp->d_sgu_gate_output,
        gmlp->batch_size * gmlp->ffn_dim
    );
    
    // 2b. Project to FFN dimension (for element-wise multiplication with gate)
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            gmlp->ffn_dim, gmlp->batch_size, gmlp->hidden_dim,
                            &alpha,
                            gmlp->d_sgu_gate_weight, gmlp->hidden_dim,
                            gmlp->d_proj_in_output, gmlp->hidden_dim,
                            &beta,
                            gmlp->d_sgu_proj_input, gmlp->ffn_dim));
    
    // 2c. Apply gating (element-wise multiplication)
    element_wise_multiply_kernel<<<num_blocks, block_size>>>(
        gmlp->d_sgu_proj_input,
        gmlp->d_gate_activated,
        gmlp->batch_size * gmlp->ffn_dim
    );
    
    // 2d. Project back to hidden dimension
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            gmlp->hidden_dim, gmlp->batch_size, gmlp->ffn_dim,
                            &alpha,
                            gmlp->d_sgu_proj_weight, gmlp->ffn_dim,
                            gmlp->d_sgu_proj_input, gmlp->ffn_dim,
                            &beta,
                            gmlp->d_sgu_output, gmlp->hidden_dim));
    
    // 3. Output projection: hidden_dim → output_dim
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            gmlp->output_dim, gmlp->batch_size, gmlp->hidden_dim,
                            &alpha,
                            gmlp->d_proj_out_weight, gmlp->hidden_dim,
                            gmlp->d_sgu_output, gmlp->hidden_dim,
                            &beta,
                            gmlp->d_predictions, gmlp->output_dim));
    
    // Copy predictions back to host
    CHECK_CUDA(cudaMemcpy(gmlp->predictions, gmlp->d_predictions,
                         gmlp->batch_size * gmlp->output_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    // Free temporary memory
    cudaFree(d_pre_activation);
}

// Calculate loss (Mean Squared Error)
float calculate_loss_gmlp(GMLP* gmlp, float* y) {
    // Copy target to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_y, y, gmlp->batch_size * gmlp->output_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Calculate error on device
    int block_size = 256;
    int num_blocks = (gmlp->batch_size * gmlp->output_dim + block_size - 1) / block_size;
    calc_error_kernel<<<num_blocks, block_size>>>(
        gmlp->d_error,
        gmlp->d_predictions,
        gmlp->d_y,
        gmlp->batch_size * gmlp->output_dim
    );
    
    // Copy error back to host for loss calculation
    CHECK_CUDA(cudaMemcpy(gmlp->error, gmlp->d_error,
                         gmlp->batch_size * gmlp->output_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    float loss = 0.0f;
    for (int i = 0; i < gmlp->batch_size * gmlp->output_dim; i++) {
        loss += gmlp->error[i] * gmlp->error[i];
    }
    return loss / (gmlp->batch_size * gmlp->output_dim);
}

// Zero out all gradients
void zero_gradients_gmlp(GMLP* gmlp) {
    CHECK_CUDA(cudaMemset(gmlp->d_proj_in_weight_grad, 0, 
                         gmlp->hidden_dim * gmlp->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_gate_weight_grad, 0, 
                         gmlp->ffn_dim * gmlp->hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_proj_weight_grad, 0, 
                         gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_proj_out_weight_grad, 0, 
                         gmlp->output_dim * gmlp->hidden_dim * sizeof(float)));
}

// Backward pass
void backward_pass_gmlp(GMLP* gmlp, float* X) {
    // Ensure input data is on device
    CHECK_CUDA(cudaMemcpy(gmlp->d_X, X, gmlp->batch_size * gmlp->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Need to recompute the forward pass for backwards
    float* d_pre_activation;
    CHECK_CUDA(cudaMalloc(&d_pre_activation, gmlp->batch_size * gmlp->hidden_dim * sizeof(float)));
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const float add_beta = 1.0f; // For accumulation
    
    // 1. Gradient of output projection
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            gmlp->hidden_dim, gmlp->output_dim, gmlp->batch_size,
                            &alpha,
                            gmlp->d_sgu_output, gmlp->hidden_dim,
                            gmlp->d_error, gmlp->output_dim,
                            &beta,
                            gmlp->d_proj_out_weight_grad, gmlp->output_dim));
    
    // 2. Gradient flowing back to SGU output
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            gmlp->hidden_dim, gmlp->batch_size, gmlp->output_dim,
                            &alpha,
                            gmlp->d_proj_out_weight, gmlp->hidden_dim,
                            gmlp->d_error, gmlp->output_dim,
                            &beta,
                            gmlp->d_sgu_output_grad, gmlp->hidden_dim));
    
    // 3. Gradient of SGU projection (from hidden_dim back to ffn_dim)
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            gmlp->ffn_dim, gmlp->hidden_dim, gmlp->batch_size,
                            &alpha,
                            gmlp->d_sgu_proj_input, gmlp->ffn_dim,
                            gmlp->d_sgu_output_grad, gmlp->hidden_dim,
                            &beta,
                            gmlp->d_sgu_proj_weight_grad, gmlp->hidden_dim));
    
    // 4. Gradient flowing through SGU projection to gated input
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            gmlp->ffn_dim, gmlp->batch_size, gmlp->hidden_dim,
                            &alpha,
                            gmlp->d_sgu_proj_weight, gmlp->ffn_dim,
                            gmlp->d_sgu_output_grad, gmlp->hidden_dim,
                            &beta,
                            gmlp->d_sgu_proj_grad, gmlp->ffn_dim));
    
    // 5. Gradient through the gating mechanism
    // 5a. Gradient to gate activation
    int block_size = 256;
    int num_blocks = (gmlp->batch_size * gmlp->ffn_dim + block_size - 1) / block_size;
    gate_activation_grad_kernel<<<num_blocks, block_size>>>(
        gmlp->d_gate_activated_grad,
        gmlp->d_sgu_proj_grad,
        gmlp->d_sgu_proj_input,
        gmlp->d_gate_activated,
        gmlp->batch_size * gmlp->ffn_dim
    );
    
    // 5b. Gradient through sigmoid
    sigmoid_derivative_kernel<<<num_blocks, block_size>>>(
        gmlp->d_sgu_gate_grad,
        gmlp->d_gate_activated,
        gmlp->d_gate_activated_grad,
        gmlp->batch_size * gmlp->ffn_dim
    );
    
    // 6. Gradient to gate weights
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            gmlp->hidden_dim, gmlp->ffn_dim, gmlp->batch_size,
                            &alpha,
                            gmlp->d_proj_in_output, gmlp->hidden_dim,
                            gmlp->d_sgu_gate_grad, gmlp->ffn_dim,
                            &beta,
                            gmlp->d_sgu_gate_weight_grad, gmlp->ffn_dim));
    
    // 7. Gradient flowing back to the input projection
    // 7a. From the gate
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            gmlp->hidden_dim, gmlp->batch_size, gmlp->ffn_dim,
                            &alpha,
                            gmlp->d_sgu_gate_weight, gmlp->hidden_dim,
                            gmlp->d_sgu_gate_grad, gmlp->ffn_dim,
                            &beta,
                            gmlp->d_proj_in_grad, gmlp->hidden_dim));
    
    // 7b. Also add gradient from the projection input
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            gmlp->hidden_dim, gmlp->batch_size, gmlp->ffn_dim,
                            &alpha,
                            gmlp->d_sgu_gate_weight, gmlp->hidden_dim,
                            gmlp->d_sgu_proj_grad, gmlp->ffn_dim,
                            &add_beta, // Add to existing gradient
                            gmlp->d_proj_in_grad, gmlp->hidden_dim));
    
    // 8. Compute pre-activation values for GELU gradient
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_T, CUBLAS_OP_T,
                            gmlp->hidden_dim, gmlp->batch_size, gmlp->input_dim,
                            &alpha,
                            gmlp->d_proj_in_weight, gmlp->input_dim,
                            gmlp->d_X, gmlp->input_dim,
                            &beta,
                            d_pre_activation, gmlp->hidden_dim));
    
    // Gradient through GELU activation
    num_blocks = (gmlp->batch_size * gmlp->hidden_dim + block_size - 1) / block_size;
    gelu_derivative_kernel<<<num_blocks, block_size>>>(
        gmlp->d_proj_in_grad,
        d_pre_activation,
        gmlp->d_proj_in_grad,
        gmlp->batch_size * gmlp->hidden_dim
    );
    
    // 9. Gradient to input projection weights
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle,
                            CUBLAS_OP_N, CUBLAS_OP_T,
                            gmlp->input_dim, gmlp->hidden_dim, gmlp->batch_size,
                            &alpha,
                            gmlp->d_X, gmlp->input_dim,
                            gmlp->d_proj_in_grad, gmlp->hidden_dim,
                            &beta,
                            gmlp->d_proj_in_weight_grad, gmlp->hidden_dim));
    
    // Clean up
    cudaFree(d_pre_activation);
    
    // Copy gradients back to host
    CHECK_CUDA(cudaMemcpy(gmlp->proj_in_weight_grad, gmlp->d_proj_in_weight_grad,
                         gmlp->hidden_dim * gmlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->sgu_gate_weight_grad, gmlp->d_sgu_gate_weight_grad,
                         gmlp->ffn_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->sgu_proj_weight_grad, gmlp->d_sgu_proj_weight_grad,
                         gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->proj_out_weight_grad, gmlp->d_proj_out_weight_grad,
                         gmlp->output_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

// Update weights using AdamW
void update_weights_gmlp(GMLP* gmlp, float learning_rate) {
    gmlp->t++;  // Increment time step
    
    float beta1_t = powf(gmlp->beta1, gmlp->t);
    float beta2_t = powf(gmlp->beta2, gmlp->t);
    float alpha_t = learning_rate * sqrtf(1.0f - beta2_t) / (1.0f - beta1_t);
    
    // Copy updated Adam states to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_m, gmlp->proj_in_m,
                         gmlp->hidden_dim * gmlp->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_v, gmlp->proj_in_v,
                         gmlp->hidden_dim * gmlp->input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_m, gmlp->sgu_gate_m,
                         gmlp->ffn_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_v, gmlp->sgu_gate_v,
                         gmlp->ffn_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_m, gmlp->sgu_proj_m,
                         gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_v, gmlp->sgu_proj_v,
                         gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_m, gmlp->proj_out_m,
                         gmlp->output_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_v, gmlp->proj_out_v,
                         gmlp->output_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int num_blocks;
    
    // Update projection in weights
    num_blocks = (gmlp->hidden_dim * gmlp->input_dim + block_size - 1) / block_size;
    adamw_update_kernel<<<num_blocks, block_size>>>(
        gmlp->d_proj_in_weight,
        gmlp->d_proj_in_weight_grad,
        gmlp->d_proj_in_m,
        gmlp->d_proj_in_v,
        gmlp->beta1,
        gmlp->beta2,
        gmlp->epsilon,
        learning_rate,
        gmlp->weight_decay,
        alpha_t,
        gmlp->batch_size,
        gmlp->hidden_dim * gmlp->input_dim
    );
    
    // Update SGU gate weights
    num_blocks = (gmlp->ffn_dim * gmlp->hidden_dim + block_size - 1) / block_size;
    adamw_update_kernel<<<num_blocks, block_size>>>(
        gmlp->d_sgu_gate_weight,
        gmlp->d_sgu_gate_weight_grad,
        gmlp->d_sgu_gate_m,
        gmlp->d_sgu_gate_v,
        gmlp->beta1,
        gmlp->beta2,
        gmlp->epsilon,
        learning_rate,
        gmlp->weight_decay,
        alpha_t,
        gmlp->batch_size,
        gmlp->ffn_dim * gmlp->hidden_dim
    );
    
    // Update SGU projection weights
    num_blocks = (gmlp->hidden_dim * gmlp->ffn_dim + block_size - 1) / block_size;
    adamw_update_kernel<<<num_blocks, block_size>>>(
        gmlp->d_sgu_proj_weight,
        gmlp->d_sgu_proj_weight_grad,
        gmlp->d_sgu_proj_m,
        gmlp->d_sgu_proj_v,
        gmlp->beta1,
        gmlp->beta2,
        gmlp->epsilon,
        learning_rate,
        gmlp->weight_decay,
        alpha_t,
        gmlp->batch_size,
        gmlp->hidden_dim * gmlp->ffn_dim
    );
    
    // Update projection out weights
    num_blocks = (gmlp->output_dim * gmlp->hidden_dim + block_size - 1) / block_size;
    adamw_update_kernel<<<num_blocks, block_size>>>(
        gmlp->d_proj_out_weight,
        gmlp->d_proj_out_weight_grad,
        gmlp->d_proj_out_m,
        gmlp->d_proj_out_v,
        gmlp->beta1,
        gmlp->beta2,
        gmlp->epsilon,
        learning_rate,
        gmlp->weight_decay,
        alpha_t,
        gmlp->batch_size,
        gmlp->output_dim * gmlp->hidden_dim
    );
    
    // Copy updated weights and Adam states back to host
    CHECK_CUDA(cudaMemcpy(gmlp->proj_in_weight, gmlp->d_proj_in_weight,
                         gmlp->hidden_dim * gmlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->sgu_gate_weight, gmlp->d_sgu_gate_weight,
                         gmlp->ffn_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->sgu_proj_weight, gmlp->d_sgu_proj_weight,
                         gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->proj_out_weight, gmlp->d_proj_out_weight,
                         gmlp->output_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaMemcpy(gmlp->proj_in_m, gmlp->d_proj_in_m,
                         gmlp->hidden_dim * gmlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->proj_in_v, gmlp->d_proj_in_v,
                         gmlp->hidden_dim * gmlp->input_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->sgu_gate_m, gmlp->d_sgu_gate_m,
                         gmlp->ffn_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->sgu_gate_v, gmlp->d_sgu_gate_v,
                         gmlp->ffn_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->sgu_proj_m, gmlp->d_sgu_proj_m,
                         gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->sgu_proj_v, gmlp->d_sgu_proj_v,
                         gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->proj_out_m, gmlp->d_proj_out_m,
                         gmlp->output_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->proj_out_v, gmlp->d_proj_out_v,
                         gmlp->output_dim * gmlp->hidden_dim * sizeof(float),
                         cudaMemcpyDeviceToHost));
}

// Save model weights to binary file
void save_gmlp(GMLP* gmlp, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    
    // Save dimensions
    fwrite(&gmlp->input_dim, sizeof(int), 1, file);
    fwrite(&gmlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&gmlp->ffn_dim, sizeof(int), 1, file);
    fwrite(&gmlp->output_dim, sizeof(int), 1, file);
    fwrite(&gmlp->batch_size, sizeof(int), 1, file);
    
    // Save weights
    fwrite(gmlp->proj_in_weight, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(gmlp->sgu_gate_weight, sizeof(float), gmlp->ffn_dim * gmlp->hidden_dim, file);
    fwrite(gmlp->sgu_proj_weight, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(gmlp->proj_out_weight, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    
    // Save Adam state
    fwrite(&gmlp->t, sizeof(int), 1, file);
    fwrite(gmlp->proj_in_m, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(gmlp->proj_in_v, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(gmlp->sgu_gate_m, sizeof(float), gmlp->ffn_dim * gmlp->hidden_dim, file);
    fwrite(gmlp->sgu_gate_v, sizeof(float), gmlp->ffn_dim * gmlp->hidden_dim, file);
    fwrite(gmlp->sgu_proj_m, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(gmlp->sgu_proj_v, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(gmlp->proj_out_m, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    fwrite(gmlp->proj_out_v, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// Load model weights from binary file
GMLP* load_gmlp(const char* filename, int custom_batch_size) {
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
    GMLP* gmlp = init_gmlp(input_dim, hidden_dim, ffn_dim, output_dim, batch_size);
    
    // Load weights
    fread(gmlp->proj_in_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(gmlp->sgu_gate_weight, sizeof(float), ffn_dim * hidden_dim, file);
    fread(gmlp->sgu_proj_weight, sizeof(float), hidden_dim * ffn_dim, file);
    fread(gmlp->proj_out_weight, sizeof(float), output_dim * hidden_dim, file);
    
    // Load Adam state
    fread(&gmlp->t, sizeof(int), 1, file);
    fread(gmlp->proj_in_m, sizeof(float), hidden_dim * input_dim, file);
    fread(gmlp->proj_in_v, sizeof(float), hidden_dim * input_dim, file);
    fread(gmlp->sgu_gate_m, sizeof(float), ffn_dim * hidden_dim, file);
    fread(gmlp->sgu_gate_v, sizeof(float), ffn_dim * hidden_dim, file);
    fread(gmlp->sgu_proj_m, sizeof(float), hidden_dim * ffn_dim, file);
    fread(gmlp->sgu_proj_v, sizeof(float), hidden_dim * ffn_dim, file);
    fread(gmlp->proj_out_m, sizeof(float), output_dim * hidden_dim, file);
    fread(gmlp->proj_out_v, sizeof(float), output_dim * hidden_dim, file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_weight, gmlp->proj_in_weight,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_weight, gmlp->sgu_gate_weight,
                         ffn_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_weight, gmlp->sgu_proj_weight,
                         hidden_dim * ffn_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_weight, gmlp->proj_out_weight,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_m, gmlp->proj_in_m,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_v, gmlp->proj_in_v,
                         hidden_dim * input_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_m, gmlp->sgu_gate_m,
                         ffn_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_v, gmlp->sgu_gate_v,
                         ffn_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_m, gmlp->sgu_proj_m,
                         hidden_dim * ffn_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_v, gmlp->sgu_proj_v,
                         hidden_dim * ffn_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_m, gmlp->proj_out_m,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_v, gmlp->proj_out_v,
                         output_dim * hidden_dim * sizeof(float),
                         cudaMemcpyHostToDevice));
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return gmlp;
}

#endif