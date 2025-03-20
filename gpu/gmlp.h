#ifndef GMLP_H
#define GMLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------
// Error checking macros for CUDA and cuBLAS calls
// ---------------------------------------------------------------------
#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error in %s at line %d: %d\n", \
                __FILE__, __LINE__, status); \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// ---------------------------------------------------------------------
// Structure definition for the gMLP model
// This version uses cuBLAS and integrates the AdamW optimizer.
// ---------------------------------------------------------------------
typedef struct {
    // Network dimensions
    int input_dim;
    int hidden_dim;
    int ffn_dim;     // Dimension of feed-forward network
    int output_dim;
    int batch_size;
    
    // Weight matrices (stored on device)
    float* d_proj_in_weight;    // hidden_dim x input_dim
    float* d_sgu_gate_weight;   // ffn_dim x (hidden_dim/2)
    float* d_sgu_proj_weight;   // ffn_dim x (hidden_dim/2)
    float* d_sgu_out_weight;    // hidden_dim x ffn_dim
    float* d_proj_out_weight;   // output_dim x hidden_dim
    
    // Host copies for saving/loading the model
    float* h_proj_in_weight;
    float* h_sgu_gate_weight;
    float* h_sgu_proj_weight;
    float* h_sgu_out_weight;
    float* h_proj_out_weight;
    
    // Gradients (device pointers)
    float* d_proj_in_weight_grad;
    float* d_sgu_gate_weight_grad;
    float* d_sgu_proj_weight_grad;
    float* d_sgu_out_weight_grad;
    float* d_proj_out_weight_grad;
    
    // Adam optimizer first (m) and second (v) moment estimates (device pointers)
    float* d_proj_in_m;
    float* d_proj_in_v;
    float* d_sgu_gate_m;
    float* d_sgu_gate_v;
    float* d_sgu_proj_m;
    float* d_sgu_proj_v;
    float* d_sgu_out_m;
    float* d_sgu_out_v;
    float* d_proj_out_m;
    float* d_proj_out_v;
    
    // Adam hyperparameters and counter
    float beta1;         // e.g., 0.9
    float beta2;         // e.g., 0.999
    float epsilon;       // e.g., 1e-8
    float weight_decay;  // e.g., 0.01
    int adam_t;          // time step counter
    
    // Intermediate activations for forward/backward pass
    float* d_proj_in_output;  // batch_size x hidden_dim
    float* d_u_part;          // batch_size x (hidden_dim/2) - First half for gate
    float* d_v_part;          // batch_size x (hidden_dim/2) - Second half for projection
    float* d_sgu_gate_output; // batch_size x ffn_dim
    float* d_gate_activated;  // batch_size x ffn_dim
    float* d_sgu_proj_output; // batch_size x ffn_dim
    float* d_gated_output;    // batch_size x ffn_dim - After applying gate
    float* d_sgu_output;      // batch_size x hidden_dim
    float* d_predictions;     // batch_size x output_dim
    
    // Intermediates for backward pass
    float* d_error;              // batch_size x output_dim
    float* d_sgu_output_grad;    // batch_size x hidden_dim
    float* d_gated_output_grad;  // batch_size x ffn_dim
    float* d_sgu_proj_grad;      // batch_size x ffn_dim
    float* d_gate_activated_grad;// batch_size x ffn_dim
    float* d_sgu_gate_grad;      // batch_size x ffn_dim
    float* d_u_part_grad;        // batch_size x (hidden_dim/2)
    float* d_v_part_grad;        // batch_size x (hidden_dim/2)
    float* d_proj_in_grad;       // batch_size x hidden_dim
    
    // CUDA library handles
    cublasHandle_t cublas_handle;
    
    // Host predictions for evaluation
    float* h_predictions;         // batch_size x output_dim
} GMLP;

// ---------------------------------------------------------------------
// CUDA kernel: GELU activation forward pass
// GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------
__global__ void gelu_activation_kernel_gmlp(float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        input[idx] = x * 0.5f * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: GELU activation backward pass
// ---------------------------------------------------------------------
__global__ void gelu_backward_kernel_gmlp(float* grad_out, float* input, float* output, int size) {
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

// ---------------------------------------------------------------------
// CUDA kernel: Sigmoid activation forward pass
// sigmoid(x) = 1 / (1 + exp(-x))
// ---------------------------------------------------------------------
__global__ void sigmoid_activation_kernel_gmlp(float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f / (1.0f + expf(-input[idx]));
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Element-wise multiplication
// ---------------------------------------------------------------------
__global__ void multiply_elements_kernel_gmlp(float* a, float* b, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] * b[idx];
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Mean Squared Error loss computation (elementwise error)
// ---------------------------------------------------------------------
__global__ void mse_loss_kernel_gmlp(float* error, float* predictions, float* targets, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        error[idx] = predictions[idx] - targets[idx];
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: Sigmoid backward pass
// ---------------------------------------------------------------------
__global__ void sigmoid_backward_kernel_gmlp(float* grad_out, float* sigmoid_output, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sigmoid = sigmoid_output[idx];
        output[idx] = grad_out[idx] * sigmoid * (1.0f - sigmoid);
    }
}

// ---------------------------------------------------------------------
// CUDA kernel: AdamW update (per weight element)
// ---------------------------------------------------------------------
__global__ void adamw_update_kernel_gmlp(float* W, const float* grad, float* m, float* v, 
                                  int size, float beta1, float beta2, float epsilon, 
                                  float weight_decay, float learning_rate, int batch_size, 
                                  float bias_correction1, float bias_correction2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / ((float) batch_size);
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
        float m_hat = m[idx] / bias_correction1;
        float v_hat = v[idx] / bias_correction2;
        W[idx] = W[idx] * (1.0f - learning_rate * weight_decay) - learning_rate * (m_hat / (sqrtf(v_hat) + epsilon));
    }
}

// ---------------------------------------------------------------------
// Function: init_gmlp
// Initializes the GMLP structure, allocates host and device memory,
// sets initial weights with scaled random values, and copies them to device.
// Also initializes Adam optimizer parameters.
// ---------------------------------------------------------------------
GMLP* init_gmlp(int input_dim, int hidden_dim, int ffn_dim, int output_dim, int batch_size) {
    GMLP* gmlp = (GMLP*)malloc(sizeof(GMLP));
    if (!gmlp) {
        fprintf(stderr, "Failed to allocate memory for GMLP\n");
        return NULL;
    }
    
    // Store dimensions
    gmlp->input_dim = input_dim;
    gmlp->hidden_dim = hidden_dim;
    gmlp->ffn_dim = ffn_dim;
    gmlp->output_dim = output_dim;
    gmlp->batch_size = batch_size;
    
    // Set Adam hyperparameters
    gmlp->beta1 = 0.9f;
    gmlp->beta2 = 0.999f;
    gmlp->epsilon = 1e-8f;
    gmlp->weight_decay = 0.01f;
    gmlp->adam_t = 0;
    
    // Calculate half hidden dimension for SGU
    int half_hidden = hidden_dim / 2;
    
    // Create cuBLAS handle
    CHECK_CUBLAS(cublasCreate(&gmlp->cublas_handle));
    
    // Allocate host memory for weight matrices
    gmlp->h_proj_in_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    gmlp->h_sgu_gate_weight = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    gmlp->h_sgu_proj_weight = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    gmlp->h_sgu_out_weight = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    gmlp->h_proj_out_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate host memory for predictions
    gmlp->h_predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Initialize matrices with He initialization
    float scale_in = sqrtf(2.0f / input_dim);
    float scale_half_hidden = sqrtf(2.0f / half_hidden);
    float scale_ffn = sqrtf(2.0f / ffn_dim);
    float scale_hidden = sqrtf(2.0f / hidden_dim);
    
    for (int i = 0; i < hidden_dim * input_dim; i++) {
        gmlp->h_proj_in_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_in;
    }
    
    for (int i = 0; i < ffn_dim * half_hidden; i++) {
        gmlp->h_sgu_gate_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_half_hidden;
        gmlp->h_sgu_proj_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_half_hidden;
    }
    
    for (int i = 0; i < hidden_dim * ffn_dim; i++) {
        gmlp->h_sgu_out_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_ffn;
    }
    
    for (int i = 0; i < output_dim * hidden_dim; i++) {
        gmlp->h_proj_out_weight[i] = ((float)rand() / (float)RAND_MAX * 2.0f - 1.0f) * scale_hidden;
    }
    
    // Allocate device memory for weight matrices
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_weight, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_weight, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_weight, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_out_weight, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_out_weight, output_dim * hidden_dim * sizeof(float)));
    
    // Allocate device memory for gradients
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_weight_grad, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_weight_grad, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_weight_grad, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_out_weight_grad, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_out_weight_grad, output_dim * hidden_dim * sizeof(float)));
    
    // Allocate device memory for Adam first and second moment estimates and initialize to zero
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_m, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_v, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_m, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_v, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_m, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_v, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_out_m, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_out_v, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_out_m, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_out_v, output_dim * hidden_dim * sizeof(float)));
    
    CHECK_CUDA(cudaMemset(gmlp->d_proj_in_m, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_proj_in_v, 0, hidden_dim * input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_gate_m, 0, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_gate_v, 0, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_proj_m, 0, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_proj_v, 0, ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_out_m, 0, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_out_v, 0, hidden_dim * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_proj_out_m, 0, output_dim * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_proj_out_v, 0, output_dim * hidden_dim * sizeof(float)));
    
    // Allocate intermediate activations on device
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_output, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_gate_activated, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_output, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_gated_output, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_output, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_predictions, batch_size * output_dim * sizeof(float)));
    
    // Allocate intermediate gradients on device
    CHECK_CUDA(cudaMalloc(&gmlp->d_error, batch_size * output_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_output_grad, batch_size * hidden_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_gated_output_grad, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_proj_grad, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_gate_activated_grad, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_sgu_gate_grad, batch_size * ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_u_part_grad, batch_size * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_v_part_grad, batch_size * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&gmlp->d_proj_in_grad, batch_size * hidden_dim * sizeof(float)));
    
    // Set up u_part and v_part pointers as views into the hidden state
    gmlp->d_u_part = gmlp->d_proj_in_output;
    gmlp->d_v_part = gmlp->d_proj_in_output + batch_size * half_hidden;
    
    // Copy weight matrices from host to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_weight, gmlp->h_proj_in_weight, 
                     hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_weight, gmlp->h_sgu_gate_weight, 
                     ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_weight, gmlp->h_sgu_proj_weight, 
                     ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_out_weight, gmlp->h_sgu_out_weight, 
                     hidden_dim * ffn_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_weight, gmlp->h_proj_out_weight, 
                     output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    return gmlp;
}

// ---------------------------------------------------------------------
// Function: forward_pass_gmlp
// Computes the forward pass:
//   1. Input projection: X → hidden_dim with GELU activation
//   2. Split hidden dim into u and v parts
//   3. Apply SGU: gated_output = sigmoid(gate(u)) * proj(v)
//   4. Project back to hidden_dim
//   5. Project to output_dim
// ---------------------------------------------------------------------
void forward_pass_gmlp(GMLP* gmlp, float* d_X) {
    const float alpha = 1.0f, beta = 0.0f;
    int half_hidden = gmlp->hidden_dim / 2;
    
    // 1. Input projection: X → hidden_dim
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           gmlp->hidden_dim, gmlp->batch_size, gmlp->input_dim,
                           &alpha,
                           gmlp->d_proj_in_weight, gmlp->hidden_dim,
                           d_X, gmlp->input_dim,
                           &beta,
                           gmlp->d_proj_in_output, gmlp->hidden_dim));
    
    // Apply GELU activation to the input projection
    int block_size = 256;
    int num_blocks = (gmlp->batch_size * gmlp->hidden_dim + block_size - 1) / block_size;
    gelu_activation_kernel_gmlp<<<num_blocks, block_size>>>(gmlp->d_proj_in_output, 
                                                      gmlp->batch_size * gmlp->hidden_dim);
    CHECK_CUDA(cudaGetLastError());
    
    // 2. Spatial Gating Unit (SGU)
    // 2a. Compute gate values from first half of hidden states (u)
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           gmlp->ffn_dim, gmlp->batch_size, half_hidden,
                           &alpha,
                           gmlp->d_sgu_gate_weight, gmlp->ffn_dim,
                           gmlp->d_u_part, half_hidden,
                           &beta,
                           gmlp->d_sgu_gate_output, gmlp->ffn_dim));
    
    // Apply sigmoid activation to gate values
    num_blocks = (gmlp->batch_size * gmlp->ffn_dim + block_size - 1) / block_size;
    sigmoid_activation_kernel_gmlp<<<num_blocks, block_size>>>(gmlp->d_sgu_gate_output, gmlp->d_gate_activated, 
                                                         gmlp->batch_size * gmlp->ffn_dim);
    CHECK_CUDA(cudaGetLastError());
    
    // 2b. Project second half of hidden states (v) to FFN dimension
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           gmlp->ffn_dim, gmlp->batch_size, half_hidden,
                           &alpha,
                           gmlp->d_sgu_proj_weight, gmlp->ffn_dim,
                           gmlp->d_v_part, half_hidden,
                           &beta,
                           gmlp->d_sgu_proj_output, gmlp->ffn_dim));
    
    // 2c. Apply gating (element-wise multiplication)
    num_blocks = (gmlp->batch_size * gmlp->ffn_dim + block_size - 1) / block_size;
    multiply_elements_kernel_gmlp<<<num_blocks, block_size>>>(gmlp->d_sgu_proj_output, gmlp->d_gate_activated, 
                                                        gmlp->d_gated_output, gmlp->batch_size * gmlp->ffn_dim);
    CHECK_CUDA(cudaGetLastError());
    
    // 2d. Project back to hidden dimension
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           gmlp->hidden_dim, gmlp->batch_size, gmlp->ffn_dim,
                           &alpha,
                           gmlp->d_sgu_out_weight, gmlp->hidden_dim,
                           gmlp->d_gated_output, gmlp->ffn_dim,
                           &beta,
                           gmlp->d_sgu_output, gmlp->hidden_dim));
    
    // 3. Output projection: hidden_dim → output_dim
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           gmlp->output_dim, gmlp->batch_size, gmlp->hidden_dim,
                           &alpha,
                           gmlp->d_proj_out_weight, gmlp->output_dim,
                           gmlp->d_sgu_output, gmlp->hidden_dim,
                           &beta,
                           gmlp->d_predictions, gmlp->output_dim));
}

// ---------------------------------------------------------------------
// Function: calculate_loss_gmlp
// Computes the Mean Squared Error loss between predictions and targets.
// ---------------------------------------------------------------------
float calculate_loss_gmlp(GMLP* gmlp, float* d_y) {
    // Compute error: error = predictions - targets
    int size = gmlp->batch_size * gmlp->output_dim;
    int block_size = 256;
    int num_blocks = (size + block_size - 1) / block_size;
    mse_loss_kernel_gmlp<<<num_blocks, block_size>>>(gmlp->d_error, gmlp->d_predictions, d_y, size);
    CHECK_CUDA(cudaGetLastError());
    
    // Compute MSE loss
    float loss = 0.0f;
    CHECK_CUBLAS(cublasSdot(gmlp->cublas_handle, size,
                          gmlp->d_error, 1,
                          gmlp->d_error, 1,
                          &loss));
    
    return loss / size;
}

// ---------------------------------------------------------------------
// Function: zero_gradients_gmlp
// Clears the gradient arrays on the device.
// ---------------------------------------------------------------------
void zero_gradients_gmlp(GMLP* gmlp) {
    int half_hidden = gmlp->hidden_dim / 2;
    
    CHECK_CUDA(cudaMemset(gmlp->d_proj_in_weight_grad, 0, 
                     gmlp->hidden_dim * gmlp->input_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_gate_weight_grad, 0, 
                     gmlp->ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_proj_weight_grad, 0, 
                     gmlp->ffn_dim * half_hidden * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_sgu_out_weight_grad, 0, 
                     gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float)));
    CHECK_CUDA(cudaMemset(gmlp->d_proj_out_weight_grad, 0, 
                     gmlp->output_dim * gmlp->hidden_dim * sizeof(float)));
}

// ---------------------------------------------------------------------
// Function: backward_pass_gmlp
// Computes gradients through the network using the chain rule.
// ---------------------------------------------------------------------
void backward_pass_gmlp(GMLP* gmlp, float* d_X) {
    int half_hidden = gmlp->hidden_dim / 2;
    const float alpha = 1.0f, beta = 0.0f;
    
    // 1. Gradient of output projection: d_proj_out_weight_grad = error * (sgu_output)^T
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           gmlp->output_dim, gmlp->hidden_dim, gmlp->batch_size,
                           &alpha,
                           gmlp->d_error, gmlp->output_dim,
                           gmlp->d_sgu_output, gmlp->hidden_dim,
                           &beta,
                           gmlp->d_proj_out_weight_grad, gmlp->output_dim));
    
    // 2. Gradient flowing back to SGU output: d_sgu_output_grad = (proj_out_weight)^T * error
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           gmlp->hidden_dim, gmlp->batch_size, gmlp->output_dim,
                           &alpha,
                           gmlp->d_proj_out_weight, gmlp->output_dim,
                           gmlp->d_error, gmlp->output_dim,
                           &beta,
                           gmlp->d_sgu_output_grad, gmlp->hidden_dim));
    
    // 3. Gradient of SGU output weight: d_sgu_out_weight_grad = sgu_output_grad * (gated_output)^T
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           gmlp->hidden_dim, gmlp->ffn_dim, gmlp->batch_size,
                           &alpha,
                           gmlp->d_sgu_output_grad, gmlp->hidden_dim,
                           gmlp->d_gated_output, gmlp->ffn_dim,
                           &beta,
                           gmlp->d_sgu_out_weight_grad, gmlp->hidden_dim));
    
    // 4. Gradient flowing back to gated_output: d_gated_output_grad = (sgu_out_weight)^T * sgu_output_grad
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           gmlp->ffn_dim, gmlp->batch_size, gmlp->hidden_dim,
                           &alpha,
                           gmlp->d_sgu_out_weight, gmlp->hidden_dim,
                           gmlp->d_sgu_output_grad, gmlp->hidden_dim,
                           &beta,
                           gmlp->d_gated_output_grad, gmlp->ffn_dim));
    
    // 5. Gradient through the gating mechanism
    // 5a. Gradient to sgu_proj_output: d_sgu_proj_grad = d_gated_output_grad * gate_activated
    int block_size = 256;
    int num_blocks = (gmlp->batch_size * gmlp->ffn_dim + block_size - 1) / block_size;
    multiply_elements_kernel_gmlp<<<num_blocks, block_size>>>(gmlp->d_gated_output_grad, gmlp->d_gate_activated, 
                                                        gmlp->d_sgu_proj_grad, gmlp->batch_size * gmlp->ffn_dim);
    CHECK_CUDA(cudaGetLastError());
    
    // 5b. Gradient to gate_activated: d_gate_activated_grad = d_gated_output_grad * sgu_proj_output
    multiply_elements_kernel_gmlp<<<num_blocks, block_size>>>(gmlp->d_gated_output_grad, gmlp->d_sgu_proj_output, 
                                                        gmlp->d_gate_activated_grad, 
                                                        gmlp->batch_size * gmlp->ffn_dim);
    CHECK_CUDA(cudaGetLastError());
    
    // 5c. Gradient through sigmoid for gate: d_sgu_gate_grad = d_gate_activated_grad * sigmoid'
    sigmoid_backward_kernel_gmlp<<<num_blocks, block_size>>>(gmlp->d_gate_activated_grad, gmlp->d_gate_activated, 
                                                       gmlp->d_sgu_gate_grad, gmlp->batch_size * gmlp->ffn_dim);
    CHECK_CUDA(cudaGetLastError());
    
    // 6. Gradient of sgu_proj_weight: d_sgu_proj_weight_grad = d_sgu_proj_grad * (v_part)^T
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           gmlp->ffn_dim, half_hidden, gmlp->batch_size,
                           &alpha,
                           gmlp->d_sgu_proj_grad, gmlp->ffn_dim,
                           gmlp->d_v_part, half_hidden,
                           &beta,
                           gmlp->d_sgu_proj_weight_grad, gmlp->ffn_dim));
    
    // 7. Gradient of sgu_gate_weight: d_sgu_gate_weight_grad = d_sgu_gate_grad * (u_part)^T
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           gmlp->ffn_dim, half_hidden, gmlp->batch_size,
                           &alpha,
                           gmlp->d_sgu_gate_grad, gmlp->ffn_dim,
                           gmlp->d_u_part, half_hidden,
                           &beta,
                           gmlp->d_sgu_gate_weight_grad, gmlp->ffn_dim));
    
    // 8. Gradient flowing back to v_part: d_v_part_grad = (sgu_proj_weight)^T * d_sgu_proj_grad
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           half_hidden, gmlp->batch_size, gmlp->ffn_dim,
                           &alpha,
                           gmlp->d_sgu_proj_weight, gmlp->ffn_dim,
                           gmlp->d_sgu_proj_grad, gmlp->ffn_dim,
                           &beta,
                           gmlp->d_v_part_grad, half_hidden));
    
    // 9. Gradient flowing back to u_part: d_u_part_grad = (sgu_gate_weight)^T * d_sgu_gate_grad
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           half_hidden, gmlp->batch_size, gmlp->ffn_dim,
                           &alpha,
                           gmlp->d_sgu_gate_weight, gmlp->ffn_dim,
                           gmlp->d_sgu_gate_grad, gmlp->ffn_dim,
                           &beta,
                           gmlp->d_u_part_grad, half_hidden));
    
    // 10. Combine u_part_grad and v_part_grad to form proj_in_grad
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_grad, gmlp->d_u_part_grad, 
                     gmlp->batch_size * half_hidden * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_grad + gmlp->batch_size * half_hidden, gmlp->d_v_part_grad, 
                     gmlp->batch_size * half_hidden * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // 11. Gradient through GELU activation for proj_in_output
    num_blocks = (gmlp->batch_size * gmlp->hidden_dim + block_size - 1) / block_size;
    gelu_backward_kernel_gmlp<<<num_blocks, block_size>>>(gmlp->d_proj_in_grad, gmlp->d_proj_in_output, 
                                                   gmlp->d_proj_in_grad, gmlp->batch_size * gmlp->hidden_dim);
    CHECK_CUDA(cudaGetLastError());
    
    // 12. Gradient of input projection weights: d_proj_in_weight_grad = d_proj_in_grad * (input)^T
    CHECK_CUBLAS(cublasSgemm(gmlp->cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T,
                           gmlp->hidden_dim, gmlp->input_dim, gmlp->batch_size,
                           &alpha,
                           gmlp->d_proj_in_grad, gmlp->hidden_dim,
                           d_X, gmlp->input_dim,
                           &beta,
                           gmlp->d_proj_in_weight_grad, gmlp->hidden_dim));
}

// ---------------------------------------------------------------------
// Function: update_weights_gmlp
// Uses the AdamW optimizer to update weights
// ---------------------------------------------------------------------
void update_weights_gmlp(GMLP* gmlp, float learning_rate) {
    gmlp->adam_t++;  // Increment time step counter
    
    float bias_correction1 = 1.0f - powf(gmlp->beta1, (float)gmlp->adam_t);
    float bias_correction2 = 1.0f - powf(gmlp->beta2, (float)gmlp->adam_t);
    
    int half_hidden = gmlp->hidden_dim / 2;
    int block_size = 256;
    
    // Update proj_in_weight
    int size_proj_in = gmlp->hidden_dim * gmlp->input_dim;
    int num_blocks = (size_proj_in + block_size - 1) / block_size;
    adamw_update_kernel_gmlp<<<num_blocks, block_size>>>(
        gmlp->d_proj_in_weight, gmlp->d_proj_in_weight_grad, 
        gmlp->d_proj_in_m, gmlp->d_proj_in_v,
        size_proj_in, gmlp->beta1, gmlp->beta2, gmlp->epsilon, gmlp->weight_decay,
        learning_rate, gmlp->batch_size, bias_correction1, bias_correction2);
    CHECK_CUDA(cudaGetLastError());
    
    // Update sgu_gate_weight
    int size_sgu_gate = gmlp->ffn_dim * half_hidden;
    num_blocks = (size_sgu_gate + block_size - 1) / block_size;
    adamw_update_kernel_gmlp<<<num_blocks, block_size>>>(
        gmlp->d_sgu_gate_weight, gmlp->d_sgu_gate_weight_grad, 
        gmlp->d_sgu_gate_m, gmlp->d_sgu_gate_v,
        size_sgu_gate, gmlp->beta1, gmlp->beta2, gmlp->epsilon, gmlp->weight_decay,
        learning_rate, gmlp->batch_size, bias_correction1, bias_correction2);
    CHECK_CUDA(cudaGetLastError());
    
    // Update sgu_proj_weight
    int size_sgu_proj = gmlp->ffn_dim * half_hidden;
    num_blocks = (size_sgu_proj + block_size - 1) / block_size;
    adamw_update_kernel_gmlp<<<num_blocks, block_size>>>(
        gmlp->d_sgu_proj_weight, gmlp->d_sgu_proj_weight_grad, 
        gmlp->d_sgu_proj_m, gmlp->d_sgu_proj_v,
        size_sgu_proj, gmlp->beta1, gmlp->beta2, gmlp->epsilon, gmlp->weight_decay,
        learning_rate, gmlp->batch_size, bias_correction1, bias_correction2);
    CHECK_CUDA(cudaGetLastError());
    
    // Update sgu_out_weight
    int size_sgu_out = gmlp->hidden_dim * gmlp->ffn_dim;
    num_blocks = (size_sgu_out + block_size - 1) / block_size;
    adamw_update_kernel_gmlp<<<num_blocks, block_size>>>(
        gmlp->d_sgu_out_weight, gmlp->d_sgu_out_weight_grad, 
        gmlp->d_sgu_out_m, gmlp->d_sgu_out_v,
        size_sgu_out, gmlp->beta1, gmlp->beta2, gmlp->epsilon, gmlp->weight_decay,
        learning_rate, gmlp->batch_size, bias_correction1, bias_correction2);
    CHECK_CUDA(cudaGetLastError());
    
    // Update proj_out_weight
    int size_proj_out = gmlp->output_dim * gmlp->hidden_dim;
    num_blocks = (size_proj_out + block_size - 1) / block_size;
    adamw_update_kernel_gmlp<<<num_blocks, block_size>>>(
        gmlp->d_proj_out_weight, gmlp->d_proj_out_weight_grad, 
        gmlp->d_proj_out_m, gmlp->d_proj_out_v,
        size_proj_out, gmlp->beta1, gmlp->beta2, gmlp->epsilon, gmlp->weight_decay,
        learning_rate, gmlp->batch_size, bias_correction1, bias_correction2);
    CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------
// Function: get_predictions_gmlp
// Copies the predictions from device to host for evaluation
// ---------------------------------------------------------------------
void get_predictions_gmlp(GMLP* gmlp) {
    CHECK_CUDA(cudaMemcpy(gmlp->h_predictions, gmlp->d_predictions,
                     gmlp->batch_size * gmlp->output_dim * sizeof(float), cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------
// Function: save_gmlp
// Saves the model weights to a binary file.
// ---------------------------------------------------------------------
void save_gmlp(GMLP* gmlp, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if (!file) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        return;
    }
    
    int half_hidden = gmlp->hidden_dim / 2;
    
    // Write dimensions
    fwrite(&gmlp->input_dim, sizeof(int), 1, file);
    fwrite(&gmlp->hidden_dim, sizeof(int), 1, file);
    fwrite(&gmlp->ffn_dim, sizeof(int), 1, file);
    fwrite(&gmlp->output_dim, sizeof(int), 1, file);
    fwrite(&gmlp->batch_size, sizeof(int), 1, file);
    
    // Write Adam hyperparameters
    fwrite(&gmlp->beta1, sizeof(float), 1, file);
    fwrite(&gmlp->beta2, sizeof(float), 1, file);
    fwrite(&gmlp->epsilon, sizeof(float), 1, file);
    fwrite(&gmlp->weight_decay, sizeof(float), 1, file);
    fwrite(&gmlp->adam_t, sizeof(int), 1, file);

    // Allocate host buffers for Adam state
    float* h_proj_in_m = (float*)malloc(gmlp->hidden_dim * gmlp->input_dim * sizeof(float));
    float* h_proj_in_v = (float*)malloc(gmlp->hidden_dim * gmlp->input_dim * sizeof(float));
    float* h_sgu_gate_m = (float*)malloc(gmlp->ffn_dim * half_hidden * sizeof(float));
    float* h_sgu_gate_v = (float*)malloc(gmlp->ffn_dim * half_hidden * sizeof(float));
    float* h_sgu_proj_m = (float*)malloc(gmlp->ffn_dim * half_hidden * sizeof(float));
    float* h_sgu_proj_v = (float*)malloc(gmlp->ffn_dim * half_hidden * sizeof(float));
    float* h_sgu_out_m = (float*)malloc(gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float));
    float* h_sgu_out_v = (float*)malloc(gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float));
    float* h_proj_out_m = (float*)malloc(gmlp->output_dim * gmlp->hidden_dim * sizeof(float));
    float* h_proj_out_v = (float*)malloc(gmlp->output_dim * gmlp->hidden_dim * sizeof(float));
    
    // Copy weights to host (in case they were updated on device)
    CHECK_CUDA(cudaMemcpy(gmlp->h_proj_in_weight, gmlp->d_proj_in_weight, 
                     gmlp->hidden_dim * gmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->h_sgu_gate_weight, gmlp->d_sgu_gate_weight, 
                     gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->h_sgu_proj_weight, gmlp->d_sgu_proj_weight, 
                     gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->h_sgu_out_weight, gmlp->d_sgu_out_weight, 
                     gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(gmlp->h_proj_out_weight, gmlp->d_proj_out_weight, 
                     gmlp->output_dim * gmlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Copy Adam state from device to host
    CHECK_CUDA(cudaMemcpy(h_proj_in_m, gmlp->d_proj_in_m, 
                     gmlp->hidden_dim * gmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_proj_in_v, gmlp->d_proj_in_v, 
                     gmlp->hidden_dim * gmlp->input_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sgu_gate_m, gmlp->d_sgu_gate_m, 
                     gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sgu_gate_v, gmlp->d_sgu_gate_v, 
                     gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sgu_proj_m, gmlp->d_sgu_proj_m, 
                     gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sgu_proj_v, gmlp->d_sgu_proj_v, 
                     gmlp->ffn_dim * half_hidden * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sgu_out_m, gmlp->d_sgu_out_m, 
                     gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_sgu_out_v, gmlp->d_sgu_out_v, 
                     gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_proj_out_m, gmlp->d_proj_out_m, 
                     gmlp->output_dim * gmlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_proj_out_v, gmlp->d_proj_out_v, 
                     gmlp->output_dim * gmlp->hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Write weight matrices
    fwrite(gmlp->h_proj_in_weight, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(gmlp->h_sgu_gate_weight, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(gmlp->h_sgu_proj_weight, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(gmlp->h_sgu_out_weight, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(gmlp->h_proj_out_weight, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    
    // Write Adam state
    fwrite(h_proj_in_m, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(h_proj_in_v, sizeof(float), gmlp->hidden_dim * gmlp->input_dim, file);
    fwrite(h_sgu_gate_m, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(h_sgu_gate_v, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(h_sgu_proj_m, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(h_sgu_proj_v, sizeof(float), gmlp->ffn_dim * half_hidden, file);
    fwrite(h_sgu_out_m, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(h_sgu_out_v, sizeof(float), gmlp->hidden_dim * gmlp->ffn_dim, file);
    fwrite(h_proj_out_m, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    fwrite(h_proj_out_v, sizeof(float), gmlp->output_dim * gmlp->hidden_dim, file);
    
    // Free host buffers for Adam state
    free(h_proj_in_m);
    free(h_proj_in_v);
    free(h_sgu_gate_m);
    free(h_sgu_gate_v);
    free(h_sgu_proj_m);
    free(h_sgu_proj_v);
    free(h_sgu_out_m);
    free(h_sgu_out_v);
    free(h_proj_out_m);
    free(h_proj_out_v);
    
    fclose(file);
    printf("Model saved to %s\n", filename);
}

// ---------------------------------------------------------------------
// Function: load_gmlp
// Loads the model weights from a binary file and initializes a new GMLP.
// ---------------------------------------------------------------------
GMLP* load_gmlp(const char* filename, int custom_batch_size) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file for reading: %s\n", filename);
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
    
    // Initialize model
    GMLP* gmlp = init_gmlp(input_dim, hidden_dim, ffn_dim, output_dim, batch_size);
    
    // Read Adam hyperparameters
    fread(&gmlp->beta1, sizeof(float), 1, file);
    fread(&gmlp->beta2, sizeof(float), 1, file);
    fread(&gmlp->epsilon, sizeof(float), 1, file);
    fread(&gmlp->weight_decay, sizeof(float), 1, file);
    fread(&gmlp->adam_t, sizeof(int), 1, file);
    
    int half_hidden = hidden_dim / 2;
    
    // Allocate host buffers for Adam state
    float* h_proj_in_m = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* h_proj_in_v = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    float* h_sgu_gate_m = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    float* h_sgu_gate_v = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    float* h_sgu_proj_m = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    float* h_sgu_proj_v = (float*)malloc(ffn_dim * half_hidden * sizeof(float));
    float* h_sgu_out_m = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    float* h_sgu_out_v = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    float* h_proj_out_m = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    float* h_proj_out_v = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Read weights
    fread(gmlp->h_proj_in_weight, sizeof(float), hidden_dim * input_dim, file);
    fread(gmlp->h_sgu_gate_weight, sizeof(float), ffn_dim * half_hidden, file);
    fread(gmlp->h_sgu_proj_weight, sizeof(float), ffn_dim * half_hidden, file);
    fread(gmlp->h_sgu_out_weight, sizeof(float), hidden_dim * ffn_dim, file);
    fread(gmlp->h_proj_out_weight, sizeof(float), output_dim * hidden_dim, file);
    
    // Read Adam state
    fread(h_proj_in_m, sizeof(float), hidden_dim * input_dim, file);
    fread(h_proj_in_v, sizeof(float), hidden_dim * input_dim, file);
    fread(h_sgu_gate_m, sizeof(float), ffn_dim * half_hidden, file);
    fread(h_sgu_gate_v, sizeof(float), ffn_dim * half_hidden, file);
    fread(h_sgu_proj_m, sizeof(float), ffn_dim * half_hidden, file);
    fread(h_sgu_proj_v, sizeof(float), ffn_dim * half_hidden, file);
    fread(h_sgu_out_m, sizeof(float), hidden_dim * ffn_dim, file);
    fread(h_sgu_out_v, sizeof(float), hidden_dim * ffn_dim, file);
    fread(h_proj_out_m, sizeof(float), output_dim * hidden_dim, file);
    fread(h_proj_out_v, sizeof(float), output_dim * hidden_dim, file);
    
    fclose(file);
    
    // Copy weights to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_weight, gmlp->h_proj_in_weight, 
                     hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_weight, gmlp->h_sgu_gate_weight, 
                     ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_weight, gmlp->h_sgu_proj_weight, 
                     ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_out_weight, gmlp->h_sgu_out_weight, 
                     hidden_dim * ffn_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_weight, gmlp->h_proj_out_weight, 
                     output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Copy Adam state to device
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_m, h_proj_in_m, 
                     hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_in_v, h_proj_in_v, 
                     hidden_dim * input_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_m, h_sgu_gate_m, 
                     ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_gate_v, h_sgu_gate_v, 
                     ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_m, h_sgu_proj_m, 
                     ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_proj_v, h_sgu_proj_v, 
                     ffn_dim * half_hidden * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_out_m, h_sgu_out_m, 
                     hidden_dim * ffn_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_sgu_out_v, h_sgu_out_v, 
                     hidden_dim * ffn_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_m, h_proj_out_m, 
                     output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(gmlp->d_proj_out_v, h_proj_out_v, 
                     output_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    
    // Free host buffers for Adam state
    free(h_proj_in_m);
    free(h_proj_in_v);
    free(h_sgu_gate_m);
    free(h_sgu_gate_v);
    free(h_sgu_proj_m);
    free(h_sgu_proj_v);
    free(h_sgu_out_m);
    free(h_sgu_out_v);
    free(h_proj_out_m);
    free(h_proj_out_v);
    
    printf("Model loaded from %s\n", filename);
    
    return gmlp;
}

// ---------------------------------------------------------------------
// Function: free_gmlp
// Frees all allocated memory (both device and host) and destroys handles.
// ---------------------------------------------------------------------
void free_gmlp(GMLP* gmlp) {
    if (!gmlp) return;
    
    // Free device memory for weights
    cudaFree(gmlp->d_proj_in_weight);
    cudaFree(gmlp->d_sgu_gate_weight);
    cudaFree(gmlp->d_sgu_proj_weight);
    cudaFree(gmlp->d_sgu_out_weight);
    cudaFree(gmlp->d_proj_out_weight);
    
    // Free device memory for gradients
    cudaFree(gmlp->d_proj_in_weight_grad);
    cudaFree(gmlp->d_sgu_gate_weight_grad);
    cudaFree(gmlp->d_sgu_proj_weight_grad);
    cudaFree(gmlp->d_sgu_out_weight_grad);
    cudaFree(gmlp->d_proj_out_weight_grad);
    
    // Free device memory for Adam states
    cudaFree(gmlp->d_proj_in_m);
    cudaFree(gmlp->d_proj_in_v);
    cudaFree(gmlp->d_sgu_gate_m);
    cudaFree(gmlp->d_sgu_gate_v);
    cudaFree(gmlp->d_sgu_proj_m);
    cudaFree(gmlp->d_sgu_proj_v);
    cudaFree(gmlp->d_sgu_out_m);
    cudaFree(gmlp->d_sgu_out_v);
    cudaFree(gmlp->d_proj_out_m);
    cudaFree(gmlp->d_proj_out_v);
    
    // Free device memory for intermediate activations
    cudaFree(gmlp->d_proj_in_output);
    cudaFree(gmlp->d_sgu_gate_output);
    cudaFree(gmlp->d_gate_activated);
    cudaFree(gmlp->d_sgu_proj_output);
    cudaFree(gmlp->d_gated_output);
    cudaFree(gmlp->d_sgu_output);
    cudaFree(gmlp->d_predictions);
    
    // Free device memory for intermediate gradients
    cudaFree(gmlp->d_error);
    cudaFree(gmlp->d_sgu_output_grad);
    cudaFree(gmlp->d_gated_output_grad);
    cudaFree(gmlp->d_sgu_proj_grad);
    cudaFree(gmlp->d_gate_activated_grad);
    cudaFree(gmlp->d_sgu_gate_grad);
    cudaFree(gmlp->d_u_part_grad);
    cudaFree(gmlp->d_v_part_grad);
    cudaFree(gmlp->d_proj_in_grad);
    
    // Free host memory
    free(gmlp->h_proj_in_weight);
    free(gmlp->h_sgu_gate_weight);
    free(gmlp->h_sgu_proj_weight);
    free(gmlp->h_sgu_out_weight);
    free(gmlp->h_proj_out_weight);
    free(gmlp->h_predictions);
    
    // Destroy cuBLAS handle
    cublasDestroy(gmlp->cublas_handle);
    
    // Free struct itself
    free(gmlp);
}

#endif