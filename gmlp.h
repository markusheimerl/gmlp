#ifndef GMLP_H
#define GMLP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cblas.h>

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
} GMLP;

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
    
    // Allocate weights
    gmlp->proj_in_weight = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    gmlp->sgu_gate_weight = (float*)malloc(ffn_dim * hidden_dim * sizeof(float));
    gmlp->sgu_proj_weight = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    gmlp->proj_out_weight = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate gradients
    gmlp->proj_in_weight_grad = (float*)malloc(hidden_dim * input_dim * sizeof(float));
    gmlp->sgu_gate_weight_grad = (float*)malloc(ffn_dim * hidden_dim * sizeof(float));
    gmlp->sgu_proj_weight_grad = (float*)malloc(hidden_dim * ffn_dim * sizeof(float));
    gmlp->proj_out_weight_grad = (float*)malloc(output_dim * hidden_dim * sizeof(float));
    
    // Allocate Adam buffers
    gmlp->proj_in_m = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    gmlp->proj_in_v = (float*)calloc(hidden_dim * input_dim, sizeof(float));
    gmlp->sgu_gate_m = (float*)calloc(ffn_dim * hidden_dim, sizeof(float));
    gmlp->sgu_gate_v = (float*)calloc(ffn_dim * hidden_dim, sizeof(float));
    gmlp->sgu_proj_m = (float*)calloc(hidden_dim * ffn_dim, sizeof(float));
    gmlp->sgu_proj_v = (float*)calloc(hidden_dim * ffn_dim, sizeof(float));
    gmlp->proj_out_m = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    gmlp->proj_out_v = (float*)calloc(output_dim * hidden_dim, sizeof(float));
    
    // Allocate intermediate activations
    gmlp->proj_in_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    gmlp->sgu_gate_output = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->gate_activated = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->sgu_proj_input = (float*)malloc(batch_size * ffn_dim * sizeof(float));
    gmlp->sgu_output = (float*)malloc(batch_size * hidden_dim * sizeof(float));
    gmlp->predictions = (float*)malloc(batch_size * output_dim * sizeof(float));
    
    // Allocate intermediate gradients
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
    
    // Free the struct
    free(gmlp);
}

// Forward pass
void forward_pass_gmlp(GMLP* gmlp, float* X) {
    // 1. Input projection: X → hidden_dim
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                gmlp->batch_size, gmlp->hidden_dim, gmlp->input_dim,
                1.0f, X, gmlp->input_dim,
                gmlp->proj_in_weight, gmlp->hidden_dim,
                0.0f, gmlp->proj_in_output, gmlp->hidden_dim);
    
    // Apply GELU activation to the input projection
    for (int i = 0; i < gmlp->batch_size * gmlp->hidden_dim; i++) {
        float x = gmlp->proj_in_output[i];
        // Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        gmlp->proj_in_output[i] = x * 0.5f * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
    }
    
    // 2. Spatial Gating Unit (SGU)
    // 2a. Compute gate values
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                gmlp->batch_size, gmlp->ffn_dim, gmlp->hidden_dim,
                1.0f, gmlp->proj_in_output, gmlp->hidden_dim,
                gmlp->sgu_gate_weight, gmlp->ffn_dim,
                0.0f, gmlp->sgu_gate_output, gmlp->ffn_dim);
    
    // Apply sigmoid activation to gate values
    for (int i = 0; i < gmlp->batch_size * gmlp->ffn_dim; i++) {
        gmlp->gate_activated[i] = 1.0f / (1.0f + expf(-gmlp->sgu_gate_output[i]));
    }
    
    // 2b. Project to FFN dimension (for element-wise multiplication with gate)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                gmlp->batch_size, gmlp->ffn_dim, gmlp->hidden_dim,
                1.0f, gmlp->proj_in_output, gmlp->hidden_dim,
                gmlp->sgu_gate_weight, gmlp->ffn_dim,
                0.0f, gmlp->sgu_proj_input, gmlp->ffn_dim);
    
    // 2c. Apply gating (element-wise multiplication)
    for (int i = 0; i < gmlp->batch_size * gmlp->ffn_dim; i++) {
        gmlp->sgu_proj_input[i] *= gmlp->gate_activated[i];
    }
    
    // 2d. Project back to hidden dimension
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                gmlp->batch_size, gmlp->hidden_dim, gmlp->ffn_dim,
                1.0f, gmlp->sgu_proj_input, gmlp->ffn_dim,
                gmlp->sgu_proj_weight, gmlp->hidden_dim,
                0.0f, gmlp->sgu_output, gmlp->hidden_dim);
    
    // 3. Output projection: hidden_dim → output_dim
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                gmlp->batch_size, gmlp->output_dim, gmlp->hidden_dim,
                1.0f, gmlp->sgu_output, gmlp->hidden_dim,
                gmlp->proj_out_weight, gmlp->output_dim,
                0.0f, gmlp->predictions, gmlp->output_dim);
}

// Calculate loss (Mean Squared Error)
float calculate_loss_gmlp(GMLP* gmlp, float* y) {
    float loss = 0.0f;
    for (int i = 0; i < gmlp->batch_size * gmlp->output_dim; i++) {
        gmlp->error[i] = gmlp->predictions[i] - y[i];
        loss += gmlp->error[i] * gmlp->error[i];
    }
    return loss / (gmlp->batch_size * gmlp->output_dim);
}

// Zero out all gradients
void zero_gradients_gmlp(GMLP* gmlp) {
    memset(gmlp->proj_in_weight_grad, 0, gmlp->hidden_dim * gmlp->input_dim * sizeof(float));
    memset(gmlp->sgu_gate_weight_grad, 0, gmlp->ffn_dim * gmlp->hidden_dim * sizeof(float));
    memset(gmlp->sgu_proj_weight_grad, 0, gmlp->hidden_dim * gmlp->ffn_dim * sizeof(float));
    memset(gmlp->proj_out_weight_grad, 0, gmlp->output_dim * gmlp->hidden_dim * sizeof(float));
}

// Backward pass
void backward_pass_gmlp(GMLP* gmlp, float* X) {
    // 1. Gradient of output projection
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                gmlp->hidden_dim, gmlp->output_dim, gmlp->batch_size,
                1.0f, gmlp->sgu_output, gmlp->hidden_dim,
                gmlp->error, gmlp->output_dim,
                0.0f, gmlp->proj_out_weight_grad, gmlp->output_dim);
    
    // 2. Gradient flowing back to SGU output
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                gmlp->batch_size, gmlp->hidden_dim, gmlp->output_dim,
                1.0f, gmlp->error, gmlp->output_dim,
                gmlp->proj_out_weight, gmlp->output_dim,
                0.0f, gmlp->sgu_output_grad, gmlp->hidden_dim);
    
    // 3. Gradient of SGU projection (from hidden_dim back to ffn_dim)
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                gmlp->ffn_dim, gmlp->hidden_dim, gmlp->batch_size,
                1.0f, gmlp->sgu_proj_input, gmlp->ffn_dim,
                gmlp->sgu_output_grad, gmlp->hidden_dim,
                0.0f, gmlp->sgu_proj_weight_grad, gmlp->hidden_dim);
    
    // 4. Gradient flowing through SGU projection to gated input
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                gmlp->batch_size, gmlp->ffn_dim, gmlp->hidden_dim,
                1.0f, gmlp->sgu_output_grad, gmlp->hidden_dim,
                gmlp->sgu_proj_weight, gmlp->hidden_dim,
                0.0f, gmlp->sgu_proj_grad, gmlp->ffn_dim);
    
    // 5. Gradient through the gating mechanism
    // 5a. Gradient to gate activation
    for (int i = 0; i < gmlp->batch_size * gmlp->ffn_dim; i++) {
        gmlp->gate_activated_grad[i] = gmlp->sgu_proj_grad[i] * gmlp->sgu_proj_input[i] / gmlp->gate_activated[i];
    }
    
    // 5b. Gradient through sigmoid
    for (int i = 0; i < gmlp->batch_size * gmlp->ffn_dim; i++) {
        float sigmoid = gmlp->gate_activated[i];
        gmlp->sgu_gate_grad[i] = gmlp->gate_activated_grad[i] * sigmoid * (1.0f - sigmoid);
    }
    
    // 6. Gradient to gate weights
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                gmlp->hidden_dim, gmlp->ffn_dim, gmlp->batch_size,
                1.0f, gmlp->proj_in_output, gmlp->hidden_dim,
                gmlp->sgu_gate_grad, gmlp->ffn_dim,
                0.0f, gmlp->sgu_gate_weight_grad, gmlp->ffn_dim);
    
    // 7. Gradient flowing back to the input projection
    // 7a. From the gate
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                gmlp->batch_size, gmlp->hidden_dim, gmlp->ffn_dim,
                1.0f, gmlp->sgu_gate_grad, gmlp->ffn_dim,
                gmlp->sgu_gate_weight, gmlp->ffn_dim,
                0.0f, gmlp->proj_in_grad, gmlp->hidden_dim);
    
    // 7b. Also add gradient from the projection input
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                gmlp->batch_size, gmlp->hidden_dim, gmlp->ffn_dim,
                1.0f, gmlp->sgu_proj_grad, gmlp->ffn_dim,
                gmlp->sgu_gate_weight, gmlp->ffn_dim,
                1.0f, gmlp->proj_in_grad, gmlp->hidden_dim);
    
    // 8. Gradient through GELU activation
    for (int i = 0; i < gmlp->batch_size * gmlp->hidden_dim; i++) {
        float x = gmlp->proj_in_output[i];
        // Approximate GELU derivative
        float cdf = 0.5f * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
        float pdf = 0.797885f * (1.0f + 0.134145f * x * x) * (1.0f - tanhf(0.797885f * (x + 0.044715f * x * x * x)) * tanhf(0.797885f * (x + 0.044715f * x * x * x)));
        float gelu_grad = cdf + x * pdf;
        gmlp->proj_in_grad[i] *= gelu_grad;
    }
    
    // 9. Gradient to input projection weights
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                gmlp->input_dim, gmlp->hidden_dim, gmlp->batch_size,
                1.0f, X, gmlp->input_dim,
                gmlp->proj_in_grad, gmlp->hidden_dim,
                0.0f, gmlp->proj_in_weight_grad, gmlp->hidden_dim);
}

// Update weights using AdamW
void update_weights_gmlp(GMLP* gmlp, float learning_rate) {
    gmlp->t++;  // Increment time step
    
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
    for (int i = 0; i < gmlp->ffn_dim * gmlp->hidden_dim; i++) {
        float grad = gmlp->sgu_gate_weight_grad[i] / gmlp->batch_size;
        
        gmlp->sgu_gate_m[i] = gmlp->beta1 * gmlp->sgu_gate_m[i] + (1.0f - gmlp->beta1) * grad;
        gmlp->sgu_gate_v[i] = gmlp->beta2 * gmlp->sgu_gate_v[i] + (1.0f - gmlp->beta2) * grad * grad;
        
        float update = alpha_t * gmlp->sgu_gate_m[i] / (sqrtf(gmlp->sgu_gate_v[i]) + gmlp->epsilon);
        gmlp->sgu_gate_weight[i] = gmlp->sgu_gate_weight[i] * (1.0f - learning_rate * gmlp->weight_decay) - update;
    }
    
    // Update SGU projection weights
    for (int i = 0; i < gmlp->hidden_dim * gmlp->ffn_dim; i++) {
        float grad = gmlp->sgu_proj_weight_grad[i] / gmlp->batch_size;
        
        gmlp->sgu_proj_m[i] = gmlp->beta1 * gmlp->sgu_proj_m[i] + (1.0f - gmlp->beta1) * grad;
        gmlp->sgu_proj_v[i] = gmlp->beta2 * gmlp->sgu_proj_v[i] + (1.0f - gmlp->beta2) * grad * grad;
        
        float update = alpha_t * gmlp->sgu_proj_m[i] / (sqrtf(gmlp->sgu_proj_v[i]) + gmlp->epsilon);
        gmlp->sgu_proj_weight[i] = gmlp->sgu_proj_weight[i] * (1.0f - learning_rate * gmlp->weight_decay) - update;
    }
    
    // Update projection out weights
    for (int i = 0; i < gmlp->output_dim * gmlp->hidden_dim; i++) {
        float grad = gmlp->proj_out_weight_grad[i] / gmlp->batch_size;
        
        gmlp->proj_out_m[i] = gmlp->beta1 * gmlp->proj_out_m[i] + (1.0f - gmlp->beta1) * grad;
        gmlp->proj_out_v[i] = gmlp->beta2 * gmlp->proj_out_v[i] + (1.0f - gmlp->beta2) * grad * grad;
        
        float update = alpha_t * gmlp->proj_out_m[i] / (sqrtf(gmlp->proj_out_v[i]) + gmlp->epsilon);
        gmlp->proj_out_weight[i] = gmlp->proj_out_weight[i] * (1.0f - learning_rate * gmlp->weight_decay) - update;
    }
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
    
    fclose(file);
    printf("Model loaded from %s\n", filename);
    
    return gmlp;
}

#endif