/*
Layer 4:
block 0:
Conv1: input shape(256,256,40,40), output shape(256,512,20,20); stride =2
bn1: input shape(256, 512, 20, 20), output shape(256, 512, 20, 20)
relu: skip
conv2: input shape(256, 512,20, 20), output shape(256, 512, 20, 20); stride =1
bn2: input shape(256, 512, 20, 20), output_shape (256, 512, 20, 20)

Downsample:
ds_conv1: input shape(256,256,40,40), output shape(256,512,20,20); stride =2
ds_bn1: input shape(256, 512, 20, 20), output shape(256, 512, 20, 20)

output of block 0: (256, 512, 20, 20)

block 1:
conv1: input shape(256, 512,20, 20), output shape(256, 512, 20, 20); stride =1
bn1: input shape(256, 512, 20, 20), output_shape (256, 512, 20, 20)
conv2: input shape(256, 512,20, 20), output shape(256, 512, 20, 20); stride =1
bn2: input shape(256, 512, 20, 20), output_shape (256, 512, 20, 20)

Avgpool: input shape (256, 512, 20, 20), output shape(256, 512, 1, 1) or (256, 512)
//Flatten: input shape (256, 512, 1, 1), output shape (256, 512)

Linear: input shape (256, 512), output shape (256, 512)
Tanh: input shape(256, 512), output shape(256, 512)

*/

#include "top.h"
#include <hls_math.h>
#include <iostream>
#include <cmath>
void ResNet(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t block0_weights_conv1[OUT_CHANNELS][IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block0_gamma1[OUT_CHANNELS],
    data_t block0_beta1[OUT_CHANNELS],
    data_t block0_mean1[OUT_CHANNELS],
    data_t block0_var1[OUT_CHANNELS],
    data_t block0_output_bn1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block0_weights_conv2[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block0_gamma2[OUT_CHANNELS],
    data_t block0_beta2[OUT_CHANNELS],
    data_t block0_mean2[OUT_CHANNELS],
    data_t block0_var2[OUT_CHANNELS],
    data_t block0_output_bn2[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t input_local[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t output_local[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block0_weights_ds_conv1[OUT_CHANNELS][IN_CHANNELS][DS_KERNEL_HEIGHT][DS_KERNEL_WIDTH],
    data_t block0_ds_gamma1[OUT_CHANNELS],
    data_t block0_ds_beta1[OUT_CHANNELS],
    data_t block0_ds_mean1[OUT_CHANNELS],
    data_t block0_ds_var1[OUT_CHANNELS],
    data_t output_block0[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block1_weights_conv1[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block1_gamma1[OUT_CHANNELS],
    data_t block1_beta1[OUT_CHANNELS],
    data_t block1_mean1[OUT_CHANNELS],
    data_t block1_var1[OUT_CHANNELS],
    data_t block1_output_bn1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block1_weights_conv2[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block1_gamma2[OUT_CHANNELS],
    data_t block1_beta2[OUT_CHANNELS],
    data_t block1_mean2[OUT_CHANNELS],
    data_t block1_var2[OUT_CHANNELS],
    data_t block1_output_bn2[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t fc_weights[OUT_CHANNELS][OUT_CHANNELS],
    data_t fc_bias[OUT_CHANNELS],
    data_t tanh_output[BATCH][OUT_CHANNELS])
{
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = block0_weights_conv1 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = block0_gamma1 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = block0_beta1 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = block0_mean1 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = block0_var1 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = block0_output_bn1 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = block0_weights_conv2 offset = slave bundle = gmem7
#pragma HLS INTERFACE m_axi port = block0_gamma2 offset = slave bundle = gmem8
#pragma HLS INTERFACE m_axi port = block0_beta2 offset = slave bundle = gmem9
#pragma HLS INTERFACE m_axi port = block0_mean2 offset = slave bundle = gmem10
#pragma HLS INTERFACE m_axi port = block0_var2 offset = slave bundle = gmem11
#pragma HLS INTERFACE m_axi port = block0_output_bn2 offset = slave bundle = gmem12
#pragma HLS INTERFACE m_axi port = input_local offset = slave bundle = gmem13
#pragma HLS INTERFACE m_axi port = output_local offset = slave bundle = gmem14
#pragma HLS INTERFACE m_axi port = block0_weights_ds_conv1 offset = slave bundle = gmem15
#pragma HLS INTERFACE m_axi port = block0_ds_gamma1 offset = slave bundle = gmem16
#pragma HLS INTERFACE m_axi port = block0_ds_beta1 offset = slave bundle = gmem17
#pragma HLS INTERFACE m_axi port = block0_ds_mean1 offset = slave bundle = gmem18
#pragma HLS INTERFACE m_axi port = block0_ds_var1 offset = slave bundle = gmem19
#pragma HLS INTERFACE m_axi port = output_block0 offset = slave bundle = gmem20
#pragma HLS INTERFACE m_axi port = block1_weights_conv1 offset = slave bundle = gmem21
#pragma HLS INTERFACE m_axi port = block1_gamma1 offset = slave bundle = gmem22
#pragma HLS INTERFACE m_axi port = block1_beta1 offset = slave bundle = gmem23
#pragma HLS INTERFACE m_axi port = block1_mean1 offset = slave bundle = gmem24
#pragma HLS INTERFACE m_axi port = block1_var1 offset = slave bundle = gmem25
#pragma HLS INTERFACE m_axi port = block1_output_bn1 offset = slave bundle = gmem26
#pragma HLS INTERFACE m_axi port = block1_weights_conv2 offset = slave bundle = gmem27
#pragma HLS INTERFACE m_axi port = block1_gamma2 offset = slave bundle = gmem28
#pragma HLS INTERFACE m_axi port = block1_beta2 offset = slave bundle = gmem29
#pragma HLS INTERFACE m_axi port = block1_mean2 offset = slave bundle = gmem30
#pragma HLS INTERFACE m_axi port = block1_var2 offset = slave bundle = gmem31
#pragma HLS INTERFACE m_axi port = block1_output_bn2 offset = slave bundle = gmem32
#pragma HLS INTERFACE m_axi port = fc_weights offset = slave bundle = gmem33
#pragma HLS INTERFACE m_axi port = fc_bias offset = slave bundle = gmem34
#pragma HLS INTERFACE m_axi port = tanh_output offset = slave bundle = gmem35

// S-AXILITE control interface (required for each argument and return)
#pragma HLS INTERFACE s_axilite port = return bundle = CTRL
#pragma HLS INTERFACE s_axilite port = input bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_weights_conv1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_gamma1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_beta1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_mean1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_var1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_output_bn1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_weights_conv2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_gamma2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_beta2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_mean2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_var2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_output_bn2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = input_local bundle = CTRL
#pragma HLS INTERFACE s_axilite port = output_local bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_weights_ds_conv1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_ds_gamma1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_ds_beta1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_ds_mean1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block0_ds_var1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = output_block0 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_weights_conv1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_gamma1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_beta1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_mean1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_var1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_output_bn1 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_weights_conv2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_gamma2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_beta2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_mean2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_var2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = block1_output_bn2 bundle = CTRL
#pragma HLS INTERFACE s_axilite port = fc_weights bundle = CTRL
#pragma HLS INTERFACE s_axilite port = fc_bias bundle = CTRL
#pragma HLS INTERFACE s_axilite port = tanh_output bundle = CTRL
    // block 0
    // data_t local_batch0_conv2d_output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2];
    data_t avg_pooling_local_output[BATCH][OUT_CHANNELS];
#pragma HLS ARRAY_PARTITION variable = avg_pooling_local_output complete dim = 1

    data_t local_weights_ds[OUT_CHANNELS][IN_CHANNELS];
#pragma HLS ARRAY_PARTITION variable = local_weights_ds complete dim = 1

    load_weights_ds(block0_weights_ds_conv1, local_weights_ds); // Load weights for downsample conv
    // 1. Block 0
    // 1a. Conv1+BatchNorm+ReLU
    conv2d_stride(input, block0_weights_conv1, block0_output_bn1, block0_gamma1, block0_beta1, block0_mean1, block0_var1, true);
    // 1b. Conv2+BatchNorm+ReLU
    conv2d(block0_output_bn1, block0_weights_conv2, block0_output_bn2, block0_gamma2, block0_beta2, block0_mean2, block0_var2, true);
    // 1c. Downsample Conv1+BatchNorm
    conv2d_stride_ds(input_local, local_weights_ds, output_local, block0_ds_gamma1, block0_ds_beta1, block0_ds_mean1, block0_ds_var1); // ReLU activation
    Add_4D(block0_output_bn2, output_local, output_block0);
    // 2. Block 1
    // 2a. Conv1+BatchNorm+ReLU
    conv2d(output_block0, block1_weights_conv1, block1_output_bn1, block1_gamma1, block1_beta1, block1_mean1, block1_var1, true);
    // 2b. Conv2+BatchNorm+ReLU
    conv2d(block1_output_bn1, block1_weights_conv2, block1_output_bn2, block1_gamma2, block1_beta2, block1_mean2, block1_var2, true);
    // 3. Average Pooling
    average_pooling(block1_output_bn2, avg_pooling_local_output);
    // 4. Fully Connected Layer+Tanh
    fully_connected(avg_pooling_local_output, fc_weights, fc_bias, tanh_output);
}

void load_weights_ds(data_t weights[OUT_CHANNELS][IN_CHANNELS][DS_KERNEL_HEIGHT][DS_KERNEL_WIDTH], data_t local_weights[OUT_CHANNELS][IN_CHANNELS])
{
    for (int oc = 0; oc < OUT_CHANNELS; ++oc)
    {
        for (int ic = 0; ic < IN_CHANNELS; ++ic)
        {
#pragma HLS pipeline II = 1
            local_weights[oc][ic] = weights[oc][ic][0][0];
        }
    }
}
// Simple Addition of 4D tensors
//  4D tensor: input[BATCH][IN_CHANNELS][HEIGHT][WIDTH]
void Add_4D(
    data_t input1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t input2[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2])
{
    data_t local_y;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int ic = 0; ic < OUT_CHANNELS; ++ic)
        {
            for (int i = 0; i < HEIGHT / 2; ++i)
            {
#pragma HLS pipeline II = 1
                for (int j = 0; j < WIDTH / 2; ++j)
                {
                    local_y = input1[b][ic][i][j] + input2[b][ic][i][j];
                    output[b][ic][i][j] = (local_y > 0) ? local_y : data_t(0); // ReLU activation
                }
            }
        }
    }
}

void conv2d_stride(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t gamma[CHANNELS],
    data_t beta[CHANNELS],
    data_t mean[CHANNELS],
    data_t var[CHANNELS],
    bool relu) // ReLU activation flag
{
    const data_t eps = 1e-5;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            data_t inv_std = data_t(1.0) / std::sqrt(var[oc] + eps); //(gamma/sqrt(var[c] + eps))
            data_t b_local = beta[oc];
            data_t gamma_local = gamma[oc] * inv_std;
            data_t mean_local = mean[oc];
            for (int i = 0; i < HEIGHT; i += 2)
            {
                for (int j = 0; j < WIDTH; j += 2)
                {
                    data_t sum = 0;
                    for (int ic = 0; ic < IN_CHANNELS; ++ic)
                    { // loop over input channels
                        for (int kh = 0; kh < KERNEL_HEIGHT; ++kh)
                        {
                            for (int kw = 0; kw < KERNEL_WIDTH; ++kw)
                            {
                                int in_h = i + kh - PADDING;
                                int in_w = j + kw - PADDING;
                                if (in_h >= 0 && in_h < HEIGHT && in_w >= 0 && in_w < WIDTH)
                                {
                                    sum += input[b][ic][in_h][in_w] * weights[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    sum = gamma_local * (sum - mean_local) + b_local;
                    if (relu)
                    {
                        sum = (sum > 0) ? sum : data_t(0); // ReLU activation
                    }
                    output[b][oc][i / 2][j / 2] = sum;
                }
            }
        }
    }
}

void conv2d_stride_ds(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t weights[OUT_CHANNELS][IN_CHANNELS],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS]) // ReLU activation flag,
{
    const data_t eps = 1e-5;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            data_t inv_std = data_t(1.0) / std::sqrt(var[oc] + eps); //(gamma/sqrt(var[c] + eps))
            data_t b_local = beta[oc];
            data_t gamma_local = gamma[oc] * inv_std;
            data_t mean_local = mean[oc];

            for (int i = 0; i < HEIGHT; i += 2)
            { // stride 2
                for (int j = 0; j < WIDTH; j += 2)
                {
                    data_t sum = 0;
                    // bring ic loop outside i,j?
                    // go to every 256 channel and multiply over all channel weights
                    for (int ic = 0; ic < IN_CHANNELS; ++ic)
                    { // TODO: Can parallelize across ic
                        sum += input[b][ic][i][j] * weights[oc][ic];
                    }
                    sum = gamma_local * (sum - mean_local) + b_local;
                    output[b][oc][i / 2][j / 2] = sum;
                }
            }
        }
    }
}

// conv for stride =1
void conv2d(
    data_t input[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t weights[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t gamma[OUT_CHANNELS],
    data_t beta[OUT_CHANNELS],
    data_t mean[OUT_CHANNELS],
    data_t var[OUT_CHANNELS],
    bool relu) // ReLU activation flag
{
    const data_t eps = 1e-5;
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            data_t inv_std = data_t(1.0) / std::sqrt(var[oc] + eps); //(gamma/sqrt(var[c] + eps))
            data_t b_local = beta[oc];
            data_t gamma_local = gamma[oc] * inv_std;
            data_t mean_local = mean[oc];

            for (int i = 0; i < HEIGHT / 2; ++i)
            {
                for (int j = 0; j < WIDTH / 2; ++j)
                {
                    data_t sum = 0;
                    for (int ic = 0; ic < OUT_CHANNELS; ++ic)
                    {
                        for (int kh = 0; kh < KERNEL_HEIGHT; ++kh)
                        {
                            for (int kw = 0; kw < KERNEL_WIDTH; ++kw)
                            {
                                int in_h = i + kh - PADDING;
                                int in_w = j + kw - PADDING;
                                if (in_h >= 0 && in_h < HEIGHT / 2 && in_w >= 0 && in_w < WIDTH / 2)
                                {
                                    sum += input[b][ic][in_h][in_w] * weights[oc][ic][kh][kw];
                                }
                            }
                        }
                    }
                    sum = gamma_local * (sum - mean_local) + b_local;
                    if (relu)
                    {
                        sum = (sum > 0) ? sum : data_t(0); // ReLU activation
                    }
                    output[b][oc][i][j] = sum;
                }
            }
        }
    }
}

void average_pooling(
    data_t input[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t output[BATCH][OUT_CHANNELS])
{ // perform simple average
    for (int b = 0; b < BATCH; ++b)
    {
        for (int c = 0; c < OUT_CHANNELS; ++c)
        {
            data_t sum = 0;
            for (int h = 0; h < HEIGHT / 2; ++h)
            {
                for (int w = 0; w < WIDTH / 2; ++w)
                {
                    sum += input[b][c][h][w];
                }
            }
            output[b][c] = sum / ((HEIGHT / 2) * (WIDTH / 2)); // Average pooling
        }
    }
}

void fully_connected(
    data_t input[BATCH][OUT_CHANNELS],
    data_t weights[OUT_CHANNELS][OUT_CHANNELS],
    data_t bias[OUT_CHANNELS],
    data_t output[BATCH][OUT_CHANNELS])
{
    // Perform a Matrix Multiplication between input and weights
    // Y=X*W^T+Bias
    // Have to take transpose of weights
    // First do over batch
    for (int b = 0; b < BATCH; b++)
    {
        // Now for each column of the weights
        for (int j = 0; j < OUT_CHANNELS; j++)
        {
            // Now access X
            output[b][j] = 0; // is this necessary?
            // Now access each element of the columns of the weights
            for (int k = 0; k < OUT_CHANNELS; k++)
            {
                // Multiply the input with the weights and add bias
                // k
                output[b][j] += input[b][k] * weights[j][k]; // If wrong maybe try swapping k with j
            }
            output[b][j] += bias[j]; // Add bias to the output
                                     // apply tanh
            // output[b][j]=hls::tanh(output[b][j]);
            output[b][j] = output[b][j] * (27 + output[b][j] * output[b][j]) / (27 + 9 * output[b][j] * output[b][j]);
        }
    }
}

// void tanh_approx(data_t input[BATCH][OUT_CHANNELS], data_t output[BATCH][OUT_CHANNELS])
// {
//     for (int b = 0; b < BATCH; b++)
//     {
//         for (int c = 0; c < OUT_CHANNELS; c++)
//         {
//             output[b][c] = input[b][c] * (27 + input[b][c] * input[b][c]) / (27 + 9 * input[b][c] * input[b][c]);
//         }
//     }
// }