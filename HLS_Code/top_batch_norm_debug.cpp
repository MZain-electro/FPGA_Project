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
    data_t block0_output_conv1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block0_gamma1[OUT_CHANNELS],
    data_t block0_beta1[OUT_CHANNELS],
    data_t block0_mean1[OUT_CHANNELS],
    data_t block0_var1[OUT_CHANNELS],
    data_t block0_output_bn1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block0_weights_conv2[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block0_output_conv2[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block0_gamma2[OUT_CHANNELS],
    data_t block0_beta2[OUT_CHANNELS],
    data_t block0_mean2[OUT_CHANNELS],
    data_t block0_var2[OUT_CHANNELS],
    data_t block0_output_bn2[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t input_local[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t output_local[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block0_weights_ds_conv1[OUT_CHANNELS][IN_CHANNELS][DS_KERNEL_HEIGHT][DS_KERNEL_WIDTH],
    data_t block0_output_ds_conv1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block0_ds_gamma1[OUT_CHANNELS],
    data_t block0_ds_beta1[OUT_CHANNELS],
    data_t block0_ds_mean1[OUT_CHANNELS],
    data_t block0_ds_var1[OUT_CHANNELS],
    data_t output_block0[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block1_weights_conv1[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block1_output_conv1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block1_gamma1[OUT_CHANNELS],
    data_t block1_beta1[OUT_CHANNELS],
    data_t block1_mean1[OUT_CHANNELS],
    data_t block1_var1[OUT_CHANNELS],
    data_t block1_output_bn1[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block1_weights_conv2[OUT_CHANNELS][OUT_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t block1_output_conv2[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t block1_gamma2[OUT_CHANNELS],
    data_t block1_beta2[OUT_CHANNELS],
    data_t block1_mean2[OUT_CHANNELS],
    data_t block1_var2[OUT_CHANNELS],
    data_t block1_output_bn2[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2],
    data_t output_avgpooling[BATCH][OUT_CHANNELS],
    data_t fc_weights[OUT_CHANNELS][OUT_CHANNELS],
    data_t fc_bias[OUT_CHANNELS],
    data_t tanh_output[BATCH][OUT_CHANNELS])
{

    // block 0
    //data_t local_batch0_conv2d_output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2];
    // conv1
    conv2d_stride(input, block0_weights_conv1, block0_output_conv1);
    //comvine conv2d_stride to batchnorm 2d
    // bn1
    batchnorm2d(block0_output_conv1, block0_output_bn1, block0_gamma1, block0_beta1, block0_mean1, block0_var1);
    // conv2
    conv2d(block0_output_bn1, block0_weights_conv2, block0_output_conv2);
    
    // bn2. Some problem here. 
    batchnorm2d(block0_output_conv2, block0_output_bn2, block0_gamma2, block0_beta2, block0_mean2, block0_var2);
    // downsample
    // conv1
    conv2d_stride_ds(input_local, block0_weights_ds_conv1, block0_output_ds_conv1);
    // bn1
    batchnorm2d(block0_output_ds_conv1, output_local, block0_ds_gamma1, block0_ds_beta1, block0_ds_mean1, block0_ds_var1, false);

    // skip connections and final output of block 0
    Add_4D(block0_output_bn2, output_local, output_block0);
    // need to check output of block 0
    // Block 1
    // conv 1
    conv2d(output_block0, block1_weights_conv1, block1_output_conv1);
    // bn1
    batchnorm2d(block1_output_conv1, block1_output_bn1, block1_gamma1, block1_beta1, block1_mean1, block1_var1);
    // conv2
    //correct till here. 
    //TODO: Check from here
    conv2d(block1_output_bn1, block1_weights_conv2, block1_output_conv2);
    // bn2
    batchnorm2d(block1_output_conv2, block1_output_bn2, block1_gamma2, block1_beta2, block1_mean2, block1_var2);
    average_pooling(block1_output_bn2, output_avgpooling);
    fully_connected(output_avgpooling, fc_weights, fc_bias, tanh_output);


}
// Function to perform 2D convolution with stride =2
//Simple Addition of 4D tensors
// 4D tensor: input[BATCH][IN_CHANNELS][HEIGHT][WIDTH]
void Add_4D(
    data_t input1[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
    data_t input2[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT/2][WIDTH/2])
{
    for (int b = 0; b < BATCH; ++b)
    {
        for (int ic = 0; ic < OUT_CHANNELS; ++ic)
        {
            for (int i = 0; i < HEIGHT/2; ++i)
            {
                for (int j = 0; j < WIDTH/2; ++j)
                {
                    output[b][ic][i][j] = input1[b][ic][i][j] + input2[b][ic][i][j];
                    //add RelU
                    output[b][ic][i][j] = (output[b][ic][i][j] > 0) ? output[b][ic][i][j] : data_t(0); // ReLU activation
                }
            }
        }
    }
}

void conv2d_stride(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t weights[OUT_CHANNELS][IN_CHANNELS][KERNEL_HEIGHT][KERNEL_WIDTH],
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2])
{
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            for (int i = 0; i < HEIGHT; i += 2)
            {
                for (int j = 0; j < WIDTH; j += 2)
                {
                    data_t sum = 0;
                    for (int ic = 0; ic < IN_CHANNELS; ++ic)
                    { // loop over input channels
                        //create local copies of weights, inputs for faster access
                        //data_t weights_local[oc][ic] = weights[oc][ic];
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

                    output[b][oc][i / 2][j / 2] = sum;
                }
            }
        }
    }
}

void conv2d_stride_ds(
    data_t input[BATCH][IN_CHANNELS][HEIGHT][WIDTH],
    data_t weights[OUT_CHANNELS][IN_CHANNELS][DS_KERNEL_HEIGHT][DS_KERNEL_WIDTH], // 🔥 CHANGED: IN_CHANNELS first
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2])
{
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
            // repeat the step below for 512 times
            for (int i = 0; i < HEIGHT; i += 2)
            { // stride 2
                for (int j = 0; j < WIDTH; j += 2)
                {
                    data_t sum = 0;
                    // bring ic loop outside i,j?
                    // go to every 256 channel and multiply over all channel weights
                    for (int ic = 0; ic < IN_CHANNELS; ++ic)
                    {
                        sum += input[b][ic][i][j] * weights[oc][ic][0][0];
                    }
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
    data_t output[BATCH][OUT_CHANNELS][HEIGHT / 2][WIDTH / 2])
{
    for (int b = 0; b < BATCH; ++b)
    {
        for (int oc = 0; oc < OUT_CHANNELS; ++oc)
        {
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
                    output[b][oc][i][j] = sum;
                }
            }
        }
    }
}

// Function to perform batch normalization
void batchnorm2d(
    data_t input[BATCH][CHANNELS][HEIGHT/2][WIDTH/2],
    data_t output[BATCH][CHANNELS][HEIGHT/2][WIDTH/2],
    data_t gamma[CHANNELS],
    data_t beta[CHANNELS],
    data_t mean[CHANNELS],
    data_t var[CHANNELS],
    bool relu // ReLU activation flag
) {
    
    const data_t eps = 1e-5; 
    for (int n = 0; n < BATCH; ++n) {
        //go per batch
        for (int c = 0; c < CHANNELS; ++c) {
            //get weights available per channel
            float inv_std = data_t(1.0) / std::sqrt(var[c] + eps); //(gamma/sqrt(var[c] + eps))
            //float g = gamma[c];
            data_t b_local = beta[c];
            data_t mean_local = mean[c];
            data_t gamma_local = gamma[c];
            for (int h = 0; h < HEIGHT/2; ++h) {
                for (int w = 0; w < WIDTH/2; ++w) {
                    //get value
                    data_t x = input[n][c][h][w];
                    data_t y; //stores output
                    //apply normalization per channel
                    y = gamma_local* (x - mean_local) * data_t(inv_std) + b_local;
                    // //store the new normalized value
                    // output[n][c][h][w] = x_norm + b;
                    //Add ReLU activation
                    if(relu){
                    output[n][c][h][w] = ( y > 0) ? y : data_t(0); // ReLU activation
                    }
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
            //apply tanh
           // output[b][j]=hls::tanh(output[b][j]);
           output[b][j]=output[b][j] * (27 + output[b][j] * output[b][j]) / (27 + 9 * output[b][j] * output[b][j]);

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