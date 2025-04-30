open_project project_5

# set top function of the HLS design
set_top ResNet

# add source file
add_files top_new.cpp

# add testbench
add_files -tb host.cpp


# add data file
add_files -tb input_1.bin
add_files -tb layer4_block0_conv1_weights.bin
add_files -tb layer4_block0_batch_norm_1_beta.bin
add_files -tb layer4_block0_batch_norm_1_gamma.bin
add_files -tb block0_output.bin
add_files -tb layer4_block0_batch_norm_1_mean.bin
add_files -tb layer4_block0_batch_norm_1_var.bin
add_files -tb layer4_block0_conv2_weights.bin
add_files -tb layer4_block0_batch_norm_2_beta.bin
add_files -tb layer4_block0_batch_norm_2_gamma.bin
add_files -tb layer4_block0_batch_norm_2_mean.bin
add_files -tb layer4_block0_batch_norm_2_var.bin
add_files -tb layer4_block0_downsample_weights.bin
add_files -tb layer4_block0_downsample_batch_norm_gamma.bin
add_files -tb layer4_block0_downsample_batch_norm_beta.bin
add_files -tb layer4_block0_downsample_batch_norm_mean.bin
add_files -tb layer4_block0_downsample_batch_norm_var.bin
add_files -tb layer4_block1_conv1_weights.bin
add_files -tb layer4_block1_batch_norm_1_beta.bin
add_files -tb layer4_block1_batch_norm_1_gamma.bin
#add_files -tb layer4_block1_batch_norm_output.bin
add_files -tb layer4_block1_batch_norm_1_mean.bin
add_files -tb layer4_block1_batch_norm_1_var.bin
add_files -tb layer4_block1_conv2_weights.bin
add_files -tb layer4_block1_batch_norm_2_beta.bin
add_files -tb layer4_block1_batch_norm_2_gamma.bin
add_files -tb layer4_block1_batch_norm_2_mean.bin
add_files -tb layer4_block1_batch_norm_2_var.bin
add_files -tb fc_bias.bin
add_files -tb fc_weights.bin
add_files -tb output_1.bin

open_solution "solution1"

# FPGA part and clock configuration
set_part {xcku115-flva2104-3-e}

# default frequency is 100 MHz
#create_clock -period 4 -name default

# C synthesis for HLS design, generating RTL
csynth_design

# C/RTL co-simulation; can be commented if not needed
cosim_design

# export generated RTL as an IP; can be commented if not needed
# export_design -format ip_catalog -flow syn

exit