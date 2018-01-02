############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2017 Xilinx, Inc. All Rights Reserved.
############################################################
set_directive_pipeline "bfnn1/L2"
set_directive_pipeline "bfnn2/L2"
set_directive_pipeline "bfnn3/L2"
set_directive_pipeline "bfnn4/L2"
set_directive_pipeline "sf/hl2"
set_directive_pipeline "bfnn4_dw/dwL3"
set_directive_unroll -factor 4 "bfnn2_dw/dwL4"
set_directive_pipeline "bfnn1_dw/dwL3"
set_directive_pipeline "bfnn4_dx/dxL3"
set_directive_pipeline "bfnn2_dx/dxL4"
set_directive_pipeline "bfnn3_dx/dxL4"
set_directive_inline "learn_model_v_1"
set_directive_interface -mode s_axilite "learn_model_v_1" in
set_directive_interface -mode s_axilite "learn_model_v_1" weight1
set_directive_interface -mode s_axilite "learn_model_v_1" weight2
set_directive_interface -mode s_axilite "learn_model_v_1" weight3
set_directive_interface -mode s_axilite "learn_model_v_1" weight4
set_directive_interface -mode s_axilite "learn_model_v_1" T
set_directive_interface -mode s_axilite "learn_model_v_1" bias1
set_directive_interface -mode s_axilite "learn_model_v_1" bias2
set_directive_interface -mode s_axilite "learn_model_v_1" bias3
set_directive_interface -mode s_axilite "learn_model_v_1" bias4
set_directive_interface -mode s_axilite "learn_model_v_1" out
set_directive_interface -mode s_axilite "learn_model_v_1" k
set_directive_interface -mode s_axilite "learn_model_v_1"
set_directive_array_partition -type cyclic -factor 98 -dim 1 "learn_model_v_1" dw1
set_directive_array_partition -type cyclic -factor 15 -dim 1 "learn_model_v_1" dw2
set_directive_array_partition -type cyclic -factor 15 -dim 1 "learn_model_v_1" dw3
set_directive_array_partition -type cyclic -factor 15 -dim 1 "learn_model_v_1" dw4
set_directive_pipeline "learn_model_v_1/dw4L2"
set_directive_pipeline "learn_model_v_1/dw3L2"
set_directive_pipeline "learn_model_v_1/dw2L2"
set_directive_pipeline "learn_model_v_1/dw1L2"
set_directive_inline "learn_model_v_2"
set_directive_interface -mode s_axilite "learn_model_v_2" in
set_directive_interface -mode s_axilite "learn_model_v_2" weight1
set_directive_interface -mode s_axilite "learn_model_v_2" weight2
set_directive_interface -mode s_axilite "learn_model_v_2" weight3
set_directive_interface -mode s_axilite "learn_model_v_2" weight4
set_directive_interface -mode s_axilite "learn_model_v_2" T
set_directive_interface -mode s_axilite "learn_model_v_2" bias1
set_directive_interface -mode s_axilite "learn_model_v_2" bias2
set_directive_interface -mode s_axilite "learn_model_v_2" bias3
set_directive_interface -mode s_axilite "learn_model_v_2" bias4
set_directive_interface -mode s_axilite "learn_model_v_2" out
set_directive_interface -mode s_axilite "learn_model_v_2" k
set_directive_interface -mode s_axilite "learn_model_v_2"
set_directive_array_partition -type cyclic -factor 98 -dim 1 "learn_model_v_2" dw1
set_directive_array_partition -type cyclic -factor 15 -dim 1 "learn_model_v_2" dw2
set_directive_array_partition -type cyclic -factor 15 -dim 1 "learn_model_v_2" dw3
set_directive_array_partition -type cyclic -factor 15 -dim 1 "learn_model_v_2" dw4
set_directive_pipeline "learn_model_v_2/dw4L2"
set_directive_pipeline "learn_model_v_2/dw3L2"
set_directive_pipeline "learn_model_v_2/dw2L2"
set_directive_pipeline "learn_model_v_2/dw1L2"
