def paths(directory, outputfile, benchmarks):

    rodinia2_ = {
        'backprop': directory+'/backprop-rodinia-2.0-ft/4096___data_result_4096_txt/'+outputfile,
        'bfs': directory+'bfs-rodinia-2.0-ft/__data_graph4096_txt___data_graph4096_result_txt/'+outputfile,
        'hotspot': directory+'/hotspot-rodinia-2.0-ft/30_6_40___data_result_30_6_40_txt/'+outputfile,
        'nn': directory+'/nn-rodinia-2.0-ft/__data_filelist_4_3_30_90___data_filelist_4_3_30_90_result_txt/'+outputfile
    }

    rodinia3_ = {
    'b': directory+'/b+tree-rodinia-3.1/file___data_mil_txt_command___data_command_txt/'+outputfile,
    'backprop': directory+'/backprop-rodinia-3.1/65536/'+outputfile,
    'bfs1MW': directory+'/bfs-rodinia-3.1/__data_graph1MW_6_txt/'+outputfile,
    'bfs4096': directory+'/bfs-rodinia-3.1/__data_graph4096_txt/'+outputfile,
    'bfs65536': directory+'/bfs-rodinia-3.1/__data_graph65536_txt/'+outputfile,
    'dwt2d_data_192_bmp': directory+'/dwt2d-rodinia-3.1/__data_192_bmp__d_192x192__f__5__l_3/'+outputfile,
    'dwt2d_data_rgb_bmp__d': directory+'/dwt2d-rodinia-3.1/__data_rgb_bmp__d_1024x1024__f__5__l_3/'+outputfile,
    'gaussian_208': directory+'/gaussian-rodinia-3.1/_f___data_matrix208_txt/'+outputfile,
    'gaussian_mat4': directory+'/gaussian-rodinia-3.1/_f___data_matrix4_txt/'+outputfile,
    'gaussian_s_16': directory+'/gaussian-rodinia-3.1/_s_16/'+outputfile,
    'gaussian_s_256': directory+'/gaussian-rodinia-3.1/_s_256/'+outputfile,
    'gaussian_s_64': directory+'/gaussian-rodinia-3.1/_s_64/'+outputfile,
    'hotspot_1024': directory+'/hotspot-rodinia-3.1/1024_2_2___data_temp_1024___data_power_1024_output_out/'+outputfile,
    'hotspot_512': directory+'/hotspot-rodinia-3.1/512_2_2___data_temp_512___data_power_512_output_out/'+outputfile,
    'hybridsort_500K': directory+'/hybridsort-rodinia-3.1/__data_500000_txt/'+outputfile,
    'hybridsort_r': directory+'/hybridsort-rodinia-3.1/r/'+outputfile,
    'kmeans_28k': directory+'/kmeans-rodinia-3.1/_o__i___data_28k_4x_features_txt/'+outputfile,
    'kmeans_819k': directory+'/kmeans-rodinia-3.1/_o__i___data_819200_txt/'+outputfile,
    'kmeans_oi': directory+'/kmeans-rodinia-3.1/_o__i___data_kdd_cup/'+outputfile,
    'lavaMD': directory+'/lavaMD-rodinia-3.1/_boxes1d_10/'+outputfile,
    'lud_i': directory+'/lud-rodinia-3.1/_i___data_512_dat/'+outputfile,
    'lud_256': directory+'/lud-rodinia-3.1/_s_256__v/'+outputfile,
    'myocyte': directory+'/myocyte-rodinia-3.1/100_1_0/'+outputfile,
    'nn': directory+'/nn-rodinia-3.1/__data_filelist_4__r_5__lat_30__lng_90/'+outputfile,
    'nw': directory+'/nw-rodinia-3.1/2048_10/'+outputfile,
    'streamcluster': directory+'/streamcluster-rodinia-2.0-ft/3_6_16_1024_1024_100_none_output_txt_1___data_result_3_6_16_1024_1024_100_none_1_txt/'+outputfile,
    #'particlefilter_float': directory+'/particlefilter_float-rodinia-3.1/_x_128__y_128__z_10__np_1000/'+outputfile,
    #'particlefilter_naive': directory+'/particlefilter_naive-rodinia-3.1/_x_128__y_128__z_10__np_1000/'+outputfile,
    #'pathfinder': directory+'/pathfinder-rodinia-3.1/100000_100_20___result_txt/'+outputfile,
    'srad_v1': directory+'/srad_v1-rodinia-3.1/100_0_5_502_458/'+outputfile}

    parboil_ = {'parboil-bfs': directory+'/parboil-bfs/_i___data_NY_input_graph_input_dat__o_bfs_NY_out/'+outputfile,
    'parboil-cutcp': directory+'/parboil-cutcp/_i___data_small_input_watbox_sl40_pqr__o_lattice_dat/'+outputfile,
    'parboil-histo': directory+'/parboil-histo/_i___data_default_input_img_bin__o_ref_bmp____20_4/'+outputfile,
    #'parboil-mri-gridding': directory+'/parboil-mri-gridding/_i___data_small_input_small_uks__o_output_txt____32_0/'+outputfile,
    'parboil-mri-q': directory+'/parboil-mri-q/_i___data_small_input_32_32_32_dataset_bin__o_32_32_32_dataset_out/'+outputfile,
    'parboil-sad': directory+'/parboil-sad/_i___data_default_input_reference_bin___data_default_input_frame_bin__o_out_bin/'+outputfile,
    'parboil-sgemm': directory+'/parboil-sgemm/_i___data_medium_input_matrix1_txt___data_medium_input_matrix2t_txt___data_medium_input_matrix2t_txt__o_matrix3_txt/'+outputfile,
    'parboil-spmv': directory+'/parboil-spmv/_i___data_large_input_Dubcova3_mtx_bin___data_large_input_vector_bin__o_Dubcova3_mtx_out/'+outputfile,
    'parboil-stencil': directory+'/parboil-stencil/_i___data_small_input_128x128x32_bin__o_128x128x32_out____128_128_32_100/'+outputfile
    }

    polybench_ = {'polybench-2Dcnn': directory+'/polybench-2DConvolution/NO_ARGS/'+outputfile,
    'polybench-2mm': directory+'/polybench-2mm/NO_ARGS/'+outputfile,
    'polybench-3dconvolution': directory+'/polybench-3DConvolution/NO_ARGS/'+outputfile,
    'polybench-3mm': directory+'/polybench-3mm/NO_ARGS/'+outputfile,
    'polybench-atax': directory+'/polybench-atax/NO_ARGS/'+outputfile,
    'polybench-bicg': directory+'/polybench-bicg/NO_ARGS/'+outputfile,
    'polybench-correlation': directory+'/polybench-correlation/NO_ARGS/'+outputfile,
    'polybench-covariance': directory+'/polybench-covariance/NO_ARGS/'+outputfile,
    'polybench-fdtd2d': directory+'/polybench-fdtd2d/NO_ARGS/'+outputfile,
    'polybench-gemm': directory+'/polybench-gemm/NO_ARGS/'+outputfile,
    'polybench-gsummv': directory+'/polybench-gesummv/NO_ARGS/'+outputfile,
    'polybench-gramschmidt': directory+'/polybench-gramschmidt/NO_ARGS/'+outputfile,
    'polybench-mvt': directory+'/polybench-mvt/NO_ARGS/'+outputfile,
    'polybench-syr2k': directory+'/polybench-syr2k/NO_ARGS/'+outputfile,
    'polybench-syrk': directory+'/polybench-syrk/NO_ARGS/'+outputfile
    }

    cutlass_ = {'cutlass_2560_n_1024_k_2560_sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_1024___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_128_k_2560__sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_128___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_128_k_2560__wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_128___k_2560___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_16_k_2560___sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_16___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_16_k_2560___wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_16___k_2560___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_2560_k_2560_sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_2560___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_32_k_2560___sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_32___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_32_k_2560___wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_32___k_2560___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_512_k_2560__sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_512___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_64_k_2560___sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_64___k_2560___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_64_k_2560___wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_64___k_2560___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_2560_n_7000_k_2560_wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_2560___n_7000___k_2560___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_128_k_4096__sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_128___k_4096___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_128_k_4096__wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_128___k_4096___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_16_k_4096___sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_16___k_4096___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_16_k_4096___wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_16___k_4096___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_32_k_4096___sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_32___k_4096___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_32_k_4096___wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_32___k_4096___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_4096_k_4096_sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_4096___k_4096___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_64_k_4096___sgemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_64___k_4096___kernels_sgemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_64_k_4096___wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_64___k_4096___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile,
    'cutlass_4096_n_7000_k_4096_wmma_gemm': directory+'/cutlass_perf_test/__seed_2020___dist_0____m_4096___n_7000___k_4096___kernels_wmma_gemm_nn____iterations_5___providers_cutlass/'+outputfile
    }

    deepbench_ = {'conv_bench-inference_half_341_79_32_1_32_10_5_0_0_2_2': directory+'/conv_bench/inference_half_341_79_32_1_32_10_5_0_0_2_2/'+outputfile,
'conv_bench-inference_half_341_79_32_2_32_10_5_0_0_2_2': directory+'/conv_bench/inference_half_341_79_32_2_32_10_5_0_0_2_2/'+outputfile,
'conv_bench-inference_half_480_48_1_1_16_3_3_1_1_1_1': directory+'/conv_bench/inference_half_480_48_1_1_16_3_3_1_1_1_1/'+outputfile,
'conv_bench-inference_half_700_161_1_1_32_20_5_0_0_2_2': directory+'/conv_bench/inference_half_700_161_1_1_32_20_5_0_0_2_2/'+outputfile,
'conv_bench-inference_half_700_161_1_2_32_20_5_0_0_2_2': directory+'/conv_bench/inference_half_700_161_1_2_32_20_5_0_0_2_2/'+outputfile,
'conv_bench-train_half_14_14_512_16_512_3_3_1_1_1_1': directory+'/conv_bench/train_half_14_14_512_16_512_3_3_1_1_1_1/'+outputfile,
'conv_bench-train_half_7_7_512_8_512_1_1_0_0_1_1': directory+'/conv_bench/train_half_7_7_512_8_512_1_1_0_0_1_1/'+outputfile,
'conv_bench-train_half_7_7_512_8_512_3_3_1_1_1_1': directory+'/conv_bench/train_half_7_7_512_8_512_3_3_1_1_1_1/'+outputfile,
'conv_bench-train_half_7_7_832_16_128_5_5_2_2_1_1': directory+'/conv_bench/train_half_7_7_832_16_128_5_5_2_2_1_1/'+outputfile,
'conv_bench-train_half_7_7_832_16_256_1_1_0_0_1_1': directory+'/conv_bench/train_half_7_7_832_16_256_1_1_0_0_1_1/'+outputfile,
'conv_bench-tencore-inference_half_341_79_32_1_32_10_5_0_0_2_2': directory+'/conv_bench-tencore/inference_half_341_79_32_1_32_10_5_0_0_2_2/'+outputfile,
'conv_bench-tencore-inference_half_341_79_32_2_32_10_5_0_0_2_2': directory+'/conv_bench-tencore/inference_half_341_79_32_2_32_10_5_0_0_2_2/'+outputfile,
'conv_bench-tencore-inference_half_480_48_1_1_16_3_3_1_1_1_1': directory+'/conv_bench-tencore/inference_half_480_48_1_1_16_3_3_1_1_1_1/'+outputfile,
'conv_bench-tencore-inference_half_700_161_1_1_32_20_5_0_0_2_2': directory+'/conv_bench-tencore/inference_half_700_161_1_1_32_20_5_0_0_2_2/'+outputfile,
'conv_bench-tencore-inference_half_700_161_1_2_32_20_5_0_0_2_2': directory+'/conv_bench-tencore/inference_half_700_161_1_2_32_20_5_0_0_2_2/'+outputfile,
'conv_bench-tencore-train_half_14_14_512_16_512_3_3_1_1_1_1': directory+'/conv_bench-tencore/train_half_14_14_512_16_512_3_3_1_1_1_1/'+outputfile,
'conv_bench-tencore-train_half_7_7_512_8_512_1_1_0_0_1_1': directory+'/conv_bench-tencore/train_half_7_7_512_8_512_1_1_0_0_1_1/'+outputfile,
'conv_bench-tencore-train_half_7_7_512_8_512_3_3_1_1_1_1': directory+'/conv_bench-tencore/train_half_7_7_512_8_512_3_3_1_1_1_1/'+outputfile,
'conv_bench-tencore-train_half_7_7_832_16_128_5_5_2_2_1_1': directory+'/conv_bench-tencore/train_half_7_7_832_16_128_5_5_2_2_1_1/'+outputfile,
'conv_bench-tencore-train_half_7_7_832_16_256_1_1_0_0_1_1': directory+'/conv_bench-tencore/train_half_7_7_832_16_256_1_1_0_0_1_1/'+outputfile,
'gemm_bench-inference_half_35_1500_2560_0_0': directory+'/gemm_bench/inference_half_35_1500_2560_0_0/'+outputfile,
'gemm_bench-inference_half_5124_1500_2048_0_0': directory+'/gemm_bench/inference_half_5124_1500_2048_0_0/'+outputfile,
'gemm_bench-inference_half_512_3000_1536_0_0': directory+'/gemm_bench/inference_half_512_3000_1536_0_0/'+outputfile,
'gemm_bench-inference_half_6144_4_2048_0_0': directory+'/gemm_bench/inference_half_6144_4_2048_0_0/'+outputfile,
'gemm_bench-inference_half_7680_1_2560_0_0': directory+'/gemm_bench/inference_half_7680_1_2560_0_0/'+outputfile,
'gemm_bench-train_half_1760_7000_1760_0_0': directory+'/gemm_bench/train_half_1760_7000_1760_0_0/'+outputfile,
'gemm_bench-train_half_1760_7000_1760_1_0': directory+'/gemm_bench/train_half_1760_7000_1760_1_0/'+outputfile,
'gemm_bench-train_half_1760_7133_1760_0_1': directory+'/gemm_bench/train_half_1760_7133_1760_0_1/'+outputfile,
'gemm_bench-train_half_2048_128_2048_0_0': directory+'/gemm_bench/train_half_2048_128_2048_0_0/'+outputfile,
'gemm_bench-train_half_2048_64_2048_1_0': directory+'/gemm_bench/train_half_2048_64_2048_1_0/'+outputfile,
'gemm_bench-tencore-inference_half_35_1500_2560_0_0': directory+'/gemm_bench-tencore/inference_half_35_1500_2560_0_0/'+outputfile,
'gemm_bench-tencore-inference_half_5124_1500_2048_0_0': directory+'/gemm_bench-tencore/inference_half_5124_1500_2048_0_0/'+outputfile,
'gemm_bench-tencore-inference_half_512_3000_1536_0_0': directory+'/gemm_bench-tencore/inference_half_512_3000_1536_0_0/'+outputfile,
'gemm_bench-tencore-inference_half_6144_4_2048_0_0': directory+'/gemm_bench-tencore/inference_half_6144_4_2048_0_0/'+outputfile,
'gemm_bench-tencore-inference_half_7680_1_2560_0_0': directory+'/gemm_bench-tencore/inference_half_7680_1_2560_0_0/'+outputfile,
'gemm_bench-tencore-train_half_1760_7000_1760_0_0': directory+'/gemm_bench-tencore/train_half_1760_7000_1760_0_0/'+outputfile,
'gemm_bench-tencore-train_half_1760_7000_1760_1_0': directory+'/gemm_bench-tencore/train_half_1760_7000_1760_1_0/'+outputfile,
'gemm_bench-tencore-train_half_1760_7133_1760_0_1': directory+'/gemm_bench-tencore/train_half_1760_7133_1760_0_1/'+outputfile,
'gemm_bench-tencore-train_half_2048_128_2048_0_0': directory+'/gemm_bench-tencore/train_half_2048_128_2048_0_0/'+outputfile,
'gemm_bench-tencore-train_half_2048_64_2048_1_0': directory+'/gemm_bench-tencore/train_half_2048_64_2048_1_0/'+outputfile,
'rnn_bench-inference_half_1024_1_25_lstm': directory+'/rnn_bench/inference_half_1024_1_25_lstm/'+outputfile,
'rnn_bench-inference_half_1536_1_50_lstm': directory+'/rnn_bench/inference_half_1536_1_50_lstm/'+outputfile,
'rnn_bench-inference_half_1536_1_750_gru': directory+'/rnn_bench/inference_half_1536_1_750_gru/'+outputfile,
'rnn_bench-inference_half_256_4_150_lstm': directory+'/rnn_bench/inference_half_256_4_150_lstm/'+outputfile,
'rnn_bench-inference_half_512_1_1_gru': directory+'/rnn_bench/inference_half_512_1_1_gru/'+outputfile,
'rnn_bench-inference_half_512_1_25_lstm': directory+'/rnn_bench/inference_half_512_1_25_lstm/'+outputfile,
'rnn_bench-inference_half_512_2_1_gru': directory+'/rnn_bench/inference_half_512_2_1_gru/'+outputfile,
'rnn_bench-inference_half_512_2_25_lstm': directory+'/rnn_bench/inference_half_512_2_25_lstm/'+outputfile,
'rnn_bench-inference_half_512_4_1_gru': directory+'/rnn_bench/inference_half_512_4_1_gru/'+outputfile,
'rnn_bench-train_half_128_32_15_lstm': directory+'/rnn_bench/train_half_128_32_15_lstm/'+outputfile,
'rnn_bench-train_half_128_32_1_gru': directory+'/rnn_bench/train_half_128_32_1_gru/'+outputfile,
'rnn_bench-train_half_128_32_1_lstm': directory+'/rnn_bench/train_half_128_32_1_lstm/'+outputfile,
'rnn_bench-train_half_128_64_15_lstm': directory+'/rnn_bench/train_half_128_64_15_lstm/'+outputfile,
'rnn_bench-train_half_128_64_1_gru': directory+'/rnn_bench/train_half_128_64_1_gru/'+outputfile,
'rnn_bench-tencore-inference_half_1024_1_25_lstm': directory+'/rnn_bench-tencore/inference_half_1024_1_25_lstm/'+outputfile,
'rnn_bench-tencore-inference_half_1536_1_50_lstm': directory+'/rnn_bench-tencore/inference_half_1536_1_50_lstm/'+outputfile,
'rnn_bench-tencore-inference_half_1536_1_750_gru': directory+'/rnn_bench-tencore/inference_half_1536_1_750_gru/'+outputfile,
'rnn_bench-tencore-inference_half_256_4_150_lstm': directory+'/rnn_bench-tencore/inference_half_256_4_150_lstm/'+outputfile,
'rnn_bench-tencore-inference_half_256_1_150_lstm': directory+'/rnn_bench-tencore/inference_half_256_1_150_lstm/'+outputfile,
'rnn_bench-tencore-inference_half_512_1_1_gru': directory+'/rnn_bench-tencore/inference_half_512_1_1_gru/'+outputfile,
'rnn_bench-tencore-inference_half_512_1_25_lstm': directory+'/rnn_bench-tencore/inference_half_512_1_25_lstm/'+outputfile,
'rnn_bench-tencore-inference_half_512_2_1_gru': directory+'/rnn_bench-tencore/inference_half_512_2_1_gru/'+outputfile,
'rnn_bench-tencore-inference_half_512_2_25_lstm': directory+'/rnn_bench-tencore/inference_half_512_2_25_lstm/'+outputfile,
'rnn_bench-tencore-inference_half_512_4_1_gru': directory+'/rnn_bench-tencore/inference_half_512_4_1_gru/'+outputfile,
'rnn_bench-tencore-train_half_128_32_15_lstm': directory+'/rnn_bench-tencore/train_half_128_32_15_lstm/'+outputfile,
'rnn_bench-tencore-train_half_128_32_1_gru': directory+'/rnn_bench-tencore/train_half_128_32_1_gru/'+outputfile,
'rnn_bench-tencore-train_half_128_32_1_lstm': directory+'/rnn_bench-tencore/train_half_128_32_1_lstm/'+outputfile,
'rnn_bench-tencore-train_half_128_64_15_lstm': directory+'/rnn_bench-tencore/train_half_128_64_15_lstm/'+outputfile,
'rnn_bench-tencore-train_half_128_64_1_gru': directory+'/rnn_bench-tencore/train_half_128_64_1_gru/'+outputfile,
    }

    mlperf_ = {
    'bert_offline':directory+'/BERT/offline/'+outputfile,
    'bert_single_stream':directory+'/BERT/singlestream/'+outputfile,
    'bert_1M': directory+'/BERT/bert_1M/'+outputfile,
    'ssd': directory+'/ssd/tensorflow_offline_32_32/'+outputfile,
    'ssd_pytorch': directory+'/ssd/pytorch/'+outputfile,
    'resnet_50_64b_4000': directory+'/ResNet/ImageNet_64b_4000/'+outputfile,
    'resnet_50_128b_4000': directory+'/ResNet/ImageNet_128b_4000/'+outputfile,
    'resnet_50_256b_4000': directory+'/ResNet/ImageNet_256b_4000/'+outputfile,
    'gnmt_training': directory+'/rnn_translator/'+outputfile,
    '3d-unet': directory+'/3d-unet/'+outputfile
#    'gnmt_inference': directory+'/gnmt/Offline'+outputfile
#    'imagenet-128b-4096': 'G:/Research/Profiles/ImageNet/ImageNet-Offline-128-batch-4096-samples/out.csv',
#    'imagenet-64b-4096': 'G:/Research/Data/Volta/11.1/resnet_pytorch_imagenet_64_4000_offline.ncu-rep'}
    }
    output_dict = {}

    if('rodinia-3.1' in benchmarks):
        output_dict = {**rodinia3_}

    if('rodinia_2.1-ft' in benchmarks):
        outputfile = {**output_dict, **rodinia2_}

    if('parboil' in benchmarks):
        output_dict = {**output_dict, **parboil_}

    if('polybench' in benchmarks):
        output_dict = {**output_dict, **polybench_}

    if('deepbench' in benchmarks):
        output_dict = {**output_dict, **deepbench_}

    if('mlperf' in benchmarks):
        output_dict = {**output_dict, **mlperf_}

    if('cutlass' in benchmarks):
        output_dict = {**output_dict, **cutlass_}


    return output_dict