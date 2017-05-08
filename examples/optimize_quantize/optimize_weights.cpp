#include "caffe/caffe.hpp"
#include "caffe/layers/conv_q_layer.hpp"
#include "caffe/layers/memory_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/program_options.hpp>

#include <memory>

using namespace caffe;
using namespace boost::program_options;

void build_A(int, int, int, int, int, int, int, int, int, int, int,
				 const float*, const int*, float*);

void build_bsdiffs(int, int, int, int, int, int, int, int, int, int, int,
						 int, const float*, const float*, const float*, float*);

void multiply(int, int, int, const float*, const float*, float*);

const int TOTAL_BITS = 8 * sizeof(float);
float STUB = 0.f;
float ALPHA = 1.f;

float calc_loss(Blob<float>* top_src, Blob<float>* top_q) {
	float error = 0.f;
	const float* src = top_src->cpu_data();
	const float* quant = top_q->cpu_data();
	for (int i = 0; i < top_src->count(); ++i) {
		float diff = src[i] - quant[i];
		error += diff * diff;
	}
	error /= top_src->count();
	return error;
}

float calc_norm(Blob<float>* top) {
	float norm = 0.f;
	const float* data = top->cpu_data();
	for (int i = 0; i < top->count(); ++i) {
		norm += data[i];
	}
	return norm;
}

void extract_b(const int* B_hash, const int BITS, int k, int* B, int b_size) {
	const int REST_BITS = TOTAL_BITS - BITS;

	for (int i = 0, total_bit_shift = 0; i < b_size; ++i, total_bit_shift += BITS) {
		int byte_shift = total_bit_shift / TOTAL_BITS;
		int bit_shift = total_bit_shift % TOTAL_BITS;
		int shift = REST_BITS - bit_shift;
		B[i] = (int)((shift < 0 ? B_hash[byte_shift] << -shift | B_hash[byte_shift + 1] >> (TOTAL_BITS + shift) :
										  B_hash[byte_shift] >> shift) & (k - 1));
	}
}

shared_ptr<Net<float> > create_one_slice_net(int num_images, int width, int height, int out_channels, int pad, int stride, int kernel_size, int k, int m) {
	shared_ptr<Net<float> > net_q_slice;
	NetParameter param;

	LayerParameter* memory_layer_param = param.add_layer();
	memory_layer_param->set_type("MemoryData");
	memory_layer_param->add_top("data");
	memory_layer_param->add_top("fakelabel");
	MemoryDataParameter* memory_param = memory_layer_param->mutable_memory_data_param();
	memory_param->set_batch_size(num_images);
	memory_param->set_channels(m);
	memory_param->set_height(height);
	memory_param->set_width(width);

	LayerParameter* q_layer_param = param.add_layer();
	q_layer_param->set_type("Convolution");
	ConvolutionParameter* slice_conv_param = q_layer_param->mutable_convolution_param();
	slice_conv_param->add_kernel_size(kernel_size);
	slice_conv_param->add_stride(stride);
	slice_conv_param->add_pad(pad);
	slice_conv_param->set_num_output(out_channels);
	slice_conv_param->set_engine(ConvolutionParameter_Engine_QUANT);
	slice_conv_param->set_k(k);
	slice_conv_param->set_m(m);
	q_layer_param->add_bottom("data");
	q_layer_param->add_top("conv");
	net_q_slice.reset(new Net<float>(param));

	return net_q_slice;
}

void optimize_conv_layer(shared_ptr<ConvolutionQLayer<float> > q_layer, Blob<float>* input_data, Blob<float>* src_output_data, Blob<float>* q_output_data) {
	int num_images = input_data->shape(0);
	int in_channels = input_data->shape(1);
	int in_height = input_data->shape(2);
	int in_width = input_data->shape(3);
	int input_image_size = in_channels * in_height * in_width;

	string layer_name = q_layer->layer_param().name();
	ConvolutionParameter q_conv_param = q_layer->layer_param().convolution_param();
	int out_channels  = q_conv_param.num_output();
	int m = q_conv_param.m();
	int k = q_conv_param.k();
	int kernel_size = q_conv_param.kernel_size()[0];
	int stride = q_conv_param.stride_size() > 0 ? q_conv_param.stride()[0] : 1;
	int pad = q_conv_param.pad_size() > 0 ? q_conv_param.pad()[0] : 0;
	int slice_count = in_channels / m;

	int out_height = q_output_data->shape(2);
	int out_width = q_output_data->shape(3);
	int out_size = num_images * out_channels * out_height * out_width;

	const float* input_data_ = input_data->cpu_data();
	shared_ptr<Blob<float> > input_slice(new Blob<float>(input_data->shape(0), m, input_data->shape(2), input_data->shape(3)));
	const int in_slice_image_size = input_slice->count(1);

	const vector<int> output_shape = q_output_data->shape();
	shared_ptr<Blob<float> > output_wo_slice(new Blob<float>(output_shape));
	shared_ptr<Blob<float> > output_slice(new Blob<float>(output_shape));
	shared_ptr<Blob<float> > output_slice_k(new Blob<float>(output_shape));
	shared_ptr<Blob<float> > output_slice_pos(new Blob<float>(output_shape));
	const float* src_output_ = src_output_data->cpu_data();
	const float* q_output_ = q_output_data->cpu_data();

	shared_ptr<Blob<float> > R(new Blob<float>(output_shape));
	shared_ptr<Blob<float> > Q(new Blob<float>(output_shape));
	shared_ptr<Blob<float> > P(new Blob<float>(output_shape));
	shared_ptr<Blob<float> > A(new Blob<float>(out_size, m, 1, 1));
	float* A_ = A->mutable_gpu_data();
	shared_ptr<Blob<float> > A_trans(new Blob<float>(m, out_size, 1, 1));
	float* A_trans_ = A_trans->mutable_gpu_data();
	shared_ptr<Blob<float> > A_sq(new Blob<float>(m, m, 1, 1));
	float** A_sq_;
	cudaMallocManaged(&A_sq_, sizeof(float*));
	A_sq_[0] = A_sq->mutable_gpu_data();

	shared_ptr<Blob<float> > A_inv(new Blob<float>(m, out_size, 1, 1));
	float* A_inv_ = A_inv->mutable_gpu_data();
	shared_ptr<Blob<float> > A_sq_inv(new Blob<float>(m, m, 1, 1));
	float** A_sq_inv_;
	cudaMallocManaged(&A_sq_inv_, sizeof(float*));
	A_sq_inv_[0] = A_sq_inv->mutable_gpu_data();
	int* pivot;
	cudaMalloc((void**)&pivot, m * sizeof(int));
	int* info;
	cudaMalloc((void**)&info, sizeof(int));

	const int BITS = (int)log2(k);
	const int REST_BITS = TOTAL_BITS - BITS;
	shared_ptr<Blob<float> > d_slice(new Blob<float>(k, m, 1, 1));
	float* d_slice_ = d_slice->mutable_cpu_data();
	//CHECK_EQ((BITS * out_channels) % TOTAL_BITS, 0) << "log2(k) * num_output of "
	//	<< layer_name << " layer has to be multiple of " << TOTAL_BITS << " (" << out_channels * BITS << ").";
	int b_slice_size = (out_channels * kernel_size * kernel_size * BITS + TOTAL_BITS - 1) / TOTAL_BITS;
	int b_slice_rest = out_channels * kernel_size * kernel_size * BITS % TOTAL_BITS;
	if (b_slice_rest == 0) b_slice_rest = TOTAL_BITS;
	shared_ptr<Blob<float> > b_hash_slice(new Blob<float>(b_slice_size, 1, 1, 1));
	float* b_hash_slice_ = b_hash_slice->mutable_cpu_data();

	float* q_d_ = q_layer->blobs()[0]->mutable_cpu_data();
	shared_ptr<Blob<float> > bias(new Blob<float>(q_layer->blobs()[1]->shape()));
	caffe_set(bias->count(), 0.f, bias->mutable_cpu_data());
	float* q_b_hash_ = q_layer->blobs()[2]->mutable_cpu_data();
	int b_unpacked_size = slice_count * out_channels * kernel_size * kernel_size;
	shared_ptr<Blob<int> > b_unpacked(new Blob<int>(b_unpacked_size, 1, 1, 1));
	extract_b((const int*)q_b_hash_, BITS, k, b_unpacked->mutable_cpu_data(), b_unpacked_size);
	shared_ptr<Blob<float> > bs_diffs(new Blob<float>(out_channels, k, 1, 1));

	shared_ptr<Net<float> > net_q_slice = create_one_slice_net(num_images, in_width, in_height, out_channels, pad, stride, kernel_size, k, m);
	shared_ptr<MemoryDataLayer<float> > memory_layer = boost::dynamic_pointer_cast<MemoryDataLayer<float> >(net_q_slice->layers()[0]);
	shared_ptr<Layer<float> > q_slice_layer = net_q_slice->layers()[1];

	//float previous_loss = 0.f;
	int* q_b_hash_slice_ = (int*)q_b_hash_;
	int b_slice_cur_rest = 0;
	for (int slice = 0; slice < slice_count; ++slice) {
		float* q_d_slice_ = q_d_ + slice * k * m;
		for (int i = 0; i < k * m; ++i)
			d_slice_[i] = q_d_slice_[i];
		/*q_b_hash_slice_ = q_b_hash_ + slice * b_slice_size;
		for (int i = 0; i < b_slice_size; ++i) {
			b_hash_slice_[i] = q_b_hash_slice_[i];
		}*/
		for (int i = 0; i < b_slice_size; ++i) {
			if (b_slice_cur_rest == 0) {
				b_hash_slice_[i] = reinterpret_cast<float&>(q_b_hash_slice_[i]);
			}
			else {
				int b_bits = q_b_hash_slice_[i] << b_slice_cur_rest;
				if (i < b_slice_size - 1 || b_slice_cur_rest + b_slice_rest > TOTAL_BITS) {
					b_bits |= q_b_hash_slice_[i + 1] >> (TOTAL_BITS - b_slice_cur_rest);
				}
				b_hash_slice_[i] = reinterpret_cast<float&>(b_bits);
			}
		}

		int* b_unpacked_slice = b_unpacked->mutable_cpu_data() + slice * out_channels * kernel_size * kernel_size;
		const int* b_unpacked_slice_gpu = b_unpacked->gpu_data() + slice * out_channels * kernel_size * kernel_size;
		shared_ptr<Blob<float> > q_slice_layer_blobs[] = {d_slice, bias, b_hash_slice};
		q_slice_layer->blobs() = vector<shared_ptr<Blob<float> > >(q_slice_layer_blobs, q_slice_layer_blobs + 3);

		const float* q_input_slice_ = input_data_ + slice * in_slice_image_size;
		for (int n = 0; n < num_images; ++n) {
			float* input_slice_n = input_slice->mutable_cpu_data() + n * in_slice_image_size;
			const float* q_input_slice_n = q_input_slice_ + n * input_image_size;
			for(int j = 0; j < in_slice_image_size; ++j)
				input_slice_n[j] = q_input_slice_n[j];
		}

		// HACK: stupid caffe doesn't allow to change layer_param
		const_cast<LayerParameter&>(q_slice_layer->layer_param()).mutable_convolution_param()->set_conv_mode(ConvolutionParameter_ConvMode_LOWERED_GEMM);
		memory_layer->Reset(input_slice->mutable_cpu_data(), &STUB, num_images);
		const vector<Blob<float>*>& output_slice_vec_k = net_q_slice->Forward(&STUB);
		output_slice->CopyFrom(*output_slice_vec_k[0], false, true);
		float* output_slice_ = output_slice->mutable_cpu_data();

		caffe_sub(out_size, q_output_, output_slice_, output_wo_slice->mutable_cpu_data());
		caffe_sub(out_size, src_output_, output_wo_slice->cpu_data(), R->mutable_cpu_data());

		LOG(INFO) << "\n";
		// HACK: stupid caffe doesn't allow to change layer_param
		const_cast<LayerParameter&>(q_slice_layer->layer_param()).mutable_convolution_param()->set_conv_mode(ConvolutionParameter_ConvMode_SINGLE_K);
		for (int i = 0; i < k; ++i) {
			// HACK: stupid caffe doesn't allow to change layer_param
			const_cast<LayerParameter&>(q_slice_layer->layer_param()).mutable_convolution_param()->set_for_k(i);
			const vector<Blob<float>*>& output_slice_k_vec = net_q_slice->Forward(&STUB);
			output_slice_k->CopyFrom(*output_slice_k_vec[0], false, true);

			caffe_sub(out_size, output_slice_, output_slice_k->cpu_data(), output_slice_);
			caffe_sub(out_size, R->cpu_data(), output_slice_, Q->mutable_cpu_data());

			//LOG(INFO) << "Before build";
			build_A(num_images, out_channels, out_height, out_width, i, m, kernel_size, in_height, in_width,
					  pad, stride, input_slice->gpu_data(), b_unpacked_slice_gpu, A->mutable_gpu_data());
			//LOG(INFO) << "Before first multiplication";

			/*cublasSgeam(Caffe::cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N, out_size, m, &ALPHA, A_, m, &STUB, A_, out_size, A_trans_, out_size);
			multiply(m, out_size, m, A_trans_, A_trans_, A_sq->mutable_gpu_data());
			CUBLAS_CHECK(cublasSgetrfBatched(Caffe::cublas_handle(), m, A_sq_, m, pivot, info, 1));
			CUBLAS_CHECK(cublasSgetriBatched(Caffe::cublas_handle(), m, (const float**)A_sq_, m, pivot, A_sq_inv_, m, info, 1));
			multiply(m, m, out_size, A_sq_inv->gpu_data(), A_, A_inv->mutable_gpu_data());*/

			/*caffe_gpu_gemm(CblasTrans, CblasNoTrans, m, m, out_size, 1.f, A_, A_, 0.f, A_sq->mutable_gpu_data());
			CUBLAS_CHECK(cublasSgetrfBatched(Caffe::cublas_handle(), m, A_sq_, m, pivot, info, 1));
			CUBLAS_CHECK(cublasSgetriBatched(Caffe::cublas_handle(), m, (const float**)A_sq_, m, pivot, A_sq_inv_, m, info, 1));
			caffe_gpu_gemm(CblasNoTrans, CblasTrans, m, out_size, m, 1.f, A_sq_inv->gpu_data(), A_, 0.f, A_inv->mutable_gpu_data());
			caffe_gpu_gemv(CblasNoTrans, m, out_size, 1.f, A_inv_, Q->gpu_data(), 0.f, d_slice->mutable_gpu_data() + i * m);*/

			caffe_cpu_gemm(CblasTrans, CblasNoTrans, m, m, out_size, 1.f, A->cpu_data(), A->cpu_data(), 0.f, A_sq->mutable_cpu_data());
			cv::Mat A_sq_inv_cv(m, m, CV_32FC1, const_cast<float*>(A_sq->cpu_data()));
			cv::invert(A_sq_inv_cv, A_sq_inv_cv);
			float* A_sq_inv_cv_ = (float*)A_sq_inv_cv.ptr();
			caffe_cpu_gemm(CblasNoTrans, CblasTrans, m, out_size, m, 1.f, A_sq_inv_cv_, A->cpu_data(), 0.f, A_inv->mutable_cpu_data());
			caffe_cpu_gemv(CblasNoTrans, m, out_size, 1.f, A_inv->cpu_data(), Q->cpu_data(), 0.f, d_slice->mutable_cpu_data() + i * m);
			//LOG(INFO) << "After all";

			const float* d_slice_i = d_slice->cpu_data() + i * m;
			float* q_d_slice_i = q_d_slice_ + i * m;
			for (int j = 0; j < m; ++j)
				q_d_slice_i[j] = d_slice_i[j];

			q_slice_layer->blobs() = vector<shared_ptr<Blob<float> > >(q_slice_layer_blobs, q_slice_layer_blobs + 3);
			const vector<Blob<float>*>& output_slice_k_fixed_vec = net_q_slice->Forward(&STUB);
			output_slice_k->CopyFrom(*output_slice_k_fixed_vec[0], false, true);
			caffe_add(out_size, output_slice_, output_slice_k->cpu_data(), output_slice_);

			//q_layer->Forward(vector<Blob<float>* >(1, input_data), vector<Blob<float>* >(1, q_output_data));
			//float loss = calc_loss(src_output_data, q_output_data);
			//LOG(INFO) << "Loss for " << i << " (k): " << loss << " " << (loss - previous_loss < 0 ? "-" : "+");
			//previous_loss = loss;
			//LOG(INFO) << "K " << i << " iteration";
		}
		q_layer->Forward(vector<Blob<float>* >(1, input_data), vector<Blob<float>* >(1, q_output_data));
		LOG(INFO) << "Loss for " << slice << " (k): " << calc_loss(src_output_data, q_output_data);

		// HACK: stupid caffe doesn't allow to change layer_param
		q_slice_layer->blobs() = vector<shared_ptr<Blob<float> > >(q_slice_layer_blobs, q_slice_layer_blobs + 3);
		const_cast<LayerParameter&>(q_slice_layer->layer_param()).mutable_convolution_param()->set_conv_mode(ConvolutionParameter_ConvMode_LOWERED_GEMM);
		const vector<Blob<float>*>& output_slice_vec_pos = net_q_slice->Forward(&STUB);
		output_slice->CopyFrom(*output_slice_vec_pos[0], false, true);
		// HACK: stupid caffe doesn't allow to change layer_param
		const_cast<LayerParameter&>(q_slice_layer->layer_param()).mutable_convolution_param()->set_conv_mode(ConvolutionParameter_ConvMode_SINGLE_POS);
		for (int i = 0; i < kernel_size * kernel_size; ++i) {
			// HACK: stupid caffe doesn't allow to change layer_param
			const_cast<LayerParameter&>(q_slice_layer->layer_param()).mutable_convolution_param()->set_for_pos(i);
			const vector<Blob<float>*>& output_slice_pos_vec = net_q_slice->Forward(&STUB);
			output_slice_pos->CopyFrom(*output_slice_pos_vec[0], false, true);

			caffe_sub(out_size, output_slice_, output_slice_pos->cpu_data(), output_slice_);
			caffe_sub(out_size, R->cpu_data(), output_slice_, P->mutable_cpu_data());
			build_bsdiffs(num_images, out_channels, out_height, out_width, k, m, kernel_size,
							  in_height, in_width, pad, stride, i, input_slice->gpu_data(),
							  d_slice->gpu_data(), P->gpu_data(), bs_diffs->mutable_gpu_data());

			for (int ct = 0; ct < out_channels; ++ct) {
				int min_k = 0;
				const float* bs_diffs_ct = bs_diffs->cpu_data() + ct * k;
				float min = bs_diffs_ct[0];
				for (int j = 1; j < k; ++j) {
					if (min > bs_diffs_ct[j]) {
						min = bs_diffs_ct[j];
						min_k = j;
					}
				}

				int b_unpacked_index = ct * kernel_size * kernel_size + i;
				int total_shift = b_unpacked_index * BITS / TOTAL_BITS;
				int bit_shift = REST_BITS - b_unpacked_index * BITS % TOTAL_BITS;

				int* b_hash_slice_ct = (int*)(b_hash_slice_ + total_shift);
				if (bit_shift >= 0) {
					b_hash_slice_ct[0] &= ~((k - 1) << bit_shift);
					b_hash_slice_ct[0] |= min_k << bit_shift;
				}
				else {
					b_hash_slice_ct[0] &= ~((k - 1) >> -bit_shift);
					b_hash_slice_ct[0] |= min_k >> -bit_shift;
					b_hash_slice_ct[1] &= ~0 >> -bit_shift;
					b_hash_slice_ct[1] |= min_k << (TOTAL_BITS + bit_shift);
					//q_b_hash_slice_[total_shift + 1] = reinterpret_cast<float&>(b_hash_slice_ct[1]);
				}
				//q_b_hash_slice_[total_shift] = reinterpret_cast<float&>(b_hash_slice_ct[0]);

				total_shift = (b_unpacked_index * BITS + b_slice_cur_rest) / TOTAL_BITS;
				bit_shift = REST_BITS - (b_unpacked_index * BITS + b_slice_cur_rest) % TOTAL_BITS;
				b_hash_slice_ct = q_b_hash_slice_ + total_shift;
				if (bit_shift >= 0) {
					b_hash_slice_ct[0] &= ~((k - 1) << bit_shift);
					b_hash_slice_ct[0] |= min_k << bit_shift;
				}
				else {
					b_hash_slice_ct[0] &= ~((k - 1) >> -bit_shift);
					b_hash_slice_ct[0] |= min_k >> -bit_shift;
					b_hash_slice_ct[1] &= ~0 >> -bit_shift;
					b_hash_slice_ct[1] |= min_k << (TOTAL_BITS + bit_shift);
				}

				b_unpacked_slice[b_unpacked_index] = min_k;
			}

			q_slice_layer->blobs() = vector<shared_ptr<Blob<float> > >(q_slice_layer_blobs, q_slice_layer_blobs + 3);
			const vector<Blob<float>*>& output_slice_pos_fixed_vec = net_q_slice->Forward(&STUB);
			output_slice_pos->CopyFrom(*output_slice_pos_fixed_vec[0], false, true);
			caffe_add(out_size, output_slice_, output_slice_pos->cpu_data(), output_slice_);

			// TODO: move it after cycle
			//q_layer->Forward(vector<Blob<float>* >(1, input_data), vector<Blob<float>* >(1, q_output_data));
			//net_q->layers()[index + 1]->Forward(net_q->bottom_vecs()[index + 1], net_q->top_vecs()[index + 1]);
			//LOG(INFO) << "Loss for " << i << " (pos): " << calc_loss(src_output_data, q_output_data);
			//LOG(INFO) << "Pos " << i << " iteration";
		}

		b_slice_cur_rest += b_slice_rest;
		q_b_hash_slice_ += b_slice_size - 1;
		if (b_slice_cur_rest >= TOTAL_BITS) {
			b_slice_cur_rest -= TOTAL_BITS;
			++q_b_hash_slice_;
		}

		q_layer->Forward(vector<Blob<float>* >(1, input_data), vector<Blob<float>* >(1, q_output_data));
		LOG(INFO) << "Loss for " << slice << " (pos): " << calc_loss(src_output_data, q_output_data);
	}
	cudaFree(A_sq_);
	cudaFree(A_sq_inv_);
	cudaFree(pivot);
	cudaFree(info);
}

int main(int argc, char** argv) {
	options_description desc("Options");
	desc.add_options()
		("help,h", "Help screen")
		("model_src", value<string>()->default_value("train_val"), "Source model name (without prototxt)")
		("weights_src", value<string>(), "Source model weights (without caffemodel, REQUIRED)")
		("model_q", value<string>()->default_value("compressed"), "Quantized model name (without prototxt)")
		("weights_q", value<string>(), "Quantized model weights. Last model with weights_out prefix is used without this parameter")
		("weights_out", value<string>()->default_value("optimized"), "Optimized quantized model weights save prefix")
		("iterations", value<int>()->default_value(50), "Number of optimization iterations")
		("skip", value<int>(), "Skip N batches")
		("exclude", value<vector<string> >()->multitoken()->zero_tokens()->composing(), "Layers to exclude from optimization (alternative to include)")
		("include", value<vector<string> >()->multitoken()->zero_tokens()->composing(), "Layers to include to optimization (alternative to exclude)");
	variables_map vm;
	store(parse_command_line(argc, argv, desc), vm);

	if (vm.count("help") || !vm.count("weights_src")) {
		LOG(ERROR) << desc;
	}

	shared_ptr<Net<float> > net_src;
	shared_ptr<Net<float> > net_q;

	string model_file_src = vm["model_src"].as<string>();
	string weights_file_src = vm["weights_src"].as<string>();
	string model_file_q = vm["model_q"].as<string>();
	string weights_file_q;
	string weights_file_out = vm["weights_out"].as<string>();
	int iterations = vm["iterations"].as<int>();
	vector<string> exc_layers = vm.count("exclude") ? vm["exclude"].as<vector<string> >() : vector<string>();
	vector<string> inc_layers = vm.count("include") ? vm["include"].as<vector<string> >() : vector<string>();

	net_src.reset(new Net<float>(model_file_src + ".prototxt", TEST));
	net_src->CopyTrainedLayersFrom(weights_file_src + ".caffemodel");
	net_q.reset(new Net<float>(model_file_q + ".prototxt", TEST));
	int layers_size = net_src->layers().size();

	int start_it = 0;
	if (vm.count("weights_q")) {
		weights_file_q = vm["weights_q"].as<string>() + ".caffemodel";
	}
	else {
		string weights_it = weights_file_out + ".caffemodel";
		while(1) {
			weights_file_q = weights_it;

			stringstream weights_it_ss;
			weights_it_ss << weights_file_out << "_" << start_it << ".caffemodel";
			weights_it = weights_it_ss.str();
			if (!boost::filesystem::exists(weights_it))
				break;
			++start_it;
		}
	}

	int skip = vm.count("skip") ? vm["skip"].as<int>() : start_it;
	if (boost::filesystem::exists(weights_file_q)) {
		net_q->CopyTrainedLayersFrom(weights_file_q);
		LOG(INFO) << "Loaded weights from " << weights_file_q;

		for (int it = 0; it < skip; ++it) {
			net_src->ForwardFromTo(0, 1);
			net_q->ForwardFromTo(0, 1);
			LOG(INFO) << "Skipping " << it << " batch";
		}
	}
	else {
		LOG(ERROR) << "No model with name " << weights_file_q << " exists";
	}

	LOG(INFO) << "Starting optimization with " << iterations << " iterations";
	for (int it = start_it; it < start_it + iterations; ++it) {
		net_src->Forward();

		LOG(INFO) << "Starting " << it << " optimization iteration";
		int prev_index = 0;
		for (int l_index = 0; l_index < layers_size; ++l_index) {
			shared_ptr<Layer<float> > layer = net_q->layers()[l_index];
			string layer_name = layer->layer_param().name();
			if (layer->layer_param().convolution_param().engine() == ConvolutionParameter_Engine_QUANT) {
				net_q->ForwardFromTo(prev_index, l_index + 1);
				prev_index = l_index;

				shared_ptr<ConvolutionQLayer<float> > q_layer = boost::dynamic_pointer_cast<ConvolutionQLayer<float> >(layer);
				Blob<float>* input_data = net_q->bottom_vecs()[l_index][0];
				Blob<float>* src_output_data = net_src->top_vecs()[l_index][0];
				Blob<float>* q_output_data = net_q->top_vecs()[l_index][0];
				Blob<float>* src_relu_output_data = net_src->top_vecs()[l_index + 1][0];
				Blob<float>* q_relu_output_data = net_q->top_vecs()[l_index + 1][0];

				if (find(inc_layers.begin(), inc_layers.end(), layer_name) != inc_layers.end() ||
						inc_layers.size() == 0 && find(exc_layers.begin(), exc_layers.end(), layer_name) == exc_layers.end()) {
					LOG(INFO) << "Optimization for layer " << layer_name << " (" << it << ")";
					LOG(INFO) << "Initial loss: " << calc_loss(src_output_data, q_output_data);
					LOG(INFO) << "Initial ReLU Loss: " << calc_loss(src_relu_output_data, q_relu_output_data);

					optimize_conv_layer(q_layer, input_data, src_output_data, q_output_data);
					net_q->ForwardFromTo(l_index, l_index + 1);

					LOG(INFO) << "Final loss: " << calc_loss(src_output_data, q_output_data) << " (" << it << ")";
					LOG(INFO) << "Final ReLU Loss: " << calc_loss(src_relu_output_data, q_relu_output_data) << " (" << it << ")";
				}
				else {
					LOG(INFO) << "Skipping not optimized layer " << layer_name << " (" << it << ")";
					LOG(INFO) << "Loss: " << calc_loss(src_output_data, q_output_data) << " (" << it << ")";
					LOG(INFO) << "ReLU Loss: " << calc_loss(src_relu_output_data, q_relu_output_data) << " (" << it << ")";
				}
			}
		}
		net_q->ForwardFromTo(prev_index + 1, layers_size - 1);
		LOG(INFO) << "Accuracy after " << it << " iteration: " << net_q->top_vecs()[layers_size - 1][0]->cpu_data()[0];

		NetParameter net_param;
		net_q->ToProto(&net_param, false);
		stringstream weights_out_it_ss;
		weights_out_it_ss << weights_file_out << "_" << it << ".caffemodel";
		string weights_out_it = weights_out_it_ss.str();
		WriteProtoToBinaryFile(net_param, weights_out_it);
		LOG(INFO) << "Optimized NN saved to file " << weights_out_it;
	}
	LOG(INFO) << "Optimization process has been finished";
}
