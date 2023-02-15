#ifdef DEBUG
# define DEBUG_PRINT(...) fprintf(stdout, __VA_ARGS__)
#else
# define DEBUG_PRINT(...) do {} while (0)
#endif

#include "scheduler.h"

using namespace std;

void* klib;


void* Scheduler::busy_wait(void** qbuffers, pthread_mutex_t** mutexes, int num_clients) {
	

	DEBUG_PRINT("entered busy wait!\n");	
			
	queue<struct func_record>** buffers = (queue<struct func_record>**)malloc(num_clients * sizeof(queue<struct kernel_record>*));
	//(queue<struct kernel_record>**)qbuffers;
	for (int i=0; i<num_clients; i++)
		buffers[i] = (queue<struct func_record>*)(qbuffers[i]);

	// for kernel
	cudaError_t (*kernel_function)(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, cudaStream_t stream);
	*(void **)(&kernel_function) = dlsym(RTLD_DEFAULT, "cudaLaunchKernel");

	// for memcpy
	cudaError_t (*memcpy_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
	*(void **)(&memcpy_function) = dlsym (RTLD_DEFAULT, "cudaMemcpy");
	
	// for memcpy_async
	cudaError_t (*memcpy_async_function)(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream);
	*(void **)(&memcpy_async_function) = dlsym (RTLD_DEFAULT, "cudaMemcpyAsync");


	// for malloc
	cudaError_t (*malloc_function)(void** devPtr, size_t size);
	*(void **)(&malloc_function) = dlsym (RTLD_DEFAULT, "cudaMalloc");

	// for cudnn conv
	cudnnStatus_t (*cudnn_conv_function)(cudnnHandle_t handle, const void *alpha, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnFilterDescriptor_t wDesc, const void *w, const cudnnConvolutionDescriptor_t convDesc, cudnnConvolutionFwdAlgo_t algo, void *workSpace, size_t workSpaceSizeInBytes, const void *beta, const cudnnTensorDescriptor_t yDesc, void *y) ;
	*(void **)(&cudnn_conv_function) = dlsym(RTLD_DEFAULT, "cudnnConvolutionForward");
	assert(cudnn_conv_function != NULL);

	// for bnorm train
	cudnnStatus_t (*cudnn_bnorm_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, cudnnBatchNormOps_t bnOps, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *xData, const cudnnTensorDescriptor_t zDesc,  const void *zData, const cudnnTensorDescriptor_t yDesc, void *yData, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScaleData, const void *bnBiasData, double exponentialAverageFactor, void *resultRunningMeanData, void *resultRunningVarianceData, double epsilon, void *saveMean, void *saveInvVariance, const cudnnActivationDescriptor_t activationDesc,  void *workspace, size_t workSpaceSizeInBytes, void *reserveSpace, size_t reserveSpaceSizeInBytes);

	*(void **)(&cudnn_bnorm_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardTrainingEx");
	assert(cudnn_bnorm_function != NULL);

	// for bnorm infer
	cudnnStatus_t (*cudnn_bnorm_infer_function)(cudnnHandle_t handle, cudnnBatchNormMode_t mode, const void *alpha, const void *beta, const cudnnTensorDescriptor_t xDesc, const void *x, const cudnnTensorDescriptor_t yDesc, void *y, const cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void *bnScale, const void *bnBias, const void *estimatedMean, const void *estimatedVariance, double epsilon);

	*(void **)(&cudnn_bnorm_infer_function) = dlsym(RTLD_DEFAULT, "cudnnBatchNormalizationForwardInference");
	assert(cudnn_bnorm_infer_function != NULL);


	// CUBLAS sgemm
	cublasStatus_t (*cublas_sgemm_function)(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc);

	*(void **)(&cublas_sgemm_function) = dlsym(RTLD_DEFAULT, "cublasSgemm");
	assert(cublas_sgemm_function != NULL);


	cudaStream_t sched_stream;
	cudaStreamCreate(&sched_stream);

	int seen[num_clients] = {0};
	
	int num_kernels = 289;
	int num_iters = 1;
	int it = 0;

	DEBUG_PRINT("for ID 0: mutex address is %p, buffer address is %p, buffers is %p\n", mutexes[0], buffers[0], buffers);

	while (it < num_iters) {
		for (int i=0; i<num_clients; i++) {
			while (seen[i] < num_kernels) {
				pthread_mutex_lock(mutexes[i]);
				volatile int sz = buffers[i]->size();
				if (sz > 0) {
					struct func_record frecord = buffers[i]->front();
					
					// case 1
					if (frecord.type == KERNEL_RECORD) {
						DEBUG_PRINT("found a new kernel record!\n");
						kernel_record record = frecord.data.krecord;
						(*kernel_function)(record.func, record.gridDim, record.blockDim, record.args, record.sharedMem, sched_stream);
					}

					else if (frecord.type == MEMCPY_RECORD) {
						DEBUG_PRINT("found a new memcpy record!\n");
						memcpy_record record = frecord.data.mrecord;
						if (not record.async) {
							(*memcpy_function)(record.dst, record.src, record.count, record.kind);
						} else {
							(*memcpy_async_function)(record.dst, record.src, record.count, record.kind, sched_stream);
						}

					}

					else if (frecord.type == MALLOC_RECORD) {
						DEBUG_PRINT("found a new malloc record!\n");
						malloc_record record = frecord.data.malrecord;
						(*malloc_function)(record.devPtr, record.size);

					}

					else if (frecord.type == CUDNN_CONV_RECORD) {					
						DEBUG_PRINT("found a new cudnn conv record!\n");
						cudnnConvolutionForward_record record = frecord.data.cudnnConvRecord;
						cudnnSetStream(record.handle, 0);
						(*cudnn_conv_function)(record.handle, record.alpha, record.xDesc, record.x, record.wDesc, record.w, record.convDesc, record.algo, record.workSpace, record.workSpaceSizeInBytes, record.beta, record.yDesc, record.y);
						cudnnSetStream(record.handle, 0); // TODO: I want to set the default stream here
					}

					else if (frecord.type == CUDNN_BNORM_RECORD) {
						DEBUG_PRINT("found a new bnorm record!\n");
						cudnnBatchNormalizationForwardTrainingEx_record record = frecord.data.cudnnBNormRecord;
						cudnnSetStream(record.handle, 0);
						(*cudnn_bnorm_function)(record.handle, record.mode, record.bnOps, record.alpha, record.beta, record.xDesc, record.xData, record.zDesc, record.zData, record.yDesc, record.yData, record.bnScaleBiasMeanVarDesc, record.bnScaleData, record.bnBiasData, record.exponentialAverageFactor, record.resultRunningMeanData, record.resultRunningVarianceData, record.epsilon, record.saveMean, record.saveInvVariance, record.activationDesc, record.workspace, record.workSpaceSizeInBytes, record.reserveSpace, record.reserveSpaceSizeInBytes);
						cudnnSetStream(record.handle, 0); // TODO: I want to set the default stream here
					}

					else if (frecord.type == CUDNN_BNORM_INF_RECORD) {
						DEBUG_PRINT("found a new bnorm inf record!\n"); 
						cudnnBatchNormalizationForwardInference_record record = frecord.data.cudnnBNormInfRecord;
						cudnnSetStream(record.handle, 0);
						(*cudnn_bnorm_infer_function)(record.handle, record.mode, record.alpha, record.beta, record.xDesc, record.x, record.yDesc, record.y, record.bnScaleBiasMeanVarDesc, record.bnScale, record.bnBias, record.estimatedMean, record.estimatedVariance, record.epsilon);
						cudnnSetStream(record.handle, 0);

					}

					else if (frecord.type == CUBLAS_SGEMM_RECORD) {
						DEBUG_PRINT("found a new sgemm record!\n");
					
						// TODO: what to do about streams?
						cublasSgemm_record record = frecord.data.cublasSgemmRecord;
						(*cublas_sgemm_function)(record.handle, record.transa, record.transb, record.m, record.n, record.k, record.alpha, record.A, record.lda, record.B, record.ldb, record.beta, record.C, record.ldc);
					}

					//buffers[i]->pop();

					// run
					// case 2
					/*if (!record.run) {
					/	buffers[i]->front().sched_stream = sched_stream;
						buffers[i]->front().run = true;   
						seen[i] += 1;*/
						//printf("%d, kernel record func ptr is %p, args is %p, run is %d, stream is %d\n", seen[i], record.func, record.args, record.run, sched_stream);

					//}
				}
				//pthread_mutex_unlock(mutexes[i]);
			}

		}
		it += 1;
		for (int i=0; i<num_clients; i++)
			seen[i] = 0;
		DEBUG_PRINT("restart! %d\n", it);
	}

	return NULL;
	
}

extern "C" {

	Scheduler* sched_init() {
		
		Scheduler* sched = new Scheduler();
		return sched;
	}


	void populate_kernel_names(vector<char*>* kernel_vector) {

		// TODO: make this more generic, e.g. pass files/models w.r.t input
		string line;
		std::ifstream infile("kernel_file");
		assert (infile.is_open());
		while (std::getline(infile, line))
		{
			char* kernel_name = (char*)malloc(line.length());
			strcpy(kernel_name, line.c_str());
			kernel_vector->push_back(kernel_name);
		}

		for (auto s: *kernel_vector)
			printf("kernel: %s\n", s);

		printf("--------------------------------\n");

	}


	void setup(Scheduler* scheduler, int tid0, int tid1) {

		struct passwd *pw = getpwuid(getuid());
		char *homedir = pw->pw_dir;
		char* lib_path = "/gpu_share_repo/cpp_backend/cuda_capture/libinttemp.so";

		klib = dlopen(strcat(homedir, lib_path), RTLD_NOW | RTLD_GLOBAL);

		if (!klib) {
			fprintf(stderr, "Error: %s\n", dlerror());
			return;
		}
		
#ifdef SYS_gettid
		pid_t mytid = syscall(SYS_gettid);
#else
#error "SYS_gettid unavailable on this system"
#endif

		pid_t* thread_ids_all = (pid_t*)dlsym(klib, "thread_ids");
		thread_ids_all[0] = tid0;
		thread_ids_all[1] = tid1;
		thread_ids_all[2] = mytid;
		
		DEBUG_PRINT("Scheduler setup the thread ids to be %d, %d, %d\n", thread_ids_all[0], thread_ids_all[1], thread_ids_all[2]);


		int num_kernels = 1;
		vector<char*>** func_names_all = (vector<char*>**)dlsym(klib, "func_names");
		printf("func_names_all is %p\n", func_names_all);
		printf("fname0 ptr is %p, fname1 ptr is %p\n", func_names_all[0], func_names_all[1]);
		populate_kernel_names(func_names_all[0]);
		populate_kernel_names(func_names_all[1]);

	}

	void* sched_func(Scheduler* scheduler) { //void* buffer, pthread_mutex_t* mutex) {

		
		//Scheduler* scheduler = (Scheduler*)(arg);
		void** buffers = (void**)dlsym(klib, "kqueues"); 
	
		DEBUG_PRINT("buffers is %p, %p, %p\n", buffers, buffers[0], buffers[1]);
		pthread_mutex_t** mutexes = (pthread_mutex_t**)dlsym(klib, "mutexes"); 
		int num_clients = 1;

		DEBUG_PRINT("entered sched func!\n");
		scheduler->busy_wait(buffers, mutexes, num_clients);
		DEBUG_PRINT("exited sched func!\n");  
		return NULL;
	}
}


