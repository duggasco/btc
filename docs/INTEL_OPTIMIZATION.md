# Intel GPU and CPU Optimization Guide

This guide explains how to enable Intel hardware optimizations for the BTC trading system, specifically for Intel N150 processors with integrated Intel UHD Graphics.

## Overview

The Intel N150 processor features:
- Intel UHD Graphics (Gen 12, Alder Lake-N)
- Efficient cores with AVX2 support
- Integrated GPU with limited compute capabilities

For this hardware, CPU optimizations typically provide better performance than GPU compute for model training.

## Optimization Strategies

### 1. Intel Extension for PyTorch (IPEX)
- Provides optimized operators for Intel CPUs
- Automatic mixed precision support
- Better memory allocation strategies
- Thread optimization for Intel architectures

### 2. CPU Optimizations
- Intel MKL-DNN (already included in PyTorch)
- OpenMP threading optimization
- AVX2 vectorization
- Optimal thread affinity settings

### 3. Docker Configuration
- Device mapping for Intel GPU access
- Environment variables for thread optimization
- Group permissions for GPU device access

## Setup Instructions

### Option 1: Use Intel-Optimized Docker Compose

```bash
# Build and run with Intel optimizations
docker compose -f docker-compose.intel.yml up --build -d
```

### Option 2: Update Existing Setup

1. Install Intel Extension for PyTorch in your container:
```bash
docker exec btc-trading-backend pip install intel-extension-for-pytorch==2.1.0
```

2. Enable GPU device access:
```yaml
# Add to docker-compose.yml under backend service
devices:
  - /dev/dri:/dev/dri
group_add:
  - video
```

3. Set optimization environment variables:
```yaml
environment:
  - KMP_BLOCKTIME=1
  - KMP_SETTINGS=1
  - KMP_AFFINITY=granularity=fine,compact,1,0
  - OMP_NUM_THREADS=4
  - MKL_NUM_THREADS=4
```

## Performance Expectations

For Intel N150 with integrated GPU:

### Training Performance
- CPU with IPEX: ~20-30% faster than baseline
- Intel iGPU: Limited benefit due to few compute units
- Recommendation: Use CPU with IPEX optimizations

### Inference Performance
- CPU with IPEX: ~30-50% faster than baseline
- Batch size 1: Optimal for real-time predictions
- Enable autocast for additional speedup

### Memory Usage
- IPEX reduces memory footprint by ~15-20%
- Better cache utilization on Intel CPUs
- Efficient memory allocation patterns

## Monitoring Optimizations

Check optimization status:
```bash
curl http://localhost:8090/ml/status | jq .optimization
```

Expected output:
```json
{
  "ipex_available": true,
  "mkl_available": true,
  "openmp_threads": 4,
  "cpu_count": 4,
  "ipex_version": "2.1.0",
  "xpu_available": false,
  "cpu_brand": "Intel(R) N150",
  "cpu_features": {
    "avx2": true,
    "avx512": false
  }
}
```

## Best Practices

1. **Thread Configuration**: Set threads to number of physical cores (not hyperthreads)
2. **Batch Size**: Use smaller batch sizes (16-32) for better CPU cache utilization
3. **Memory Pinning**: Enable for consistent performance
4. **Process Affinity**: Bind processes to specific cores for reduced context switching

## Troubleshooting

### IPEX Not Loading
```bash
# Check installation
docker exec btc-trading-backend python -c "import intel_extension_for_pytorch as ipex; print(ipex.__version__)"
```

### Performance Not Improved
- Verify thread settings: `echo $OMP_NUM_THREADS`
- Check CPU frequency scaling: `cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`
- Monitor CPU usage: `htop` or `docker stats`

### GPU Device Not Accessible
- Check permissions: `ls -la /dev/dri/`
- Verify video group: `groups`
- Test in container: `docker exec btc-trading-backend ls -la /dev/dri/`

## Future Enhancements

1. **OpenVINO Integration**: For optimized inference
2. **Neural Compressor**: For INT8 quantization
3. **Profile-Guided Optimization**: Custom kernels for specific workloads
4. **Distributed Training**: Multi-node training with Intel MPI