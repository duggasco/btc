"""Intel-optimized LSTM model for efficient training on Intel hardware"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

# Try to import Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    logger.info("Intel Extension for PyTorch (IPEX) is available")
except ImportError:
    IPEX_AVAILABLE = False
    logger.info("IPEX not available, using standard PyTorch")

class IntelOptimizedLSTMTrainer:
    """LSTM trainer with Intel-specific optimizations"""
    
    def __init__(self, base_trainer):
        """
        Initialize Intel-optimized trainer as a wrapper around the base trainer
        
        Args:
            base_trainer: Instance of LSTMTrainer from enhanced_lstm.py
        """
        self.base_trainer = base_trainer
        self.device = self._setup_device()
        self.ipex_optimized = False
        
    def _setup_device(self):
        """Setup compute device with Intel optimizations"""
        # For Intel integrated GPUs, CPU with IPEX is often faster than GPU compute
        # This is because Intel iGPUs have limited compute units
        
        if IPEX_AVAILABLE:
            # Check if XPU (Intel GPU) is available
            try:
                import intel_extension_for_pytorch as ipex
                if hasattr(ipex, 'xpu') and ipex.xpu.is_available():
                    device = 'xpu'
                    logger.info("Intel XPU device available")
                else:
                    device = 'cpu'
                    logger.info("Using CPU with IPEX optimizations")
            except:
                device = 'cpu'
                logger.info("Using CPU with IPEX optimizations")
        else:
            device = 'cpu'
            logger.info("Using CPU without IPEX")
            
        return torch.device(device)
    
    def optimize_model(self, model: nn.Module, optimizer: torch.optim.Optimizer):
        """Apply Intel optimizations to model and optimizer"""
        if IPEX_AVAILABLE:
            try:
                # Apply IPEX optimizations
                model, optimizer = ipex.optimize(model, optimizer=optimizer, level="O1")
                self.ipex_optimized = True
                logger.info("Applied IPEX optimizations to model")
                
                # Set optimal thread settings for Intel CPUs
                if self.device.type == 'cpu':
                    # Get number of physical cores
                    import os
                    num_cores = os.cpu_count() // 2  # Assume hyperthreading
                    torch.set_num_threads(num_cores)
                    logger.info(f"Set PyTorch threads to {num_cores}")
                    
            except Exception as e:
                logger.warning(f"Failed to apply IPEX optimizations: {e}")
                self.ipex_optimized = False
        
        return model, optimizer
    
    def train(self, *args, **kwargs):
        """Train model with Intel optimizations"""
        # Get the base training setup
        result = self.base_trainer.train(*args, **kwargs)
        
        # Apply Intel optimizations to the trained model
        if self.base_trainer.model is not None and IPEX_AVAILABLE:
            try:
                # Optimize model for inference
                self.base_trainer.model = ipex.optimize(self.base_trainer.model, level="O1")
                logger.info("Optimized trained model with IPEX")
            except Exception as e:
                logger.warning(f"Failed to optimize trained model: {e}")
        
        return result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Intel optimizations"""
        if self.base_trainer.model is None:
            raise ValueError("Model not trained yet")
            
        # Use base trainer's predict method
        with torch.no_grad():
            if IPEX_AVAILABLE and self.device.type == 'cpu':
                # Enable Intel optimizations for inference
                with torch.cpu.amp.autocast():
                    return self.base_trainer.predict(X)
            else:
                return self.base_trainer.predict(X)
    
    @staticmethod
    def get_optimization_info() -> Dict[str, Any]:
        """Get information about available optimizations"""
        info = {
            'ipex_available': IPEX_AVAILABLE,
            'mkl_available': torch.backends.mkl.is_available(),
            'openmp_threads': torch.get_num_threads(),
            'cpu_count': torch.get_num_threads(),
        }
        
        if IPEX_AVAILABLE:
            try:
                import intel_extension_for_pytorch as ipex
                info['ipex_version'] = ipex.__version__
                info['xpu_available'] = hasattr(ipex, 'xpu') and ipex.xpu.is_available()
            except:
                pass
                
        # Check CPU features
        try:
            import cpuinfo
            cpu_info = cpuinfo.get_cpu_info()
            info['cpu_brand'] = cpu_info.get('brand_raw', 'Unknown')
            info['cpu_features'] = {
                'avx2': 'avx2' in cpu_info.get('flags', []),
                'avx512': any('avx512' in flag for flag in cpu_info.get('flags', [])),
            }
        except:
            pass
            
        return info

# Utility function to wrap existing trainer
def create_intel_optimized_trainer(model_dir: str = '/app/models', device: str = None):
    """
    Create an Intel-optimized LSTM trainer
    
    Note: Intel optimizations are now integrated directly into LSTMTrainer in enhanced_lstm.py
    This function is kept for backward compatibility but returns the standard trainer
    which already includes Intel optimizations when available.
    """
    from models.enhanced_lstm import LSTMTrainer
    
    # LSTMTrainer now has Intel optimizations built-in
    return LSTMTrainer(model_dir=model_dir, device=device)