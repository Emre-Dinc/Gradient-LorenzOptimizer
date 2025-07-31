import numpy as np
import time
from typing import Callable, Tuple, Dict, List, Optional, Union
import math

class GlobalLorenzChaos:
    """
    Global Lorenz Chaos Engine - generates high-dimensional chaotic flow
    from a single 3D Lorenz attractor state
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        
        # Single 3D Lorenz system for the entire parameter space
        self.lorenz_state = np.random.randn(3) * 0.1  # (x, y, z)
        
        # Classic Lorenz parameters
        self.sigma = 10.0
        self.rho = 28.0  
        self.beta = 8.0/3.0
        self.dt = 0.01
        
        # Chaos flow generation parameters
        self.flow_frequency_base = 0.1  # Base frequency for sine waves
        self.flow_amplitude = 1.0       # Amplitude of chaotic flow
        
        self.chaos_strength = 0.5       
        self.min_chaos_strength = 0.2
        self.max_chaos_strength = 0.7
        
        # Performance tracking
        self.recent_improvements = []
        self.stagnation_counter = 0
        
    def update_lorenz_state(self, current_params: np.ndarray) -> None:
        """
        Update the internal 3D Lorenz state
        Uses global parameter statistics to influence dynamics
        """
        lx, ly, lz = self.lorenz_state
        
        # Global parameter influence on Lorenz dynamics
        param_mean = np.mean(current_params)
        param_std = np.std(current_params)
        
        # Modulate Lorenz parameters based on global parameter state
        sigma_mod = self.sigma * (1 + 0.05 * np.tanh(param_mean))
        rho_mod = self.rho * (1 + 0.02 * np.tanh(param_std))
        beta_mod = self.beta
        
        # Standard Lorenz equations
        dx_dt = sigma_mod * (ly - lx)
        dy_dt = lx * (rho_mod - lz) - ly
        dz_dt = lx * ly - beta_mod * lz
        
        # Euler integration with bounds
        new_lx = np.clip(lx + self.dt * dx_dt, -25, 25)
        new_ly = np.clip(ly + self.dt * dy_dt, -25, 25)
        new_lz = np.clip(lz + self.dt * dz_dt, -25, 25)
        
        # Update state if finite
        if np.all(np.isfinite([new_lx, new_ly, new_lz])):
            self.lorenz_state = np.array([new_lx, new_ly, new_lz])
    
    def generate_global_chaos_flow(self, current_params: np.ndarray) -> np.ndarray:
        """
        Generate high-dimensional chaotic flow vector from 3D Lorenz state
        structured chaos across full parameter space
        """
        # Update Lorenz state based on current parameters
        self.update_lorenz_state(current_params)
        
        lx, ly, lz = self.lorenz_state
        
        # Generate structured high-dimensional chaos
        chaos_flow = np.zeros(self.dimension)
        
        for i in range(self.dimension):
            # Create rich, structured patterns from Lorenz state
            # Each dimension gets a unique combination of the 3D state
            
            # Base frequency modulated by dimension index
            freq_x = self.flow_frequency_base * (1 + i * 0.01)
            freq_y = self.flow_frequency_base * (1 + i * 0.007)
            freq_z = self.flow_frequency_base * (1 + i * 0.013)
            
            # Structured sine/cosine combinations
            component_x = np.sin(freq_x * lx + i * ly * 0.1)
            component_y = np.cos(freq_y * ly + i * lz * 0.1) 
            component_z = np.sin(freq_z * lz + i * lx * 0.1)
            
            # Cross-coupling between components
            cross_coupling = 0.1 * np.sin(lx * ly * freq_x + i * 0.1)
            
            # Combine all components
            chaos_flow[i] = (component_x + component_y + component_z + cross_coupling)
            
            # Add spiral structure
            spiral_phase = (i / self.dimension) * 2 * np.pi
            spiral_component = 0.2 * np.sin(lx + spiral_phase) * np.cos(ly + spiral_phase)
            chaos_flow[i] += spiral_component
        
        # Normalize to unit vector
        flow_norm = np.linalg.norm(chaos_flow)
        if flow_norm > 1e-8:
            chaos_flow = chaos_flow / flow_norm
        else:
            # Fallback: simple structured pattern
            for i in range(self.dimension):
                chaos_flow[i] = np.sin(i * 0.1 + lx) * np.cos(i * 0.1 + ly)
            chaos_flow = chaos_flow / (np.linalg.norm(chaos_flow) + 1e-8)
        
        return chaos_flow * self.flow_amplitude
    
    def update_performance_tracking(self, improvement: float):
        """Track performance for adaptive chaos strength"""
        self.recent_improvements.append(improvement)
        
        # Keep recent history
        if len(self.recent_improvements) > 20:
            self.recent_improvements.pop(0)
            
        # Update stagnation counter
        if improvement > 1e-8:
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
    
    def adapt_chaos_strength(self):
        """Adapt chaos strength based on performance"""
        if len(self.recent_improvements) < 5:
            return
            
        avg_improvement = np.mean(self.recent_improvements)
        
        if avg_improvement > 1e-4:  # Good progress
            # Reduce chaos for exploitation
            self.chaos_strength *= 0.98
        elif self.stagnation_counter > 20:  # Stagnation
            # Increase chaos for exploration  
            self.chaos_strength *= 1.02
            
        # Clamp chaos strength
        self.chaos_strength = np.clip(self.chaos_strength,
                                     self.min_chaos_strength,
                                     self.max_chaos_strength)


class GlobalLorenzOptimizer:
    """
    Global Gradient-Guided Chaos Optimizer
    Operates on full parameter vector with holistic Lorenz dynamics
    """
    
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.chaos_engine = GlobalLorenzChaos(dimension)
        
        # Guidance parameters
        self.guidance_strength = 0.8  # How much to follow gradient vs chaos
        self.min_guidance = 0.7
        self.max_guidance = 0.9
        
        # Step size control
        self.step_size = 0.01
        self.min_step_size = 1e-4
        self.max_step_size = 5
        
        # Momentum for smooth flow
        self.momentum = np.zeros(dimension)
        self.momentum_decay = 0.9
        
        # Internal adaptation tracking
        self.iteration_count = 0
        self.prev_loss = None
        
        print(f"Global Gradient-Guided Chaos Optimizer")
        print(f"   Dimension: {dimension}D")
        print(f"   Guidance strength: {self.guidance_strength:.2f}")
        print(f"   Step size: {self.step_size:.4f}")
    
    def optimize_step(self, 
                     current_params: np.ndarray, 
                     gradient: np.ndarray,
                     loss_value: Optional[float] = None) -> np.ndarray:
        """
        Single optimization step using global gradient-guided chaos
        
        Args:
            current_params: Full parameter vector
            gradient: Global gradient for all parameters
            loss_value: Current loss (for performance tracking)
            
        Returns:
            Updated parameter vector
        """
        # Global compass: Normalize gradient direction
        grad_norm = np.linalg.norm(gradient)
        if grad_norm > 1e-8:
            gradient_direction = -gradient / grad_norm  # Negative for descent
        else:
            gradient_direction = np.zeros_like(gradient)
        
        # Global flow: Generate chaotic flow from Lorenz state
        chaos_flow = self.chaos_engine.generate_global_chaos_flow(current_params)
        
        # Apply chaos strength scaling
        chaos_flow = chaos_flow * self.chaos_engine.chaos_strength
        
        # Final update: Blend chaos flow with gradient compass
        final_direction = ((1 - self.guidance_strength) * chaos_flow + 
                          self.guidance_strength * gradient_direction)
        
        # Update momentum for smooth flow
        self.momentum = (self.momentum_decay * self.momentum + 
                        (1 - self.momentum_decay) * final_direction)
        
        # Apply step with adaptive step size
        update = self.step_size * self.momentum
        
        # Update parameters
        new_params = current_params + update
        
        # Auto-adaptation every 50 iterations
        self.iteration_count += 1
        if self.iteration_count % 50 == 0 and self.prev_loss is not None and loss_value is not None:
            improvement = self.prev_loss - loss_value
            self.chaos_engine.update_performance_tracking(improvement)
            self.chaos_engine.adapt_chaos_strength()
            self.adapt_guidance_strength(improvement)
            self.adapt_step_size(improvement)
        
        # Store current loss for next iteration
        if loss_value is not None:
            self.prev_loss = loss_value
        
        return new_params
    
    def adapt_guidance_strength(self, improvement: float):
        """Adapt guidance strength based on performance"""
        if improvement > 1e-4:  # Good progress with current guidance
            # Keep current balance
            pass
        elif improvement < -1e-4:  # Getting worse
            # Increase chaos exploration
            self.guidance_strength *= 0.95
        else:  # Stagnation
            # Try more gradient following
            self.guidance_strength *= 1.02
            
        self.guidance_strength = np.clip(self.guidance_strength,
                                        self.min_guidance, 
                                        self.max_guidance)
    
    def adapt_step_size(self, improvement: float):
        """Adapt step size based on performance"""
        if improvement > 1e-4:  # Good progress
            # Slightly increase step size
            self.step_size *= 1.01
        elif improvement < -1e-4:  # Getting worse
            # Reduce step size
            self.step_size *= 0.95
            
        self.step_size = np.clip(self.step_size,
                                self.min_step_size,
                                self.max_step_size)
