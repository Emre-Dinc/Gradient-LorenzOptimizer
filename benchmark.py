import numpy as np
import matplotlib.pyplot as plt
import time
from global_lorenz_optimizer import GlobalLorenzOptimizer

class BenchmarkFunctions:
    """Collection of optimization benchmark functions with their gradients"""
    
    @staticmethod
    def sphere(x):
        return np.sum(x**2)
    
    @staticmethod
    def sphere_gradient(x):
        return 2 * x
    
    @staticmethod
    def rosenbrock(x):
        if len(x) < 2:
            return (x[0] - 1)**2
        total = 0
        for i in range(len(x) - 1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return total
    
    @staticmethod
    def rosenbrock_gradient(x):
        if len(x) < 2:
            return np.array([2 * (x[0] - 1)])
        grad = np.zeros_like(x)
        for i in range(len(x) - 1):
            grad[i] += -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
            grad[i+1] += 200 * (x[i+1] - x[i]**2)
        return grad
    
    @staticmethod
    def rastrigin(x):
        A = 10
        n = len(x)
        return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
    
    @staticmethod
    def rastrigin_gradient(x):
        A = 10
        return 2 * x + 2 * A * np.pi * np.sin(2 * np.pi * x)
    
    @staticmethod
    def ackley(x):
        a, b, c = 20, 0.2, 2 * np.pi
        n = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(c * x))
        return (-a * np.exp(-b * np.sqrt(sum_sq / n)) - 
                np.exp(sum_cos / n) + a + np.e)
    
    @staticmethod
    def ackley_gradient(x):
        a, b, c = 20, 0.2, 2 * np.pi
        n = len(x)
        sum_sq = np.sum(x**2)
        sqrt_term = np.sqrt(sum_sq / n)
        grad1 = (a * b * np.exp(-b * sqrt_term)) / (n * sqrt_term + 1e-8) * x
        grad2 = (c * np.exp(np.sum(np.cos(c * x)) / n) / n) * np.sin(c * x)
        return grad1 + grad2
    
    @staticmethod
    def griewank(x):
        sum_sq = np.sum(x**2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        return 1 + sum_sq - prod_cos
    
    @staticmethod
    def griewank_gradient(x):
        n = len(x)
        indices = np.arange(1, n + 1)
        sqrt_indices = np.sqrt(indices)
        grad_sum = x / 2000
        cos_terms = np.cos(x / sqrt_indices)
        prod_cos = np.prod(cos_terms)
        grad_prod = np.zeros_like(x)
        for i in range(n):
            if abs(cos_terms[i]) > 1e-8:
                grad_prod[i] = prod_cos * np.tan(x[i] / sqrt_indices[i]) / sqrt_indices[i]
        return grad_sum + grad_prod
    
    @staticmethod
    def schwefel(x):
        n = len(x)
        return 418.9829 * n - np.sum(x * np.sin(np.sqrt(np.abs(x))))
    
    @staticmethod
    def schwefel_gradient(x):
        grad = np.zeros_like(x)
        for i, xi in enumerate(x):
            if abs(xi) > 1e-8:
                sqrt_abs_xi = np.sqrt(abs(xi))
                sin_term = np.sin(sqrt_abs_xi)
                cos_term = np.cos(sqrt_abs_xi)
                grad[i] = -sin_term - (xi * cos_term) / (2 * sqrt_abs_xi) * np.sign(xi)
        return grad
    
    @staticmethod
    def levy(x):
        def w(x):
            return 1 + (x - 1) / 4
        w_x = w(x)
        n = len(x)
        term1 = np.sin(np.pi * w_x[0])**2
        term2 = 0
        for i in range(n - 1):
            term2 += (w_x[i] - 1)**2 * (1 + 10 * np.sin(np.pi * w_x[i] + 1)**2)
        term3 = (w_x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w_x[-1])**2)
        return term1 + term2 + term3
    
    @staticmethod
    def levy_gradient(x):
        epsilon = 1e-8
        grad = np.zeros_like(x)
        f_x = BenchmarkFunctions.levy(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += epsilon
            grad[i] = (BenchmarkFunctions.levy(x_plus) - f_x) / epsilon
        return grad
    
    @staticmethod
    def zakharov(x):
        sum_sq = np.sum(x**2)
        sum_weighted = np.sum(0.5 * np.arange(1, len(x) + 1) * x)
        return sum_sq + sum_weighted**2 + sum_weighted**4
    
    @staticmethod
    def zakharov_gradient(x):
        indices = 0.5 * np.arange(1, len(x) + 1)
        sum_weighted = np.sum(indices * x)
        grad = 2 * x + 2 * sum_weighted * indices + 4 * sum_weighted**3 * indices
        return grad
    
    @staticmethod
    def dixon_price(x):
        if len(x) == 1:
            return (x[0] - 1)**2
        result = (x[0] - 1)**2
        for i in range(1, len(x)):
            result += (i + 1) * (2 * x[i]**2 - x[i-1])**2
        return result
    
    @staticmethod
    def dixon_price_gradient(x):
        if len(x) == 1:
            return np.array([2 * (x[0] - 1)])
        grad = np.zeros_like(x)
        grad[0] = 2 * (x[0] - 1) - 4 * (2 * x[1]**2 - x[0])
        for i in range(1, len(x) - 1):
            grad[i] = 8 * (i + 1) * x[i] * (2 * x[i]**2 - x[i-1]) - 4 * (i + 2) * (2 * x[i+1]**2 - x[i])
        if len(x) > 1:
            grad[-1] = 8 * len(x) * x[-1] * (2 * x[-1]**2 - x[-2])
        return grad
    
    @staticmethod
    def michalewicz(x, m=10):
        result = 0
        for i, xi in enumerate(x):
            result -= np.sin(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m)
        return result
    
    @staticmethod
    def michalewicz_gradient(x, m=10):
        grad = np.zeros_like(x)
        for i, xi in enumerate(x):
            term1 = np.cos(xi) * (np.sin((i + 1) * xi**2 / np.pi))**(2 * m)
            sin_term = np.sin((i + 1) * xi**2 / np.pi)
            cos_term = np.cos((i + 1) * xi**2 / np.pi)
            term2 = (np.sin(xi) * 2 * m * (sin_term)**(2 * m - 1) * 
                    cos_term * 2 * (i + 1) * xi / np.pi)
            grad[i] = -(term1 + term2)
        return grad
    
    @staticmethod
    def styblinski_tang(x):
        return 0.5 * np.sum(x**4 - 16 * x**2 + 5 * x)
    
    @staticmethod
    def styblinski_tang_gradient(x):
        return 0.5 * (4 * x**3 - 32 * x + 5)


class BenchmarkSuite:
    """Simple benchmark suite for testing optimization algorithms"""
    
    def __init__(self):
        self.functions = {
            'sphere': (BenchmarkFunctions.sphere, BenchmarkFunctions.sphere_gradient, (-5.12, 5.12)),
            'rosenbrock': (BenchmarkFunctions.rosenbrock, BenchmarkFunctions.rosenbrock_gradient, (-50000, 50000)),
            'rastrigin': (BenchmarkFunctions.rastrigin, BenchmarkFunctions.rastrigin_gradient, (-5.12, 5.12)),
            'ackley': (BenchmarkFunctions.ackley, BenchmarkFunctions.ackley_gradient, (-32.768, 32.768)),
            'griewank': (BenchmarkFunctions.griewank, BenchmarkFunctions.griewank_gradient, (-600, 600)),
            'schwefel': (BenchmarkFunctions.schwefel, BenchmarkFunctions.schwefel_gradient, (-500, 500)),
            'levy': (BenchmarkFunctions.levy, BenchmarkFunctions.levy_gradient, (-10, 10)),
            'zakharov': (BenchmarkFunctions.zakharov, BenchmarkFunctions.zakharov_gradient, (-5, 10)),
            'dixon_price': (BenchmarkFunctions.dixon_price, BenchmarkFunctions.dixon_price_gradient, (-10, 10)),
            'michalewicz': (BenchmarkFunctions.michalewicz, BenchmarkFunctions.michalewicz_gradient, (0, np.pi)),
            'styblinski_tang': (BenchmarkFunctions.styblinski_tang, BenchmarkFunctions.styblinski_tang_gradient, (-5, 5))
        }
    
    def run_single_function(self, func_name, dimension, max_iterations=1000, runs=5):
        """
        Test a single benchmark function
        
        Args:
            func_name: Name of the function to test
            dimension: Problem dimension
            max_iterations: Maximum iterations per run
            runs: Number of independent runs
            
        Returns:
            Dictionary with results
        """
        if func_name not in self.functions:
            raise ValueError(f"Unknown function: {func_name}")
        
        func, grad_func, bounds = self.functions[func_name]
        low, high = bounds
        
        print(f"\nTesting {func_name.upper()} - {dimension}D")
        print(f"Domain: [{low}, {high}]")
        print(f"Running {runs} independent runs...")
        print("-" * 50)
        
        results = []
        
        for run in range(runs):
            # Initialize parameters
            x = np.random.uniform(low, high, dimension)
            
            # Create optimizer
            optimizer = GlobalLorenzOptimizer(dimension)
            
            # Track history
            history = {'iterations': [], 'values': [], 'gradient_norms': []}
            
            best_value = float('inf')
            start_time = time.time()
            
            for iteration in range(max_iterations):
                # Evaluate function and gradient
                value = func(x)
                gradient = grad_func(x)
                
                # Track best
                if value < best_value:
                    best_value = value
                
                # Record history
                history['iterations'].append(iteration)
                history['values'].append(value)
                history['gradient_norms'].append(np.linalg.norm(gradient))
                
                # Optimization step
                x = optimizer.optimize_step(x, gradient, value)
                
                # Progress
                if iteration % (max_iterations // 10) == 0:
                    print(f"  Run {run+1}, Iter {iteration}: {value:.6e}")
            
            end_time = time.time()
            
            results.append({
                'run': run + 1,
                'best_value': best_value,
                'final_value': value,
                'time': end_time - start_time,
                'history': history
            })
            
            print(f"  Run {run+1} completed: {best_value:.6e} ({end_time - start_time:.2f}s)")
        
        # Statistics
        best_values = [r['best_value'] for r in results]
        final_values = [r['final_value'] for r in results]
        times = [r['time'] for r in results]
        
        # Print only last 3 runs
        print(f"\nLast 3 runs:")
        for i in range(max(0, len(results)-3), len(results)):
            r = results[i]
            print(f"  Run {r['run']}: {r['best_value']:.6e} ({r['time']:.2f}s)")
        
        summary = {
            'function': func_name,
            'dimension': dimension,
            'runs': runs,
            'best': np.min(best_values),
            'mean': np.mean(best_values),
            'std': np.std(best_values),
            'worst': np.max(best_values),
            'avg_time': np.mean(times),
            'all_results': results
        }
        
        print(f"\nStats: Best={summary['best']:.2e}, Mean={summary['mean']:.2e}, Std={summary['std']:.2e}")
        
        return summary
    
    def run_all_functions(self, dimension, max_iterations=1000, runs=3):
        """
        Test all benchmark functions at given dimension
        
        Args:
            dimension: Problem dimension
            max_iterations: Maximum iterations per run
            runs: Number of runs per function
            
        Returns:
            Dictionary with all results
        """
        print(f"\n{'='*60}")
        print(f"TESTING ALL FUNCTIONS - {dimension}D")
        print(f"{'='*60}")
        
        all_results = {}
        start_time = time.time()
        
        for i, func_name in enumerate(self.functions.keys(), 1):
            print(f"\n[{i}/{len(self.functions)}] {func_name.upper()}")
            result = self.run_single_function(func_name, dimension, max_iterations, runs)
            all_results[func_name] = result
        
        total_time = time.time() - start_time
        
        # Summary
        print(f"\n{'='*60}")
        print(f"ALL FUNCTIONS COMPLETED - {dimension}D")
        print(f"{'='*60}")
        print(f"Total time: {total_time:.2f}s")
        print(f"\nRankings:")
        for rank, (func_name, result) in enumerate(sorted_results, 1):
            print(f"  {rank}. {func_name}: {result['best']:.2e}")
        
        return {
            'dimension': dimension,
            'total_time': total_time,
            'results': all_results,
            'rankings': [name for name, _ in sorted_results]
        }
    
    def plot_convergence(self, result):
        """Plot convergence for a single function result"""
        # Get best run
        best_run = min(result['all_results'], key=lambda x: x['best_value'])
        history = best_run['history']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Function values
        ax1.semilogy(history['iterations'], history['values'])
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Function Value (log scale)')
        ax1.set_title(f'{result["function"].title()} - Best Run')
        ax1.grid(True)
        
        # Gradient norms
        ax2.semilogy(history['iterations'], history['gradient_norms'])
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Gradient Norm (log scale)')
        ax2.set_title('Gradient Norm')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()


# Example usage
def example_usage():
    suite = BenchmarkSuite()
    
    print("Available functions:")
    for name in suite.functions.keys():
        print(f"  - {name}")
    
    # Test single function
    result = suite.run_single_function('rosenbrock', 100, max_iterations=200000, runs=3)
    suite.plot_convergence(result)
    
    #all_results = suite.run_all_functions(500, max_iterations=20000, runs=2)
    
    return suite, result, #all_results


if __name__ == "__main__":
    suite, single_result = example_usage()
    
   # suite, single_result, all_results = example_usage() 




# we offer two of the following for benchmarks:
# suite.run_all_functions(500, max_iterations= , runs= )
# suite.run_single_function('rosenbrock', 100, max_iterations=200000, runs=3)