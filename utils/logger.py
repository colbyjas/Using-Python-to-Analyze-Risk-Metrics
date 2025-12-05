# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 01:10:13 2025

@author: Colby Jaskowiak

Logger module.
Execution tracking, progress indicators, and debugging support.
"""

import time
from datetime import datetime
from functools import wraps

import brg_risk_metrics.config.settings as cfg

#%% TIMING UTILITIES
class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name="Operation", verbose=True):
        self.name = name
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {self.name}")
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed: {self.name} ({self.elapsed:.2f}s)")
    
    def get_elapsed(self):
        """Get elapsed time in seconds."""
        if self.elapsed is not None:
            return self.elapsed
        elif self.start_time is not None:
            return time.time() - self.start_time
        else:
            return 0

def time_function(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMER] {func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper

#%% PROGRESS TRACKING
class ProgressTracker:
    """Simple progress tracker."""
    
    def __init__(self, total, description="Progress", width=50):
        self.total = total
        self.current = 0
        self.description = description
        self.width = width
        self.start_time = time.time()
    
    def update(self, n=1):
        """Update progress by n steps."""
        self.current += n
        self._print_progress()
    
    def _print_progress(self):
        """Print progress bar."""
        pct = self.current / self.total
        filled = int(self.width * pct)
        bar = '█' * filled + '░' * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed / self.current * (self.total - self.current)
            eta_str = f"ETA: {eta:.0f}s"
        else:
            eta_str = "ETA: --"
        
        print(f"\r{self.description}: |{bar}| {self.current}/{self.total} ({pct:.1%}) {eta_str}", end='')
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def finish(self):
        """Mark as complete."""
        self.current = self.total
        self._print_progress()

def progress_iterator(iterable, description="Processing"):
    """Wrap iterator with progress tracking."""
    items = list(iterable)
    tracker = ProgressTracker(len(items), description=description)
    
    for item in items:
        yield item
        tracker.update(1)

#%% LOGGING UTILITIES
class ExecutionLog:
    """Simple execution logger."""
    
    def __init__(self, name="Execution"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.steps = []
    
    def start(self):
        """Start logging."""
        self.start_time = datetime.now()
        self.log_step("Execution started")
    
    def log_step(self, description, data=None):
        """Log a step."""
        step = {
            'timestamp': datetime.now(),
            'description': description,
            'data': data
        }
        self.steps.append(step)
    
    def finish(self, success=True):
        """Finish logging."""
        self.end_time = datetime.now()
        status = "completed successfully" if success else "failed"
        self.log_step(f"Execution {status}")
    
    def get_summary(self):
        """Get execution summary."""
        if self.start_time is None:
            return "No execution recorded"
        
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else None
        
        summary = f"\n{'='*60}\n"
        summary += f"EXECUTION LOG: {self.name}\n"
        summary += f"{'='*60}\n"
        summary += f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if self.end_time:
            summary += f"End time:   {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            summary += f"Duration:   {duration:.2f}s\n"
        
        summary += f"\nSteps:\n"
        for i, step in enumerate(self.steps, 1):
            summary += f"  {i}. [{step['timestamp'].strftime('%H:%M:%S')}] {step['description']}\n"
            if step['data'] is not None:
                summary += f"      Data: {step['data']}\n"
        
        summary += f"{'='*60}\n"
        return summary
    
    def print_summary(self):
        """Print execution summary."""
        print(self.get_summary())

#%% DEBUG UTILITIES
def debug_print(message, data=None, enabled=True):
    """Print debug message with timestamp."""
    if not enabled:
        return
    
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f"[DEBUG {timestamp}] {message}")
    if data is not None:
        print(f"  Data: {data}")

def log_function_call(func):
    """Decorator to log function calls."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        print(f"[CALL] {func_name}()")
        
        try:
            result = func(*args, **kwargs)
            print(f"[SUCCESS] {func_name}() completed")
            return result
        except Exception as e:
            print(f"[ERROR] {func_name}() failed: {e}")
            raise
    
    return wrapper

#%% PERFORMANCE MONITORING
class PerformanceMonitor:
    """Monitor performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_operation(self, operation_name):
        """Start timing an operation."""
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name):
        """End timing an operation."""
        if operation_name in self.start_times:
            elapsed = time.time() - self.start_times[operation_name]
            
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            
            self.metrics[operation_name].append(elapsed)
            del self.start_times[operation_name]
            
            return elapsed
        return None
    
    def get_summary(self):
        """Get performance summary."""
        summary = "\nPERFORMANCE SUMMARY\n"
        summary += "="*60 + "\n"
        
        for operation, times in self.metrics.items():
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            total_time = sum(times)
            
            summary += f"{operation}:\n"
            summary += f"  Calls: {len(times)}\n"
            summary += f"  Avg:   {avg_time:.3f}s\n"
            summary += f"  Min:   {min_time:.3f}s\n"
            summary += f"  Max:   {max_time:.3f}s\n"
            summary += f"  Total: {total_time:.3f}s\n\n"
        
        summary += "="*60 + "\n"
        return summary
    
    def print_summary(self):
        """Print performance summary."""
        print(self.get_summary())

#%%
if __name__ == "__main__":
    print("Testing logger.py...\n")
    
    # Test 1: Timer
    print("1. Testing Timer...")
    with Timer("Sleep operation"):
        time.sleep(0.5)
    
    # Test 2: Progress tracker
    print("\n2. Testing Progress tracker...")
    for i in progress_iterator(range(50), description="Processing items"):
        time.sleep(0.02)
    
    # Test 3: Execution log
    print("\n3. Testing Execution log...")
    log = ExecutionLog("Test Execution")
    log.start()
    log.log_step("Loading data", data="SPY 2020-2025")
    time.sleep(0.1)
    log.log_step("Calculating metrics")
    time.sleep(0.1)
    log.log_step("Generating plots")
    log.finish(success=True)
    log.print_summary()
    
    # Test 4: Performance monitor
    print("\n4. Testing Performance monitor...")
    monitor = PerformanceMonitor()
    
    for i in range(3):
        monitor.start_operation("test_operation")
        time.sleep(0.1)
        monitor.end_operation("test_operation")
    
    monitor.print_summary()
    
    print("Test complete!")