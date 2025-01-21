# Smoothed-Particle Hydrodynamics Simulation of a Toy Star

This project simulates the structure of a toy star using **Smoothed-Particle Hydrodynamics (SPH)**, a computational method often used in astrophysics to model fluid dynamics and gravitational systems. Originally created by Philip Mocz (Princeton University, 2020), the code has been optimized to improve performance and scalability as part of a final project for the KTH course **DD2358: Introduction to High-Performance Computing**.

**Original Code Repository**: [Philip Mocz's SPH Python Code](https://github.com/pmocz/sph-python/tree/master)  
**Optimized by**:  
- @chrylt: Christiane Kobalt  
- @mareros22: Rose Maresh  
- @peschwartz: Phoebe Schwartz  
- @ge96lip: Carlotta Hölzle  

---

## **Project Overview**
The original code simulates the density distribution and dynamic behavior of a star modeled as a collection of particles interacting via pressure gradients, gravitational forces, and viscosity. While functional, the code exhibits inefficiencies in pairwise interaction calculations, density estimation, and real-time plotting, limiting its performance.

### **Objectives**
1. Identify bottlenecks in the original SPH implementation.
2. Optimize CPU- and memory-intensive operations.
3. Leverage parallelization techniques to scale performance.
4. Reduce computational overhead while maintaining accuracy.

---

## **Optimizations Implemented**
1. **Efficient Pairwise Interactions**:  
   Replaced the \(O(N^2)\) pairwise distance computation with spatial partitioning techniques (e.g., Octree), reducing complexity to \(O(N \log N)\).

2. **Adaptive Smoothing Length**:  
   Introduced an adaptive smoothing kernel where the smoothing length (\(h\)) dynamically adjusts based on local particle density, enhancing resolution in dense regions and reducing unnecessary computations in sparse areas.

3. **Caching and Reusing Results**:  
   Minimized redundant density calculations by caching intermediate results and exploiting kernel symmetry to halve computation time.

4. **Parallelization**:  
   Utilized GPU-based acceleration using **CuPy** and vectorized computations with **Numba** for density, pressure, and acceleration calculations.

5. **Optimized Real-Time Plotting**:  
   Reduced plotting frequency and replaced Matplotlib with **VisPy** for faster rendering, enabling smoother simulations with larger particle counts.

---

## **How to Run the Optimized Code**
### **Dependencies**
Ensure the following Python libraries are installed:
- `numpy`
- `matplotlib` (for final visualization)
- `cupy` (for GPU acceleration)
- `numba`
- `vispy` (optional, for optimized real-time plotting)

Install dependencies using:
```
pip install numpy matplotlib cupy numba vispy
```

## Running the Simulation

Run the optimized SPH simulation:
```
python optimized_sph_simulation.py
```

## Performance Improvements