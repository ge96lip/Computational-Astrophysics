import numpy as np 
import numexpr as ne
def gradW( x, y, z, h ):
	"""
	Gradient of the Gausssian Smoothing kernel (3D)
	x     is a vector/matrix of x positions
	y     is a vector/matrix of y positions
	z     is a vector/matrix of z positions
	h     is the smoothing length
	wx, wy, wz     is the evaluated gradient
	"""
	
	r = np.sqrt(x**2 + y**2 + z**2)
	
	n = -2 * np.exp( -r**2 / h**2) / h**5 / (np.pi)**(3/2)
	wx = n * x
	wy = n * y
	wz = n * z
	
	return wx, wy, wz
def gradW_float32_ne(x, y, z, h, N):
    """Optimized gradient computation using numexpr."""
    h_np = np.asarray(h)  # Ensure `h` is a NumPy array

    x_np = np.asarray(x)
    y_np = np.asarray(y)
    z_np = np.asarray(z)
    pi = np.pi
    r2 = ne.evaluate("x_np**2 + y_np**2 + z_np**2")
    exp_term = ne.evaluate("exp(-r2 / h_np**2)")
    scale = ne.evaluate("-2 * exp_term / (h_np**5 * (pi**(3/2)))")

    wx = ne.evaluate("scale * x_np")
    wy = ne.evaluate("scale * y_np")
    wz = ne.evaluate("scale * z_np")

    return wx, wy, wz

def gradW_float32(x, y, z, h):
    """
    Optimized gradient of the Gaussian Smoothing kernel (3D) using float32 for better performance on M1.
    """
    r2 = x**2 + y**2 + z**2
    r = np.sqrt(r2)
    
    n = -2 * np.exp(-r2 / h**2) / (h**5 * (np.pi)**(3/2))
    
    return n * x, n * y, n * z

def gradW_inplace(x, y, z, h, N):
    """
    Optimized gradient of the Gaussian Smoothing kernel (3D) using in-place operations,
    while ensuring correct numerical results.
    """
    wx = np.empty((N, N), dtype=np.float32)
    wy = np.empty((N, N), dtype=np.float32)
    wz = np.empty((N, N), dtype=np.float32)
    r2 = x*x + y*y + z*z  

    # Compute exponential term and scale factor in-place
    np.exp(-r2 / h**2, out=r2)  # Store exp(-r^2 / h^2) in r2
    r2 *= -2 / (h**5 * (np.pi)**(3/2))  # Apply scaling factor in-place

    # Compute gradient components in one pass
    np.multiply(r2, x, out=wx)
    np.multiply(r2, y, out=wy)
    np.multiply(r2, z, out=wz)

    return wx, wy, wz


def gradW_float32_inplace(x, y, z, h, N):
    """
    Fully optimized in-place gradient computation using float32.
    - Removes unnecessary temporary arrays.
    - Uses in-place operations to minimize memory overhead.
    - Leverages NumPy’s `out=` parameter for efficiency.
    """
    h_np = np.asarray(h)
    # Allocate output arrays inside function
    wx = np.empty((N, N), dtype=np.float32)
    wy = np.empty((N, N), dtype=np.float32)
    wz = np.empty((N, N), dtype=np.float32)

    # Compute squared distances directly in-place
    r2 = np.empty((N, N), dtype=np.float32)
    np.multiply(x, x, out=r2)
    np.add(r2, y*y, out=r2)
    np.add(r2, z*z, out=r2)  # Now r2 contains x² + y² + z²

    # Compute exponential term in-place
    np.divide(r2, -h_np**2, out=r2)
    np.exp(r2, out=r2)  # Now r2 contains exp(-r² / h²)

    # Compute the scaling factor in-place
    np.multiply(r2, -2 / (h_np**5 * (np.pi)**(3/2)), out=r2)

    # Compute wx, wy, wz in-place
    np.multiply(r2, x, out=wx)
    np.multiply(r2, y, out=wy)
    np.multiply(r2, z, out=wz)

    return wx, wy, wz

def gradW_float32_ne(x, y, z, h, N):
    """Optimized gradient computation using numexpr."""
    h_np = np.asarray(h)  # Ensure `h` is a NumPy array

    x_np = np.asarray(x)
    y_np = np.asarray(y)
    z_np = np.asarray(z)
    pi = np.pi
    r2 = ne.evaluate("x_np**2 + y_np**2 + z_np**2")
    exp_term = ne.evaluate("exp(-r2 / h_np**2)")
    scale = ne.evaluate("-2 * exp_term / (h_np**5 * (pi**(3/2)))")

    wx = ne.evaluate("scale * x_np")
    wy = ne.evaluate("scale * y_np")
    wz = ne.evaluate("scale * z_np")

    return wx, wy, wz