import torch


def bspline_basis_matrix(x: torch.Tensor, t: torch.Tensor, k: int) -> torch.Tensor:
    """
    Computes the B-spline basis matrix for 1D data.

    Uses the Cox-de Boor recursion formula in a vectorized, bottom-up
    computation.

    Args:
        x (torch.Tensor): 1D tensor of shape (n,) of points to evaluate.
        t (torch.Tensor): 1D tensor of shape (m,) of knot vector.
                         Must be sorted.
        k (int): Degree of the spline.

    Returns:
        torch.Tensor: The B-spline basis matrix of shape (n, nc), where
                      nc = m - k - 1 is the number of basis functions.
                      B[i, j] = B_{j, k}(x_i).
    """
    n = x.shape[0]
    m = t.shape[0]

    # Ensure x is (n, 1) for broadcasting
    if x.dim() == 1:
        x = x.unsqueeze(1)

    # k = 0
    # B_{i, 0}(x) = 1 if t_i <= x < t_{i+1} else 0
    # This gives m-1 basis functions
    B = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)  # Shape (n, m-1)

    # k > 0
    # B_{i, k}(x) = w_{i, k} * B_{i, k-1}(x) + (1 - w_{i+1, k}) * B_{i+1, k-1}(x)
    # w_{i, k}(x) = (x - t_i) / (t_{i+k} - t_i)

    for deg in range(1, k + 1):
        # B_prev has shape (n, m - 1 - (deg-1))
        # B_curr will have shape (n, m - 1 - deg)

        # nc_prev = m - 1 - (deg - 1)
        nc_curr = m - 1 - deg

        # Calculate first term
        # i runs from 0 to nc_curr - 1
        t_i = t[:nc_curr]  # t_i for i = 0..nc_curr-1
        t_ideg = t[deg : nc_curr + deg]  # t_{i+deg} for i = 0..nc_curr-1

        den1 = t_ideg - t_i
        num1 = x - t_i
        # Handle 0/0 division, replace with 0
        # Original: w1 = torch.where(den1 > 1e-9, num1 / den1, 0.0) # Shape (n, nc_curr)

        # Differentiable safe division: (num1 * den1) / (den1**2 + 1e-9)
        # This approximates num1/den1 when den1 is non-zero, and 0.0 when den1 is zero.
        w1 = num1 * den1 / (den1.pow(2) + 1e-9)  # Shape (n, nc_curr)

        # B_prev has nc_curr + 1 columns. B[:, :-1] selects B_{i, deg-1}
        term1 = w1 * B[:, :-1]

        # Calculate second term
        # i runs from 0 to nc_curr - 1
        t_i1 = t[1 : nc_curr + 1]  # t_{i+1} for i = 0..nc_curr-1
        t_i1deg = t[1 + deg : nc_curr + 1 + deg]  # t_{i+1+deg} for i = 0..nc_curr-1

        den2 = t_i1deg - t_i1
        num2 = t_i1deg - x
        # Handle 0/0 division, replace with 0
        # Original: w2 = torch.where(den2 > 1e-9, num2 / den2, 0.0) # Shape (n, nc_curr)

        # Differentiable safe division: (num2 * den2) / (den2**2 + 1e-9)
        # This approximates num2/den2 when den2 is non-zero, and 0.0 when den2 is zero.
        w2 = num2 * den2 / (den2.pow(2) + 1e-9)  # Shape (n, nc_curr)

        # B[:, 1:] selects B_{i+1, deg-1}
        term2 = w2 * B[:, 1:]

        B = term1 + term2  # Shape (n, nc_curr)

    return B


def splev_pytorch(x_eval: torch.Tensor, t: torch.Tensor, c: torch.Tensor, k: int) -> torch.Tensor:
    """
    Evaluates a B-spline at new points, given its parameters.

    Equivalent to scipy.interpolate.splev.

    Args:
        x_eval (torch.Tensor): 1D tensor of points to evaluate the spline at.
        t (torch.Tensor): The knot vector (from make_smoothing_spline).
        c (torch.Tensor): The coefficient vector (from make_smoothing_spline).
        k (int): The degree of the spline (from make_smoothing_spline).

    Returns:
        torch.Tensor: 1D tensor of the evaluated spline, S(x_eval).
    """
    # Build the basis matrix for the new x points
    B_eval = bspline_basis_matrix(x_eval, t, k)

    # The spline is S(x) = sum(c_i * B_i,k(x))
    # In matrix form: y = B @ c

    # Ensure c is (nc, 1) for matrix multiplication
    if c.dim() == 1:
        c = c.unsqueeze(1)

    y_eval = B_eval @ c

    return y_eval.squeeze()


def splder_pytorch(t: torch.Tensor, c: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes the parameters of the B-spline representing the derivative.

    Equivalent to scipy.interpolate.splder.

    Args:
        t (torch.Tensor): The knot vector.
        c (torch.Tensor): The coefficient vector.
        k (int): The degree of the spline.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]:
            t_deriv (torch.Tensor): The knot vector for the derivative spline.
            c_deriv (torch.Tensor): The coefficients for the derivative spline.
            k_deriv (int): The degree for the derivative spline (k-1).
    """
    if k == 0:
        # Derivative of a constant (degree 0) spline is 0
        k_deriv = 0
        t_deriv = t[1:-1]  # Maintain consistency
        c_deriv = torch.zeros(t_deriv.shape[0] - k_deriv - 1, device=c.device, dtype=c.dtype)
        return t_deriv, c_deriv, k_deriv

    k_deriv = k - 1
    t_deriv = t[1:-1]  # Derivative spline knots

    nc = c.shape[0]

    # Denominator: t_{i+k} - t_i
    # Here, i runs from 1 to nc-1 (for c_i)
    # So we need t[k+1 : nc+k] - t[1 : nc]
    den = t[k + 1 : nc + k] - t[1:nc]

    # Numerator: c_i - c_{i-1}
    c_diff = c[1:] - c[:-1]

    # Differentiable safe division
    c_deriv = k * c_diff * den / (den.pow(2) + 1e-9)

    return t_deriv, c_deriv


def splantider_pytorch(t: torch.Tensor, c: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes the parameters of the B-spline representing the antiderivative (integral).

    Equivalent to scipy.interpolate.splantider.

    Args:
        t (torch.Tensor): The knot vector.
        c (torch.Tensor): The coefficient vector.
        k (int): The degree of the spline.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]:
            t_int (torch.Tensor): The knot vector for the integral spline.
            c_int (torch.Tensor): The coefficients for the integral spline.
            k_int (int): The degree for the integral spline (k+1).
    """
    k_int = k + 1
    nc = c.shape[0]

    # New knot vector (add one knot at each end)
    t_int = torch.cat([t[:1], t, t[-1:]])

    # New coefficients
    c_int = torch.zeros(nc + 1, device=c.device, dtype=c.dtype)

    # Factors for cumulative sum
    # (t_{j+k+1} - t_j) / (k+1)
    factors = (t[k + 1 : nc + k + 1] - t[:nc]) / k_int

    # Terms to be summed
    c_terms = factors * c

    # Cumulative sum
    c_int[1:] = torch.cumsum(c_terms, dim=0)

    return t_int, c_int, k_int


def splint_pytorch(a: float, b: float, t: torch.Tensor, c: torch.Tensor, k: int) -> torch.Tensor:
    """
    Calculates the definite integral of a B-spline from a to b.

    Equivalent to scipy.interpolate.splint.

    Args:
        a (float): The lower bound of integration.
        b (float): The upper bound of integration.
        t (torch.Tensor): The knot vector.
        c (torch.Tensor): The coefficient vector.
        k (int): The degree of the spline.

    Returns:
        torch.Tensor: A scalar tensor representing the definite integral.
    """
    # Get the antiderivative spline
    t_int, c_int, k_int = splantider_pytorch(t, c, k)

    # Evaluate the antiderivative at b and a
    x_eval = torch.tensor([a, b], device=t.device, dtype=t.dtype)
    y_eval = splev_pytorch(x_eval, t_int, c_int, k_int)

    # Integral = S(b) - S(a)
    return y_eval[1] - y_eval[0]


def make_smoothing_spline_pytorch(
    x: torch.Tensor,
    y: torch.Tensor,
    k: int = 3,
    knots_interior: torch.Tensor = None,
    num_knots: int = 20,
    lam: float = 1.0,
    penalty_order: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Computes a P-spline (Penalized B-spline) smoother.

    This function is a PyTorch-based equivalent to the idea of
    scipy.interpolate.make_smoothing_spline. It finds the B-spline
    coefficients `c` that minimize:

    ||y - B @ c||^2 + lam * ||D @ c||^2

    where `B` is the B-spline basis matrix and `D` is a finite difference
    matrix that penalizes roughness.

    Args:
        x (torch.Tensor): 1D tensor of x-data.
        y (torch.Tensor): 1D tensor of y-data (must be same length as x).
        k (int, optional): Degree of the B-spline. Defaults to 3 (cubic).
        knots_interior (torch.Tensor, optional): Specify the interior knots.
            If None, `num_knots` uniform knots will be created.
        num_knots (int, optional): Number of interior knots to use if
            `knots_interior` is not provided. Defaults to 20.
        lam (float, optional): The smoothing parameter (lambda).
            Larger values = smoother. Defaults to 1.0.
        penalty_order (int, optional): The order of the finite difference
            penalty. d=2 (default) penalizes the 2nd difference (like
            a second derivative), d=1 penalizes the 1st difference, etc.

    Returns:
        tuple[torch.Tensor, torch.Tensor, int]:
            t (torch.Tensor): The full (padded) knot vector.
            c (torch.Tensor): The B-spline coefficient vector.
            k (int): The degree of the spline.
    """

    # --- 1. Prepare Data and Knots ---

    # Sort x and y
    # x_sorted, sort_idx = torch.sort(x)
    # y_sorted = y[sort_idx]
    # Handle duplicate x values by averaging y
    # unique() is also available, but this is more robust
    # unique_x, inverse_indices = torch.unique(x_sorted, return_inverse=True)

    # unique_y = torch.zeros_like(unique_x, dtype=y.dtype)
    # unique_y = unique_y.scatter_add(0, inverse_indices, y_sorted)

    # counts = torch.zeros_like(unique_x, dtype=torch.int)
    # counts = counts.scatter_add(0, inverse_indices, torch.ones_like(y_sorted, dtype=torch.int))

    # unique_y = unique_y / counts.to(y.dtype)

    # x_data, y_data = unique_x, unique_y
    x_data, y_data = x, y
    n = x_data.size()

    # Create interior knots if not provided
    if knots_interior is None:
        knots_interior = torch.linspace(x_data.min(), x_data.max(), num_knots, device=x.device, dtype=x.dtype)

    # Create the full knot vector `t` by padding
    # We use a "clamped" spline by repeating end knots k times
    t = torch.cat(
        [
            torch.full((k,), knots_interior[0].item(), device=x.device, dtype=x.dtype),
            knots_interior,
            torch.full((k,), knots_interior[-1].item(), device=x.device, dtype=x.dtype),
        ]
    )

    # --- 2. Build Basis Matrix B ---
    # B will have shape (n, nc)
    # nc = m - k - 1 = (n_interior + 2k) - k - 1 = n_interior + k - 1
    B = bspline_basis_matrix(x_data, t, k)
    nc = B.shape[1]

    # --- 3. Build Penalty Matrix D ---
    # D is the matrix for the d-th order finite difference
    # D has shape (nc - d, nc)
    d = penalty_order
    D = torch.zeros(nc - d, nc, device=x.device, dtype=x.dtype)

    if d == 1:  # Penalizes c_i - c_{i-1}
        idx = torch.arange(nc - 1)
        D[idx, idx] = -1.0
        D[idx, idx + 1] = 1.0
    elif d == 2:  # Penalizes (c_i - c_{i-1}) - (c_{i-1} - c_{i-2}) = c_i - 2*c_{i-1} + c_{i-2}
        idx = torch.arange(nc - 2)
        D[idx, idx] = 1.0
        D[idx, idx + 1] = -2.0
        D[idx, idx + 2] = 1.0
    else:
        # General case (less common)
        D_prev = torch.eye(nc, device=x.device, dtype=x.dtype)
        for i in range(d):
            D_prev = D_prev[1:] - D_prev[:-1]
        D = D_prev

    # The penalty matrix (quadratic form) is R = D.T @ D
    R = D.T @ D

    # --- 4. Solve the Linear System ---
    # (B.T @ B + lam * R) @ c = B.T @ y

    BTB = B.T @ B
    L = lam * R

    # Add a small ridge (jitter) for numerical stability
    jitter = torch.eye(nc, device=x.device, dtype=x.dtype) * 1e-6

    A = BTB + L + jitter
    b = B.T @ y_data

    # Solve A @ c = b
    try:
        c = torch.linalg.solve(A, b)
    except torch.linalg.LinAlgError as e:
        print(f"Linear algebra error: {e}. Matrix may be singular.")
        print("Try increasing `lam` or `num_knots`.")
        # Fallback to pseudo-inverse (slower but more stable)
        A_pinv = torch.linalg.pinv(A)
        c = A_pinv @ b

    return c
