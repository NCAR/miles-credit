import torch
from credit.spline import make_smoothing_spline_pytorch
from typing import Sequence, Optional


class SplineCoefficients(torch.nn.Module):
    """
    Represents a neural network module for generating spline coefficients.

    This class is designed to handle the computation of spline coefficients
    using a specified degree, number of knots, interior knot positions, and a
    regularization parameter. It provides flexibility in defining the degree of
    the spline, managing the placement of knots, and controlling the regularization
    strength. The module is intended to be used to convert vertical atmospheric profile
    data to a reduced set of spline coefficients.

    Attributes:
        variables (Sequence[str]): A sequence of variable names following the `source/category/variable` format.
            If `source/category` is provided, then all variables in that category are transformed.

        degree (int): The degree of the spline. Affects the smoothness and
            continuity of the spline.
        knots_interior (Optional[Sequence[float]]): Specific positions of interior
            knots for the spline. If None, the knots are uniformly placed.
        num_knots (int): The total number of knots to use for the spline. If
            `knots_interior` is provided, this parameter is ignored.
        lam (float): A regularization parameter to control the smoothness of
            the spline. Higher values result in smoother splines.
    """

    def __init__(
        self,
        variables: Sequence[str],
        degree: int = 3,
        knots_interior: Optional[Sequence[float]] = None,
        num_knots: int = 20,
        lam: float = 1.0,
    ):
        super().__init__()
        self.variables = variables
        self.degree = degree
        self.knots_interior = knots_interior
        self.num_knots = num_knots
        self.lam = lam

    def forward(self, x):
        """

        Args:
            x:

        Returns:

        """
        make_smoothing_spline_pytorch()
        return
