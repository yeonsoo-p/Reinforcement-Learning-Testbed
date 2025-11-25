"""Curve generators for the agent to follow."""

import numpy as np
from abc import ABC, abstractmethod
from scipy.interpolate import CubicSpline
from typing import Optional


class CurveGenerator(ABC):
    """Abstract base class for curve generators."""

    def __init__(self, num_points: int = 1000):
        """
        Initialize curve generator.

        Args:
            num_points: Number of points to sample along the curve
        """
        self.num_points = num_points
        self._points: Optional[np.ndarray] = None
        self._tangents: Optional[np.ndarray] = None
        self._curvatures: Optional[np.ndarray] = None
        self._arc_lengths: Optional[np.ndarray] = None

    @abstractmethod
    def generate(self) -> np.ndarray:
        """Generate curve points. Returns array of shape (num_points, 2)."""
        pass

    @property
    def points(self) -> np.ndarray:
        """Get curve points, generating if necessary."""
        if self._points is None:
            self._points = self.generate()
            self._compute_derivatives()
        return self._points

    @property
    def tangents(self) -> np.ndarray:
        """Get tangent vectors at each point."""
        if self._tangents is None:
            _ = self.points  # Trigger generation
        return self._tangents

    @property
    def curvatures(self) -> np.ndarray:
        """Get curvature at each point."""
        if self._curvatures is None:
            _ = self.points  # Trigger generation
        return self._curvatures

    @property
    def arc_lengths(self) -> np.ndarray:
        """Get cumulative arc length at each point."""
        if self._arc_lengths is None:
            _ = self.points  # Trigger generation
        return self._arc_lengths

    @property
    def total_length(self) -> float:
        """Get total arc length of curve."""
        return self.arc_lengths[-1]

    def _compute_derivatives(self):
        """Compute tangents, curvatures, and arc lengths."""
        points = self._points

        # Compute tangents using central differences
        tangents = np.zeros_like(points)
        tangents[1:-1] = points[2:] - points[:-2]
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]

        # Normalize tangents
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        self._tangents = tangents / norms

        # Compute curvature using the formula: k = |dx * d2y - dy * d2x| / (dx^2 + dy^2)^(3/2)
        dx = np.gradient(points[:, 0])
        dy = np.gradient(points[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)

        denominator = (dx**2 + dy**2) ** 1.5
        denominator = np.maximum(denominator, 1e-8)
        self._curvatures = np.abs(dx * d2y - dy * d2x) / denominator

        # Compute arc lengths
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        self._arc_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])

    def find_closest_point(self, position: np.ndarray) -> tuple[int, float, np.ndarray]:
        """
        Find the closest point on the curve to a given position.

        Args:
            position: [x, y] position to find closest point to

        Returns:
            Tuple of (index, distance, closest_point)
        """
        distances = np.linalg.norm(self.points - position, axis=1)
        idx = np.argmin(distances)
        return idx, distances[idx], self.points[idx]

    def get_lookahead_point(self, position: np.ndarray, lookahead_distance: float) -> tuple[np.ndarray, float]:
        """
        Get a point on the curve ahead of the current position.

        Args:
            position: Current [x, y] position
            lookahead_distance: Distance to look ahead along the curve

        Returns:
            Tuple of (lookahead_point, heading_angle)
        """
        idx, _, _ = self.find_closest_point(position)
        current_arc = self.arc_lengths[idx]
        target_arc = current_arc + lookahead_distance

        # Handle wrapping for closed curves
        if target_arc > self.total_length:
            target_arc = target_arc % self.total_length

        # Find point at target arc length
        target_idx = np.searchsorted(self.arc_lengths, target_arc)
        target_idx = min(target_idx, len(self.points) - 1)

        lookahead_point = self.points[target_idx]
        heading = np.arctan2(self.tangents[target_idx, 1], self.tangents[target_idx, 0])

        return lookahead_point, heading

    def get_cross_track_error(self, position: np.ndarray, heading: float) -> tuple[float, float]:
        """
        Compute cross-track error and heading error.

        Args:
            position: Current [x, y] position
            heading: Current heading angle in radians

        Returns:
            Tuple of (cross_track_error, heading_error)
        """
        idx, distance, closest = self.find_closest_point(position)

        # Compute signed cross-track error
        tangent = self.tangents[idx]
        to_position = position - closest

        # Cross product gives signed distance (positive = left of curve)
        cross_track = tangent[0] * to_position[1] - tangent[1] * to_position[0]

        # Heading error
        curve_heading = np.arctan2(tangent[1], tangent[0])
        heading_error = heading - curve_heading

        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

        return cross_track, heading_error

    def reset(self):
        """Reset the curve (regenerate if random)."""
        self._points = None
        self._tangents = None
        self._curvatures = None
        self._arc_lengths = None


class SineCurve(CurveGenerator):
    """Sinusoidal curve."""

    def __init__(
        self,
        amplitude: float = 20.0,
        wavelength: float = 50.0,
        num_periods: float = 2.0,
        num_points: int = 1000,
    ):
        super().__init__(num_points)
        self.amplitude = amplitude
        self.wavelength = wavelength
        self.num_periods = num_periods

    def generate(self) -> np.ndarray:
        length = self.wavelength * self.num_periods
        x = np.linspace(0, length, self.num_points)
        y = self.amplitude * np.sin(2 * np.pi * x / self.wavelength)
        return np.column_stack([x, y])


class ArcCurve(CurveGenerator):
    """Open circular arc (partial circle) - no overlap."""

    def __init__(
        self,
        radius: float = 30.0,
        center: tuple[float, float] = (0.0, 0.0),
        start_angle: float = 0.0,
        arc_angle: float = np.pi * 1.5,  # 270 degrees by default
        num_points: int = 1000,
    ):
        """
        Args:
            radius: Radius of the arc
            center: Center point of the arc
            start_angle: Starting angle in radians
            arc_angle: Total angle to sweep (positive = counter-clockwise)
        """
        super().__init__(num_points)
        self.radius = radius
        self.center = np.array(center)
        self.start_angle = start_angle
        self.arc_angle = arc_angle

    def generate(self) -> np.ndarray:
        theta = np.linspace(self.start_angle, self.start_angle + self.arc_angle, self.num_points)
        x = self.center[0] + self.radius * np.cos(theta)
        y = self.center[1] + self.radius * np.sin(theta)
        return np.column_stack([x, y])


class SCurve(CurveGenerator):
    """S-shaped curve (open, non-overlapping) using connected arcs."""

    def __init__(
        self,
        length: float = 100.0,
        amplitude: float = 30.0,
        num_points: int = 1000,
    ):
        """
        Args:
            length: Total horizontal length of the S-curve
            amplitude: Vertical amplitude (half the total height)
        """
        super().__init__(num_points)
        self.length = length
        self.amplitude = amplitude

    def generate(self) -> np.ndarray:
        # Create smooth S-curve using sine-based transition
        t = np.linspace(0, 1, self.num_points)
        x = t * self.length
        # Smooth S-curve using sigmoid-like function
        y = self.amplitude * (2 / (1 + np.exp(-6 * (t - 0.5))) - 1)
        return np.column_stack([x, y])


class SpiralCurve(CurveGenerator):
    """Archimedean spiral - open, non-overlapping curve."""

    def __init__(
        self,
        initial_radius: float = 5.0,
        growth_rate: float = 3.0,
        num_turns: float = 2.5,
        center: tuple[float, float] = (0.0, 0.0),
        num_points: int = 1000,
    ):
        """
        Args:
            initial_radius: Starting radius of spiral
            growth_rate: How much radius increases per radian
            num_turns: Number of complete turns
            center: Center point of the spiral
        """
        super().__init__(num_points)
        self.initial_radius = initial_radius
        self.growth_rate = growth_rate
        self.num_turns = num_turns
        self.center = np.array(center)

    def generate(self) -> np.ndarray:
        theta = np.linspace(0, 2 * np.pi * self.num_turns, self.num_points)
        r = self.initial_radius + self.growth_rate * theta
        x = self.center[0] + r * np.cos(theta)
        y = self.center[1] + r * np.sin(theta)
        return np.column_stack([x, y])


class HairpinCurve(CurveGenerator):
    """Hairpin/switchback curve - open with tight turns."""

    def __init__(
        self,
        num_turns: int = 3,
        turn_radius: float = 15.0,
        straight_length: float = 40.0,
        spacing: float = 35.0,
        num_points: int = 1000,
    ):
        """
        Args:
            num_turns: Number of hairpin turns
            turn_radius: Radius of the U-turns
            straight_length: Length of straight sections
            spacing: Vertical spacing between parallel sections
        """
        super().__init__(num_points)
        self.num_turns = num_turns
        self.turn_radius = turn_radius
        self.straight_length = straight_length
        self.spacing = spacing

    def generate(self) -> np.ndarray:
        points = []
        current_x = 0.0
        current_y = 0.0
        direction = 1  # 1 = right, -1 = left

        points_per_segment = self.num_points // (2 * self.num_turns + 1)

        for i in range(self.num_turns):
            # Straight section
            x_straight = np.linspace(current_x, current_x + direction * self.straight_length, points_per_segment)
            y_straight = np.full_like(x_straight, current_y)
            points.append(np.column_stack([x_straight, y_straight]))
            current_x = current_x + direction * self.straight_length

            # U-turn (semicircle)
            if direction == 1:
                center_x = current_x
                center_y = current_y + self.turn_radius
                angles = np.linspace(-np.pi / 2, np.pi / 2, points_per_segment)
            else:
                center_x = current_x
                center_y = current_y + self.turn_radius
                angles = np.linspace(3 * np.pi / 2, np.pi / 2, points_per_segment)

            x_turn = center_x + self.turn_radius * np.cos(angles) * (-direction)
            y_turn = center_y + self.turn_radius * np.sin(angles)
            points.append(np.column_stack([x_turn, y_turn]))

            current_y = current_y + 2 * self.turn_radius
            direction *= -1

        # Final straight section
        x_final = np.linspace(current_x, current_x + direction * self.straight_length, points_per_segment)
        y_final = np.full_like(x_final, current_y)
        points.append(np.column_stack([x_final, y_final]))

        # Combine all segments
        all_points = np.vstack(points)

        # Resample to exact num_points
        indices = np.linspace(0, len(all_points) - 1, self.num_points).astype(int)
        return all_points[indices]


class RandomCurve(CurveGenerator):
    """Randomly generated smooth open curve using cubic splines - guaranteed non-overlapping."""

    def __init__(
        self,
        num_control_points: int = 8,
        bounds: tuple[float, float, float, float] = (-50, 50, -50, 50),
        seed: Optional[int] = None,
        num_points: int = 1000,
    ):
        """
        Initialize random curve generator.

        Args:
            num_control_points: Number of random control points
            bounds: (x_min, x_max, y_min, y_max) for control points
            seed: Random seed for reproducibility
            num_points: Number of points to sample along the curve
        """
        super().__init__(num_points)
        self.num_control_points = num_control_points
        self.bounds = bounds
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _segments_intersect(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> bool:
        """Check if line segment p1-p2 intersects with p3-p4."""
        d1 = p2 - p1
        d2 = p4 - p3
        d3 = p3 - p1

        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-10:
            return False  # Parallel lines

        t = (d3[0] * d2[1] - d3[1] * d2[0]) / cross
        u = (d3[0] * d1[1] - d3[1] * d1[0]) / cross

        # Check if intersection is within both segments (excluding endpoints)
        return 0.01 < t < 0.99 and 0.01 < u < 0.99

    def _curve_self_intersects(self, points: np.ndarray, sample_step: int = 10) -> bool:
        """Check if curve self-intersects by sampling segments."""
        n = len(points)
        sampled_indices = list(range(0, n - 1, sample_step))
        if sampled_indices[-1] != n - 2:
            sampled_indices.append(n - 2)

        for i, idx_i in enumerate(sampled_indices[:-2]):
            for idx_j in sampled_indices[i + 2:]:
                # Skip adjacent segments
                if abs(idx_i - idx_j) < sample_step * 2:
                    continue
                if self._segments_intersect(
                    points[idx_i], points[idx_i + sample_step] if idx_i + sample_step < n else points[-1],
                    points[idx_j], points[idx_j + sample_step] if idx_j + sample_step < n else points[-1]
                ):
                    return True
        return False

    def _generate_non_overlapping_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate control points that produce a non-overlapping curve."""
        x_min, x_max, y_min, y_max = self.bounds

        # Strategy: Generate points along a generally monotonic path with random perturbations
        # This naturally avoids self-intersection

        # Choose a primary direction (mostly horizontal or vertical)
        if self._rng.random() > 0.5:
            # Horizontal primary direction
            base_x = np.linspace(x_min + 10, x_max - 10, self.num_control_points)
            # Add random perturbation to x (small)
            control_x = base_x + self._rng.uniform(-5, 5, self.num_control_points)
            # Random y values within bounds
            control_y = self._rng.uniform(y_min + 10, y_max - 10, self.num_control_points)
        else:
            # Vertical primary direction
            base_y = np.linspace(y_min + 10, y_max - 10, self.num_control_points)
            control_y = base_y + self._rng.uniform(-5, 5, self.num_control_points)
            control_x = self._rng.uniform(x_min + 10, x_max - 10, self.num_control_points)

        return control_x, control_y

    def generate(self) -> np.ndarray:
        max_attempts = 10

        for attempt in range(max_attempts):
            control_x, control_y = self._generate_non_overlapping_points()

            # Create parameter values for control points
            t_control = np.linspace(0, 1, len(control_x))

            # Fit cubic splines with natural boundary conditions (open curve)
            cs_x = CubicSpline(t_control, control_x, bc_type='natural')
            cs_y = CubicSpline(t_control, control_y, bc_type='natural')

            # Sample the splines
            t_sample = np.linspace(0, 1, self.num_points)
            x = cs_x(t_sample)
            y = cs_y(t_sample)

            points = np.column_stack([x, y])

            # Verify no self-intersection
            if not self._curve_self_intersects(points):
                return points

        # If all attempts fail, return a simple S-curve as fallback
        t = np.linspace(0, 1, self.num_points)
        x_range = self.bounds[1] - self.bounds[0]
        y_range = self.bounds[3] - self.bounds[2]
        x = self.bounds[0] + t * x_range * 0.8 + x_range * 0.1
        y = (self.bounds[2] + self.bounds[3]) / 2 + y_range * 0.3 * np.sin(2 * np.pi * t)
        return np.column_stack([x, y])

    def reset(self):
        """Reset and regenerate with new random points."""
        if self.seed is None:
            self._rng = np.random.default_rng()
        super().reset()


def create_curve(curve_type: str, **kwargs) -> CurveGenerator:
    """
    Factory function to create curves by name.

    Args:
        curve_type: One of 'sine', 'arc', 'scurve', 'spiral', 'hairpin', 'random'
        **kwargs: Arguments to pass to the curve constructor

    Returns:
        CurveGenerator instance

    Available curve types (all open, non-overlapping):
        - sine: Sinusoidal wave
        - arc: Circular arc (partial circle)
        - scurve: S-shaped curve
        - spiral: Archimedean spiral
        - hairpin: Switchback/hairpin turns
        - random: Randomly generated smooth curve
    """
    curves = {
        'sine': SineCurve,
        'arc': ArcCurve,
        'scurve': SCurve,
        'spiral': SpiralCurve,
        'hairpin': HairpinCurve,
        'random': RandomCurve,
    }

    if curve_type not in curves:
        raise ValueError(f"Unknown curve type: {curve_type}. Available: {list(curves.keys())}")

    return curves[curve_type](**kwargs)
