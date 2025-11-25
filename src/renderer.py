"""OpenCV-based renderer for visualization."""

import numpy as np
import cv2
from typing import Optional

from .vehicle import VehicleDynamics
from .curves import CurveGenerator
from .config import RendererConfig, RENDERER_CONFIG


class OpenCVRenderer:
    """OpenCV-based renderer for the curve following environment."""

    def __init__(
        self,
        config: Optional[RendererConfig] = None,
        render_mode: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        background_color: Optional[tuple[int, int, int]] = None,
        curve_color: Optional[tuple[int, int, int]] = None,
        vehicle_color: Optional[tuple[int, int, int]] = None,
        target_color: Optional[tuple[int, int, int]] = None,
        trail_color: Optional[tuple[int, int, int]] = None,
        window_name: Optional[str] = None,
    ):
        """
        Initialize the renderer.

        Args:
            config: Renderer configuration (uses defaults if None)
            render_mode: 'human' for display, 'rgb_array' for array output
            width: Window width in pixels
            height: Window height in pixels
            background_color: BGR background color
            curve_color: BGR color for the curve
            vehicle_color: BGR color for the vehicle
            target_color: BGR color for target/lookahead points
            trail_color: BGR color for vehicle trail
            window_name: Name of the display window
        """
        cfg = config or RENDERER_CONFIG

        self.width = width if width is not None else cfg.width
        self.height = height if height is not None else cfg.height
        self.render_mode = render_mode if render_mode is not None else "human"
        self.background_color = background_color if background_color is not None else cfg.background_color
        self.curve_color = curve_color if curve_color is not None else cfg.curve_color
        self.vehicle_color = vehicle_color if vehicle_color is not None else cfg.vehicle_color
        self.target_color = target_color if target_color is not None else cfg.target_color
        self.trail_color = trail_color if trail_color is not None else cfg.trail_color
        self.window_name = window_name if window_name is not None else cfg.window_name

        # Camera/view state
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.pixels_per_meter = cfg.pixels_per_meter
        self.follow_vehicle = True

        # Config for zoom limits and margins
        self._cfg = cfg

        # Trail history
        self.trail_history: list[tuple[float, float]] = []
        self.max_trail_length = cfg.max_trail_length

        # Window state
        self._window_created = False

    def world_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.camera_x) * self.pixels_per_meter + self.width / 2)
        screen_y = int(self.height / 2 - (y - self.camera_y) * self.pixels_per_meter)
        return screen_x, screen_y

    def render(
        self,
        curve: CurveGenerator,
        vehicle: VehicleDynamics,
        info: dict,
    ) -> Optional[np.ndarray]:
        """
        Render the current state.

        Args:
            curve: The curve being followed
            vehicle: The vehicle
            info: Additional info to display

        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise
        """
        # Create frame
        frame = np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8)

        # Update camera to follow vehicle
        if self.follow_vehicle:
            self.camera_x = vehicle.state.x
            self.camera_y = vehicle.state.y

        # Update trail
        self.trail_history.append((vehicle.state.x, vehicle.state.y))
        if len(self.trail_history) > self.max_trail_length:
            self.trail_history.pop(0)

        # Draw curve
        self._draw_curve(frame, curve)

        # Draw trail
        self._draw_trail(frame)

        # Draw vehicle
        self._draw_vehicle(frame, vehicle)

        # Draw closest point on curve
        position = np.array([vehicle.state.x, vehicle.state.y])
        idx, distance, closest = curve.find_closest_point(position)
        closest_screen = self.world_to_screen(closest[0], closest[1])
        cv2.circle(frame, closest_screen, 5, self.target_color, -1)

        # Draw line from vehicle to closest point
        vehicle_screen = self.world_to_screen(vehicle.state.x, vehicle.state.y)
        cv2.line(frame, vehicle_screen, closest_screen, (150, 150, 150), 1)

        # Draw info overlay
        self._draw_info(frame, vehicle, info, distance)

        # Display or return
        if self.render_mode == "human":
            if not self._window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.window_name, self.width, self.height)
                self._window_created = True
            cv2.imshow(self.window_name, frame)
            cv2.waitKey(1)
            return None
        else:
            # Convert BGR to RGB for rgb_array mode
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _draw_curve(self, frame: np.ndarray, curve: CurveGenerator):
        """Draw the curve on the frame."""
        points = curve.points
        margin = self._cfg.visibility_margin

        # Convert to screen coordinates
        screen_points = []
        for x, y in points:
            sx, sy = self.world_to_screen(x, y)
            # Only include points that might be visible (with margin)
            if -margin < sx < self.width + margin and -margin < sy < self.height + margin:
                screen_points.append([sx, sy])

        if len(screen_points) > 1:
            screen_points = np.array(screen_points, dtype=np.int32)
            cv2.polylines(frame, [screen_points], isClosed=False, color=self.curve_color, thickness=2)

    def _draw_trail(self, frame: np.ndarray):
        """Draw the vehicle's trail."""
        if len(self.trail_history) < 2:
            return

        margin = self._cfg.trail_margin

        for i in range(1, len(self.trail_history)):
            # Fade color based on age
            alpha = i / len(self.trail_history)
            color = tuple(int(c * alpha) for c in self.trail_color)

            p1 = self.world_to_screen(*self.trail_history[i - 1])
            p2 = self.world_to_screen(*self.trail_history[i])

            # Only draw if both points are roughly on screen
            if (-margin < p1[0] < self.width + margin and -margin < p1[1] < self.height + margin and
                -margin < p2[0] < self.width + margin and -margin < p2[1] < self.height + margin):
                cv2.line(frame, p1, p2, color, 2)

    def _draw_vehicle(self, frame: np.ndarray, vehicle: VehicleDynamics):
        """Draw the vehicle on the frame."""
        # Get vehicle corners
        corners = vehicle.get_corners()
        screen_corners = np.array([self.world_to_screen(x, y) for x, y in corners], dtype=np.int32)

        # Draw vehicle body
        cv2.fillPoly(frame, [screen_corners], self.vehicle_color)
        cv2.polylines(frame, [screen_corners], isClosed=True, color=(255, 255, 255), thickness=1)

        # Draw heading direction
        center = self.world_to_screen(vehicle.state.x, vehicle.state.y)
        heading_length = 20
        heading_end = (
            int(center[0] + heading_length * np.cos(-vehicle.state.theta + np.pi/2)),
            int(center[1] + heading_length * np.sin(-vehicle.state.theta + np.pi/2))
        )
        # Correct the heading visualization
        heading_end = (
            int(center[0] + heading_length * np.cos(vehicle.state.theta) * self.pixels_per_meter / 8),
            int(center[1] - heading_length * np.sin(vehicle.state.theta) * self.pixels_per_meter / 8)
        )
        cv2.arrowedLine(frame, center, heading_end, (255, 255, 255), 2, tipLength=0.3)

        # Draw steering indicator
        if abs(vehicle.state.steering) > 0.01:
            steer_color = (0, 200, 255) if vehicle.state.steering > 0 else (255, 200, 0)
            steer_length = int(abs(vehicle.state.steering) / vehicle.max_steering * 30)
            steer_angle = vehicle.state.theta + (np.pi/2 if vehicle.state.steering > 0 else -np.pi/2)
            steer_end = (
                int(center[0] + steer_length * np.cos(steer_angle)),
                int(center[1] - steer_length * np.sin(steer_angle))
            )
            cv2.line(frame, center, steer_end, steer_color, 2)

    def _draw_info(self, frame: np.ndarray, vehicle: VehicleDynamics, info: dict, distance: float):
        """Draw information overlay."""
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # Text info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        y_offset = 30
        line_height = 22

        texts = [
            f"Step: {info.get('step', 0)}",
            f"Position: ({vehicle.state.x:.1f}, {vehicle.state.y:.1f})",
            f"Heading: {np.degrees(vehicle.state.theta):.1f} deg",
            f"Velocity: {vehicle.state.velocity:.2f} m/s",
            f"Steering: {np.degrees(vehicle.state.steering):.1f} deg",
            f"Cross-track: {distance:.2f} m",
        ]

        for i, text in enumerate(texts):
            cv2.putText(frame, text, (20, y_offset + i * line_height), font, font_scale, color, 1)

        # Progress bar for velocity
        bar_x = 20
        bar_y = y_offset + len(texts) * line_height + 5
        bar_width = 200
        bar_height = 10
        fill_width = int(bar_width * vehicle.state.velocity / vehicle.max_velocity)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 200, 0), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 1)

    def reset_trail(self):
        """Clear the trail history."""
        self.trail_history.clear()

    def set_zoom(self, pixels_per_meter: float):
        """Set the zoom level."""
        self.pixels_per_meter = max(self._cfg.min_zoom, min(self._cfg.max_zoom, pixels_per_meter))

    def close(self):
        """Close the renderer and release resources."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False
