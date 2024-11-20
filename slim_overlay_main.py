from PyQt5.QtWidgets import (QApplication, QMainWindow, QSlider, QVBoxLayout,
                             QWidget, QLabel, QPushButton, QHBoxLayout, QCheckBox,
                             QProgressBar)
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QElapsedTimer
import sys
import numpy as np
from PyQt5.QtGui import QPainter, QColor, QPen, QCursor, QImage
from scipy.ndimage import gaussian_filter
import cv2


class FoodSource:
    def __init__(self, position, radius, strength):
        self.position = np.array(position)
        self.radius = radius
        self.strength = strength
        self.active = True
        self.color = QColor(0, 255, 0)  # Green color for visualization

    def update_position(self, new_position):
        """Update food source position"""
        self.position = np.array(new_position)

    def draw(self, painter, scale_factor):
        """Draw food source visualization"""
        center = QPointF(self.position[0] * scale_factor, self.position[1] * scale_factor)
        # Draw center dot
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(center, 5, 5)


class AgentSystem:
    def __init__(self, num_agents, width, height):
        self.positions = np.random.rand(num_agents, 2) * np.array([width, height])
        directions = np.random.rand(num_agents, 2) - 0.5
        self.directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
        self.count = num_agents
        self.width = width
        self.height = height

        # Physarum parameters
        self.sensor_angle = np.pi / 3
        self.sensor_distance = 4
        self.turn_strength = 0.15
        self.random_turn_prob = 0.02

    def apply_influence_points(self, positions, influence_points):
        """Apply the influence of multiple points on agent directions.
        influence_points: list of (position, strength, radius)
        """
        total_influence = np.zeros_like(positions)

        for point, strength, radius in influence_points:
            point = np.array(point)
            diff = point - positions  # Attraction if strength > 0, repulsion if strength < 0
            distances = np.linalg.norm(diff, axis=1)
            influence_mask = distances < radius

            if np.any(influence_mask):
                normalized_diff = diff[influence_mask] / (distances[influence_mask, np.newaxis] + 1e-10)
                # Adjust strength_factor to allow stronger attraction
                strength_factor = abs(strength) * (1 - distances[influence_mask] / radius)
                strength_factor = np.clip(strength_factor, 0, None)
                influence = np.zeros_like(positions)
                influence_direction = normalized_diff * np.sign(strength)
                influence[influence_mask] = influence_direction * strength_factor[:, np.newaxis]
                total_influence[influence_mask] += influence[influence_mask]

        return total_influence

    def update(self, move_speed, trail_map, influence_points=None,
               typing_speed_multiplier=1.0):
        """
        Update agent positions based on trail map and influence points.
        influence_points: list of (position, strength, radius)
        """
        effective_speed = move_speed * typing_speed_multiplier
        new_positions = self.positions.copy()
        new_directions = self.directions.copy()

        # Apply influence points to directions
        if influence_points:
            total_influence = self.apply_influence_points(
                new_positions,
                influence_points
            )
            new_directions += total_influence

        # Update based on chemical sensing
        for i in range(self.count):
            concentrations = self.sense_chemicals(self.positions[i], self.directions[i], trail_map)

            if concentrations[1] > concentrations[0] and concentrations[1] > concentrations[2]:
                turn_angle = 0
            elif concentrations[0] > concentrations[2]:
                turn_angle = -self.turn_strength
            else:
                turn_angle = self.turn_strength

            if np.random.random() < self.random_turn_prob:
                turn_angle += np.random.uniform(-0.5, 0.5)

            rotation = np.array([
                [np.cos(turn_angle), -np.sin(turn_angle)],
                [np.sin(turn_angle), np.cos(turn_angle)]
            ])
            new_directions[i] = rotation @ new_directions[i]  # Use updated direction

        # Normalize directions
        norms = np.linalg.norm(new_directions, axis=1, keepdims=True)
        new_directions = new_directions / (norms + 1e-10)

        # Update positions
        new_positions += new_directions * effective_speed

        # Wrap around screen
        new_positions %= np.array([self.width, self.height])

        self.positions = new_positions
        self.directions = new_directions

        return self.positions.astype(np.int32)

    def sense_chemicals(self, position, direction, trail_map):
        base_angle = np.arctan2(direction[1], direction[0])
        angles = np.array([-self.sensor_angle, 0, self.sensor_angle]) + base_angle

        sensor_positions = np.array([
            position + self.sensor_distance * np.array([np.cos(angle), np.sin(angle)])
            for angle in angles
        ]).astype(int)

        sensor_positions = sensor_positions % np.array([self.width, self.height])

        try:
            concentrations = np.array([
                trail_map[pos[1], pos[0]]
                for pos in sensor_positions
            ])
        except IndexError:
            return np.zeros(3)

        return concentrations


class ControlPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(40, 40, 40, 200);
                color: white;
                font-size: 12px;
            }
            QSlider::handle {
                background: white;
                border: 1px solid #5c5c5c;
            }
            QPushButton {
                background-color: rgba(60, 60, 60, 200);
                border: 1px solid #5c5c5c;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: rgba(80, 80, 80, 200);
            }
            QLabel {
                color: #CCCCCC;
            }
            QCheckBox {
                color: #CCCCCC;
            }
        """)

        layout = QVBoxLayout()

        # Performance mode
        self.performance_mode = QCheckBox("Performance Mode")
        self.performance_mode.setChecked(False)
        layout.addWidget(self.performance_mode)

        # Resolution scale control
        scale_layout = QHBoxLayout()
        scale_label = QLabel("Resolution Scale (1/x):")
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(2, 8)
        self.scale_slider.setValue(4)
        scale_layout.addWidget(scale_label)
        scale_layout.addWidget(self.scale_slider)
        layout.addLayout(scale_layout)

        # Mouse interaction
        self.mouse_interaction = QCheckBox("Mouse Interaction")
        self.mouse_interaction.setChecked(True)
        layout.addWidget(self.mouse_interaction)

        # Simulation controls
        sim_box = QWidget()
        sim_layout = QVBoxLayout()
        sim_layout.addWidget(QLabel("Simulation Controls:"))

        # Number of agents control
        agents_layout = QHBoxLayout()
        agents_label = QLabel("Agents:")
        self.agents_slider = QSlider(Qt.Horizontal)
        self.agents_slider.setRange(1000, 20000)
        self.agents_slider.setValue(1500)  # 20% of original
        agents_layout.addWidget(agents_label)
        agents_layout.addWidget(self.agents_slider)
        sim_layout.addLayout(agents_layout)

        # Speed control
        speed_layout = QHBoxLayout()
        speed_label = QLabel("Speed:")
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(1, 50)
        self.speed_slider.setValue(20)
        speed_layout.addWidget(speed_label)
        speed_layout.addWidget(self.speed_slider)
        sim_layout.addLayout(speed_layout)

        sim_box.setLayout(sim_layout)
        layout.addWidget(sim_box)

        # Trail controls
        trail_box = QWidget()
        trail_layout = QVBoxLayout()
        trail_layout.addWidget(QLabel("Trail Controls:"))

        decay_layout = QHBoxLayout()
        decay_label = QLabel("Trail Decay:")
        self.decay_slider = QSlider(Qt.Horizontal)
        self.decay_slider.setRange(90, 100)
        self.decay_slider.setValue(98)  # Slightly higher default decay
        decay_layout.addWidget(decay_label)
        decay_layout.addWidget(self.decay_slider)
        trail_layout.addLayout(decay_layout)

        intensity_layout = QHBoxLayout()
        intensity_label = QLabel("Trail Intensity:")
        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(1, 100)
        self.intensity_slider.setValue(50)
        intensity_layout.addWidget(intensity_label)
        intensity_layout.addWidget(self.intensity_slider)
        trail_layout.addLayout(intensity_layout)

        trail_box.setLayout(trail_layout)
        layout.addWidget(trail_box)

        # Food source controls
        food_box = QWidget()
        food_layout = QVBoxLayout()
        food_layout.addWidget(QLabel("Food Source:"))

        food_radius_layout = QHBoxLayout()
        food_radius_label = QLabel("Radius:")
        self.food_radius_slider = QSlider(Qt.Horizontal)
        self.food_radius_slider.setRange(5, 100)  # Increased range
        self.food_radius_slider.setValue(60)  # Default radius set to 20
        food_radius_layout.addWidget(food_radius_label)
        food_radius_layout.addWidget(self.food_radius_slider)
        food_layout.addLayout(food_radius_layout)

        food_strength_layout = QHBoxLayout()
        food_strength_label = QLabel("Strength:")
        self.food_strength_slider = QSlider(Qt.Horizontal)
        self.food_strength_slider.setRange(1, 2000)  # Increased maximum value
        self.food_strength_slider.setValue(1500)  # Higher default strength
        food_strength_layout.addWidget(food_strength_label)
        food_strength_layout.addWidget(self.food_strength_slider)
        food_layout.addLayout(food_strength_layout)

        food_box.setLayout(food_layout)
        layout.addWidget(food_box)

        # Color controls
        color_box = QWidget()
        color_layout = QVBoxLayout()
        color_layout.addWidget(QLabel("Color (R/G/B):"))

        color_sliders = QHBoxLayout()
        self.color_r = QSlider(Qt.Horizontal)
        self.color_g = QSlider(Qt.Horizontal)
        self.color_b = QSlider(Qt.Horizontal)
        for slider in [self.color_r, self.color_g, self.color_b]:
            slider.setRange(0, 255)
            color_sliders.addWidget(slider)
        self.color_r.setValue(200)
        self.color_g.setValue(255)
        self.color_b.setValue(200)
        color_layout.addLayout(color_sliders)

        color_box.setLayout(color_layout)
        layout.addWidget(color_box)

        # Performance info
        self.fps_label = QLabel("FPS: --")
        layout.addWidget(self.fps_label)

        # Activity Monitor
        activity_box = QWidget()
        activity_layout = QVBoxLayout()
        activity_layout.addWidget(QLabel("Activity Monitor:"))

        # Activity monitor toggle
        self.show_activity = QCheckBox("Show Activity Bar")
        self.show_activity.setChecked(True)
        activity_layout.addWidget(self.show_activity)

        # Mouse activity bar
        mouse_activity_layout = QHBoxLayout()
        mouse_activity_layout.addWidget(QLabel("Mouse:"))
        self.mouse_activity_bar = QProgressBar()
        self.mouse_activity_bar.setRange(0, 100)
        self.mouse_activity_bar.setValue(0)
        self.mouse_activity_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
                background-color: #333333;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        mouse_activity_layout.addWidget(self.mouse_activity_bar)
        activity_layout.addLayout(mouse_activity_layout)

        # Current speed indicator
        speed_indicator_layout = QHBoxLayout()
        speed_indicator_layout.addWidget(QLabel("Current Speed:"))
        self.speed_label = QLabel("1.0x")
        self.speed_label.setStyleSheet("color: #FFD700;")  # Golden color
        speed_indicator_layout.addWidget(self.speed_label)
        activity_layout.addLayout(speed_indicator_layout)

        activity_box.setLayout(activity_layout)
        layout.addWidget(activity_box)

        # Reset button
        self.reset_button = QPushButton("Reset Simulation")
        layout.addWidget(self.reset_button)

        self.setLayout(layout)
        self.setFixedSize(300, 800)


class SlimeSimulation(QMainWindow):
    def __init__(self):
        super().__init__()  # Call parent class's __init__

        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool |
            Qt.WindowTransparentForInput  # Make window pass-through
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Get screen dimensions
        screen = QApplication.primaryScreen().geometry()
        self.display_width = screen.width()
        self.display_height = screen.height()
        self.setGeometry(screen)

        # Add mouse speed tracking variables
        self.last_mouse_pos = None
        self.last_mouse_time = QElapsedTimer()
        self.last_mouse_time.start()
        self.mouse_speed = 0
        self.mouse_speed_history = []

        # Mouse speed smoothing parameters
        self.speed_history_size = 5  # Number of samples to average
        self.max_mouse_speed = 2000  # pixels per second
        self.min_speed_multiplier = 1.0
        self.max_speed_multiplier = 3.0  # Maximum speed multiplier

        # Initialize parameters
        self.setup_parameters()

        # Initialize buffers and simulation parameters
        width = self.display_width // (self.scale_factor * 2)
        height = self.display_height // (self.scale_factor * 2)
        self.reduced_buffer = np.zeros((height, width, 4), dtype=np.uint8)
        self.trail_map = np.zeros((self.sim_height, self.sim_width), dtype=np.float32)
        self.pixel_buffer = np.zeros((self.sim_height, self.sim_width, 4), dtype=np.uint8)

        # Setup simulation
        self.setup_simulation()

        # FPS and timer setup
        self.frame_times = []
        self.last_frame_time = QElapsedTimer()
        self.last_frame_time.start()

        # Initialize webcam tracker
        print("Initializing webcam tracker...")
        self.light_tracker = WebcamLightTracker(debug_mode=True)
        print("Webcam tracker initialized.")

        # Initialize food sources list
        self.food_sources = []

        # Timing for food source updates
        self.last_food_update_time = QElapsedTimer()
        self.last_food_update_time.start()
        self.food_update_interval = 1000  # milliseconds

        # Create control panel
        self.control_panel = ControlPanel()
        self.setup_control_connections()
        self.control_panel.show()

        # Set up the simulation update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(16)  # ~60 FPS target

    def setup_parameters(self):
        """Initialize simulation parameters"""
        self.scale_factor = 4
        self.sim_width = self.display_width // self.scale_factor
        self.sim_height = self.display_height // self.scale_factor
        self.num_agents = 1500
        self.move_speed = 1.5
        self.decay_rate = 0.98  # Slightly higher decay rate
        self.trail_intensity = 0.7
        self.color = QColor(180, 255, 180)
        self.performance_mode = False

        # Influence parameters
        self.mouse_influence_radius = 70
        self.mouse_influence_strength = 1.0  # Adjusted to 1.0 for normalization

    def setup_simulation(self):
        """Initialize agent system and buffers"""
        self.agent_system = AgentSystem(self.num_agents, self.sim_width, self.sim_height)
        self.trail_map.fill(0)
        self.pixel_buffer.fill(0)

    def setup_control_connections(self):
        """Connect control panel signals to slots"""
        cp = self.control_panel
        cp.scale_slider.valueChanged.connect(self.set_scale)
        cp.agents_slider.valueChanged.connect(self.set_num_agents)
        cp.speed_slider.valueChanged.connect(self.set_speed)
        cp.decay_slider.valueChanged.connect(self.set_decay)
        cp.intensity_slider.valueChanged.connect(self.set_intensity)
        cp.reset_button.clicked.connect(self.setup_simulation)
        cp.performance_mode.stateChanged.connect(self.set_performance_mode)
        cp.color_r.valueChanged.connect(self.update_color)
        cp.color_g.valueChanged.connect(self.update_color)
        cp.color_b.valueChanged.connect(self.update_color)
        cp.food_radius_slider.valueChanged.connect(self.update_food_source)
        cp.food_strength_slider.valueChanged.connect(self.update_food_source)

    def get_mouse_speed_multiplier(self):
        """Calculate speed multiplier based on mouse movement speed"""
        if len(self.mouse_speed_history) == 0:
            return self.min_speed_multiplier

        # Average recent speed measurements
        avg_speed = sum(self.mouse_speed_history) / len(self.mouse_speed_history)

        # Linear interpolation between min and max multiplier based on speed
        speed_factor = min(avg_speed / self.max_mouse_speed, 1.0)
        multiplier = self.min_speed_multiplier + (self.max_speed_multiplier - self.min_speed_multiplier) * speed_factor

        # Update the speed indicator in control panel
        self.control_panel.speed_label.setText(f"{multiplier:.1f}x")
        self.control_panel.mouse_activity_bar.setValue(int(speed_factor * 100))

        return multiplier

    def update_mouse_speed(self):
        """Update mouse speed tracking"""
        current_pos = self.mapFromGlobal(QCursor.pos())
        current_time = self.last_mouse_time.elapsed()

        if self.last_mouse_pos is not None:
            # Calculate speed in pixels per second
            delta_pos = np.array([current_pos.x() - self.last_mouse_pos.x(),
                                  current_pos.y() - self.last_mouse_pos.y()])
            delta_time = current_time / 1000.0  # Convert to seconds

            # Calculate instantaneous speed
            if delta_time > 0:
                speed = np.linalg.norm(delta_pos) / delta_time

                # Add to speed history
                self.mouse_speed_history.append(speed)
                if len(self.mouse_speed_history) > self.speed_history_size:
                    self.mouse_speed_history.pop(0)

                self.mouse_speed = speed

        self.last_mouse_pos = current_pos
        self.last_mouse_time.restart()

    def update_food_source(self):
        """Update food source parameters from control panel sliders"""
        if hasattr(self, 'food_sources'):
            for food_source in self.food_sources:
                food_source.radius = self.control_panel.food_radius_slider.value()
                food_source.strength = self.control_panel.food_strength_slider.value()  # Use value directly

    def update_simulation(self):
        try:
            # Update mouse speed tracking
            self.update_mouse_speed()
            speed_multiplier = self.get_mouse_speed_multiplier()

            # Prepare influence points
            influence_points = []

            # Update food sources only once per second
            current_time = self.last_food_update_time.elapsed()
            if current_time >= self.food_update_interval:
                # Get and process webcam food sources
                try:
                    _, light_positions = self.light_tracker.process_frame()
                    if light_positions and isinstance(light_positions, list):
                        scaled_positions = []
                        for pos in light_positions:
                            # Scale light position to simulation coordinates
                            scaled_x = int((pos[0] * self.sim_width) / self.light_tracker.frame_width)
                            scaled_y = int((pos[1] * self.sim_height) / self.light_tracker.frame_height)
                            scaled_positions.append((scaled_x, scaled_y))

                        # Update or create food sources
                        self.food_sources = []
                        for scaled_pos in scaled_positions:
                            food_source = FoodSource(
                                position=scaled_pos,
                                radius=self.control_panel.food_radius_slider.value(),
                                strength=self.control_panel.food_strength_slider.value()
                            )
                            self.food_sources.append(food_source)
                        print(f"Food source positions: {scaled_positions}")
                except Exception as e:
                    print(f"Webcam error: {e}")
                self.last_food_update_time.restart()

            # Include food sources as attraction points
            if hasattr(self, 'food_sources'):
                for food_source in self.food_sources:
                    food_strength = food_source.strength / 1000.0  # Adjusted scaling
                    influence_points.append((food_source.position, food_strength, food_source.radius))

            # Handle mouse repulsion
            if self.control_panel.mouse_interaction.isChecked():
                mouse_pos = self.mapFromGlobal(QCursor.pos())
                mouse_x = mouse_pos.x() // self.scale_factor
                mouse_y = mouse_pos.y() // self.scale_factor
                influence_points.append(((mouse_x, mouse_y), -self.mouse_influence_strength, self.mouse_influence_radius))

            # Update agent positions
            self.agent_system.update(
                move_speed=self.move_speed * speed_multiplier,
                trail_map=self.trail_map,
                influence_points=influence_points,
            )

            # Update trail map with Gaussian blur
            self.trail_map = gaussian_filter(self.trail_map, sigma=0.3, mode='reflect') * self.decay_rate

            # Update agent trails
            agent_positions = np.round(self.agent_system.positions).astype(int)
            agent_positions = np.clip(agent_positions, 0, [self.sim_width - 1, self.sim_height - 1])
            np.add.at(self.trail_map, (agent_positions[:, 1], agent_positions[:, 0]), self.trail_intensity)
            np.clip(self.trail_map, 0, 1, out=self.trail_map)

        except Exception as e:
            print(f"Error in simulation update: {e}")
            import traceback
            traceback.print_exc()

    def create_display_buffer(self):
        """Create display buffer for rendering"""
        if self.control_panel.performance_mode.isChecked():
            width = self.sim_width // 2
            height = self.sim_height // 2
            self.reduced_buffer[..., 0] = self.color.blue()
            self.reduced_buffer[..., 1] = self.color.green()
            self.reduced_buffer[..., 2] = self.color.red()
            self.reduced_buffer[..., 3] = (self.trail_map[::2, ::2] * 255).astype(np.uint8)
            self.buffer = QImage(self.reduced_buffer.tobytes(), width, height, QImage.Format_ARGB32)
        else:
            self.pixel_buffer[..., 0] = self.color.blue()
            self.pixel_buffer[..., 1] = self.color.green()
            self.pixel_buffer[..., 2] = self.color.red()
            self.pixel_buffer[..., 3] = (self.trail_map * 255).astype(np.uint8)
            self.buffer = QImage(self.pixel_buffer.tobytes(), self.sim_width, self.sim_height, QImage.Format_ARGB32)

    def paintEvent(self, event):
        """Paint event handler with food source visualization"""
        try:
            self.update_simulation()
            self.create_display_buffer()

            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform, not self.control_panel.performance_mode.isChecked())

            # Draw the main simulation
            source_rect = QRectF(0, 0, self.buffer.width(), self.buffer.height())
            target_rect = QRectF(0, 0, self.display_width, self.display_height)
            painter.drawImage(target_rect, self.buffer, source_rect)

            # Draw food sources if they exist
            if hasattr(self, 'food_sources'):
                for food_source in self.food_sources:
                    food_source.draw(painter, self.scale_factor)

            # Update FPS counter
            elapsed = self.last_frame_time.elapsed()
            self.last_frame_time.restart()

            self.frame_times.append(elapsed)
            if len(self.frame_times) > 60:
                self.frame_times.pop(0)

            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            self.control_panel.fps_label.setText(f"FPS: {fps:.1f}")
        except Exception as e:
            print(f"Error in paint event: {e}")

    def set_scale(self, value):
        """Update simulation scale factor"""
        self.scale_factor = value
        self.sim_width = self.display_width // self.scale_factor
        self.sim_height = self.display_height // self.scale_factor
        self.setup_simulation()

    def set_num_agents(self, value):
        """Update number of simulation agents"""
        self.num_agents = value
        self.setup_simulation()

    def set_speed(self, value):
        """Update base movement speed"""
        self.move_speed = value / 10

    def set_decay(self, value):
        """Update trail decay rate"""
        self.decay_rate = value / 100

    def set_intensity(self, value):
        """Update trail intensity"""
        self.trail_intensity = value / 100

    def set_performance_mode(self, state):
        """Toggle performance mode"""
        self.performance_mode = bool(state)

    def update_color(self):
        """Update particle color from RGB sliders"""
        self.color = QColor(
            self.control_panel.color_r.value(),
            self.control_panel.color_g.value(),
            self.control_panel.color_b.value()
        )


class WebcamLightTracker:
    def __init__(self, debug_mode=True, mirror_horizontally=True, mirror_vertically=False):
        # Initialize webcam capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Could not open webcam")

        # Debug mode flag
        self.debug_mode = debug_mode

        # Mirroring flags
        self.mirror_horizontally = mirror_horizontally
        self.mirror_vertically = mirror_vertically

        # Minimum contour area to filter out small contours
        self.min_contour_area = 100

        # Frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def process_frame(self):
        # Capture a frame from the webcam
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to grab frame")

        # Apply mirroring if enabled
        if self.mirror_horizontally:
            frame = cv2.flip(frame, 1)  # Flip horizontally (left-right)
        if self.mirror_vertically:
            frame = cv2.flip(frame, 0)  # Flip vertically (up-down)

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply a threshold to detect bright spots (light source)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours of the light spots
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # List to store light positions
        light_positions = []

        # If there are contours, find centers
        if contours:
            for contour in contours:
                # Filter small contours
                if cv2.contourArea(contour) < self.min_contour_area:
                    continue

                M = cv2.moments(contour)
                # Compute the center of the contour
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    light_positions.append((cx, cy))

                    # Draw a circle around the detected light position
                    if self.debug_mode:
                        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)  # Green contour
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot at light position

        # If in debug mode, show the frame with detected light positions
        if self.debug_mode:
            cv2.imshow("Light Detection Debug", frame)

        return frame, light_positions

    def release(self):
        # Release the webcam when done
        self.cap.release()
        cv2.destroyAllWindows()

    def run(self):
        """Run the webcam and display the debug window in a loop."""
        while True:
            # Process the frame from the webcam
            _, light_pos = self.process_frame()

            # Wait for a key press and break the loop if ESC is pressed
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                print("Exiting webcam light tracker...")
                break

        # Release resources after exiting the loop
        self.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    simulation = SlimeSimulation()
    simulation.show()
    sys.exit(app.exec_())
