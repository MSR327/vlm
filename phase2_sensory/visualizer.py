"""
Real-time OpenCV HUD Visualizer for Phase 2 TransFuser-PPO Autonomous Driver.

Renders an interactive 1280x720 canvas combining:
1. Third-person chase camera view of the ego-vehicle.
2. Picture-in-picture sensor insets: Front RGB + Colormapped 2D LiDAR BEV.
3. High-contrast telemetry dashboard with dynamic gauges and safety alerts.
"""
import os
import time
import numpy as np
import cv2


class SensoryHUDVisualizer:
    """
    Real-time HUD Visualizer displaying 3rd-person chase camera, sensor PIPs,
    and telemetry gauges on an OpenCV canvas (1280x720).
    """
    def __init__(self, window_name="Phase 2 TransFuser-PPO Autonomous Driver", save_dir=None):
        self.window_name = window_name
        self.save_dir = save_dir if save_dir else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'results', 'screenshots'
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.canvas_w = 1280
        self.canvas_h = 720
        self.chase_w = 880
        self.chase_h = 720
        self.panel_w = self.canvas_w - self.chase_w  # 400px

    def _draw_bar(self, img, x, y, w, h, fill_pct, fill_color, bg_color=(40, 40, 40), border_color=(120, 120, 120)):
        """Draws a horizontal progress bar."""
        cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
        fill_w = int(np.clip(fill_pct, 0.0, 1.0) * w)
        if fill_w > 0:
            cv2.rectangle(img, (x, y), (x + fill_w, y + h), fill_color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 1)

    def _draw_centered_bar(self, img, x, y, w, h, val, fill_color=(0, 220, 255)):
        """Draws a centered bidirectional balance bar (e.g. steering in [-1, 1])."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
        mid_x = x + w // 2
        cv2.line(img, (mid_x, y), (mid_x, y + h), (180, 180, 180), 1)
        clamped_val = float(np.clip(val, -1.0, 1.0))
        offset = int(clamped_val * (w // 2))
        if offset > 0:
            cv2.rectangle(img, (mid_x, y), (mid_x + offset, y + h), fill_color, -1)
        elif offset < 0:
            cv2.rectangle(img, (mid_x + offset, y), (mid_x, y + h), fill_color, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)

    def _format_lidar_bev_color(self, bev_tensor):
        """
        Converts 2-channel LiDAR BEV tensor (2, 256, 256) into a composite RGB image.
        Channel 0: Obstacles above ground -> Warm Orange/Red
        Channel 1: Ground plane -> Cool Blue/Cyan
        """
        if hasattr(bev_tensor, 'cpu'):
            bev_np = bev_tensor.detach().cpu().numpy()
        else:
            bev_np = np.asarray(bev_tensor, dtype=np.float32)

        if bev_np.ndim == 3 and bev_np.shape[0] == 2:
            ch_obs = np.clip(bev_np[0], 0.0, 1.0)
            ch_gnd = np.clip(bev_np[1], 0.0, 1.0)
        else:
            ch_obs = np.zeros((256, 256), dtype=np.float32)
            ch_gnd = np.zeros((256, 256), dtype=np.float32)

        # Composite BGR: Blue=Ground, Green=Ground*0.7+Obs*0.5, Red=Obstacle
        b = (ch_gnd * 220.0).astype(np.uint8)
        g = (ch_gnd * 120.0 + ch_obs * 100.0).astype(np.uint8)
        r = (ch_obs * 255.0).astype(np.uint8)
        bgr = cv2.merge([b, g, r])

        # Draw ego vehicle footprint in center-bottom
        ego_x = 128
        ego_y = 224  # corresponding to x ~ 0m in [-4, 28] window
        cv2.rectangle(bgr, (ego_x - 6, ego_y - 12), (ego_x + 6, ego_y + 12), (0, 255, 0), -1)
        return bgr

    def build_canvas(self, chase_img, front_rgb, bev_tensor, telemetry):
        """
        Constructs the full 1280x720 canvas from raw camera, BEV, and telemetry dictionary.
        """
        canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        # 1. Main Chase View (Left Pane: 880 x 720)
        if chase_img is not None:
            if chase_img.shape[:2] != (self.chase_h, self.chase_w):
                main_view = cv2.resize(chase_img, (self.chase_w, self.chase_h))
            else:
                main_view = chase_img
        else:
            # Fallback placeholder if no chase camera
            main_view = np.zeros((self.chase_h, self.chase_w, 3), dtype=np.uint8)
            cv2.putText(main_view, "NO CHASE CAMERA DETECTED", (220, 360),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 100, 100), 2)

        canvas[:, :self.chase_w] = main_view

        # 2. Sensor Inset Panel (Top Right: 400 x 200)
        # 2a. Front Camera (200 x 200)
        if front_rgb is not None:
            if front_rgb.dtype == np.float32 or front_rgb.dtype == np.float64:
                front_bgr = (np.clip(front_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[:, :, ::-1]
            else:
                front_bgr = front_rgb
            front_thumb = cv2.resize(front_bgr, (200, 200))
        else:
            front_thumb = np.zeros((200, 200, 3), dtype=np.uint8)

        cv2.rectangle(front_thumb, (0, 0), (200, 22), (20, 20, 20), -1)
        cv2.putText(front_thumb, "Front RGB (256x256)", (8, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        canvas[:200, self.chase_w:self.chase_w + 200] = front_thumb

        # 2b. LiDAR BEV Grid (200 x 200)
        if bev_tensor is not None:
            bev_bgr = self._format_lidar_bev_color(bev_tensor)
            bev_thumb = cv2.resize(bev_bgr, (200, 200))
        else:
            bev_thumb = np.zeros((200, 200, 3), dtype=np.uint8)

        cv2.rectangle(bev_thumb, (0, 0), (200, 22), (20, 20, 20), -1)
        cv2.putText(bev_thumb, "LiDAR BEV (32x32m)", (10, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        canvas[:200, self.chase_w + 200:self.canvas_w] = bev_thumb

        # Divider lines
        cv2.line(canvas, (self.chase_w, 0), (self.chase_w, self.canvas_h), (80, 80, 80), 2)
        cv2.line(canvas, (self.chase_w, 200), (self.canvas_w, 200), (80, 80, 80), 2)
        cv2.line(canvas, (self.chase_w + 200, 0), (self.chase_w + 200, 200), (80, 80, 80), 1)

        # 3. Telemetry & Control Panel (Bottom Right: 400 x 520)
        dash = canvas[200:self.canvas_h, self.chase_w:self.canvas_w]
        dash[:] = (18, 18, 22)  # Dark background

        # Parse telemetry fields
        ep = telemetry.get('episode', 1)
        step = telemetry.get('step', 0)
        rew = telemetry.get('reward', 0.0)
        spd = telemetry.get('speed', 0.0)
        target_spd = telemetry.get('target_speed', 20.0)
        steer = telemetry.get('steer', 0.0)
        throttle = telemetry.get('throttle', 0.0)
        brake = telemetry.get('brake', 0.0)
        dev = telemetry.get('lane_deviation', 0.0)
        heading_err = telemetry.get('heading_error', 0.0)
        comp = telemetry.get('route_completion', 0.0)
        dist = telemetry.get('distance_covered', 0.0)
        stall = telemetry.get('stall_steps', 0)
        max_stall = telemetry.get('max_stall_steps', 100)
        ppo_val = telemetry.get('value', 0.0)
        lat_ms = telemetry.get('latency_ms', 0.0)

        # Header Title
        cv2.putText(dash, "TRANSFUSER-PPO (PHASE 2)", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 230, 255), 2)
        cv2.putText(dash, f"Ep: {ep} | Step: {step:04d} | R: {rew:+6.1f}", (15, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Status Alert Badge
        status_box_y = 66
        if stall > 40:
            status_text = f"STALL WARNING ({stall}/{max_stall})"
            badge_color = (0, 140, 255)
        elif dev > 1.5:
            status_text = f"CAUTION: LANE DRIFT ({dev:.2f}m)"
            badge_color = (0, 215, 255)
        elif brake > 0.5:
            status_text = "EMERGENCY BRAKING ACTIVE"
            badge_color = (0, 0, 255)
        else:
            status_text = "NORMAL AUTONOMOUS TRACKING"
            badge_color = (40, 200, 40)

        cv2.rectangle(dash, (15, status_box_y), (385, status_box_y + 26), badge_color, -1)
        cv2.putText(dash, status_text, (25, status_box_y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2)

        # Telemetry Gauges
        y_cursor = 118

        # 1. Longitudinal Speed (Gauge & Bar)
        spd_color = (0, 255, 120) if spd >= 10.0 else (0, 160, 255)
        cv2.putText(dash, f"Long. Speed: {spd:5.1f} km/h (Target: {target_spd:.0f})", (15, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
        self._draw_bar(dash, 15, y_cursor + 6, 370, 12, spd / 35.0, spd_color)
        y_cursor += 36

        # 2. Continuous Steering (Centered Balance Bar)
        cv2.putText(dash, f"Steering Command: {steer:+5.2f}", (15, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
        self._draw_centered_bar(dash, 15, y_cursor + 6, 370, 12, steer)
        y_cursor += 36

        # 3. Throttle and Brake Dual Meters
        cv2.putText(dash, f"Throttle: {throttle:4.2f}        Brake: {brake:4.2f}", (15, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
        self._draw_bar(dash, 15, y_cursor + 6, 175, 10, throttle, (0, 220, 0))
        self._draw_bar(dash, 210, y_cursor + 6, 175, 10, brake, (0, 0, 255))
        y_cursor += 36

        # 4. Route Navigation & Progress
        comp_color = (0, 230, 255)
        cv2.putText(dash, f"Route Completion: {comp:5.1f}% ({dist:5.1f}m / 500m)", (15, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (220, 220, 220), 1)
        self._draw_bar(dash, 15, y_cursor + 6, 370, 10, comp / 100.0, comp_color)
        y_cursor += 36

        # 5. Privileged Ground Truth Diagnostics
        dev_color = (0, 255, 0) if dev < 0.6 else ((0, 215, 255) if dev < 1.5 else (0, 0, 255))
        cv2.putText(dash, f"Lane Center Dev: {dev:4.2f} m", (15, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, dev_color, 1)
        cv2.putText(dash, f"Heading Err: {heading_err:4.1f} deg", (210, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
        y_cursor += 28

        # 6. RL Model Telemetry
        cv2.putText(dash, f"Critic Value V(s): {ppo_val:+5.2f}", (15, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
        fps = 1000.0 / lat_ms if lat_ms > 0 else 0.0
        cv2.putText(dash, f"Latency: {lat_ms:4.1f}ms ({fps:.0f} FPS)", (210, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (140, 220, 140), 1)
        y_cursor += 28

        # 7. Anti-Stall Counter
        stall_color = (180, 180, 180) if stall < 20 else (0, 120, 255)
        cv2.putText(dash, f"Anti-Stall Timer: {stall:02d} / {max_stall:02d} ticks", (15, y_cursor),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, stall_color, 1)
        self._draw_bar(dash, 15, y_cursor + 6, 370, 8, stall / float(max_stall), (0, 80, 255))
        y_cursor += 34

        # Footer Keys
        cv2.line(dash, (15, y_cursor), (385, y_cursor), (60, 60, 60), 1)
        cv2.putText(dash, "[S] Save Snapshot  |  [Q] Terminate Episode", (30, y_cursor + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1)

        return canvas

    def render(self, chase_img, front_rgb, bev_tensor, telemetry, wait_ms=1):
        """
        Builds canvas, displays in window, and handles key interactions.
        Returns:
            key_action: None, 'screenshot', or 'quit'
        """
        canvas = self.build_canvas(chase_img, front_rgb, bev_tensor, telemetry)
        try:
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('s') or key == ord('S'):
                self.save_screenshot(canvas, telemetry)
                return 'screenshot'
            elif key == ord('q') or key == ord('Q') or key == 27:  # ESC
                return 'quit'
        except cv2.error:
            # Headless environment where OpenCV GUI is not compiled/available
            pass
        return None

    def save_screenshot(self, canvas, telemetry=None, custom_tag=""):
        """Saves current HUD canvas to disk."""
        ep = telemetry.get('episode', 0) if telemetry else 0
        step = telemetry.get('step', 0) if telemetry else 0
        tag = f"_{custom_tag}" if custom_tag else ""
        filename = f"snapshot_ep{ep:03d}_step{step:04d}{tag}.png"
        path = os.path.join(self.save_dir, filename)
        cv2.imwrite(path, canvas)
        print(f"[HUD] Saved screenshot to {path}")
        return path

    def close(self):
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
