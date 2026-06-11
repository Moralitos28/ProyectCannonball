"""Simulador de movimiento parabolico para una bala de canon.

La aplicacion usa Tkinter para la interfaz, Matplotlib para la grafica y
FuncAnimation para mostrar el movimiento de un proyectil ideal sin resistencia
del aire.
"""

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class ProjectileSimulator:
    """Interfaz principal y logica de simulacion del proyectil."""

    GRAVITY = 9.81

    def __init__(self, root):
        self.root = root
        self.root.title("Simulador de movimiento parabolico")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)

        self.animation = None
        self.trajectory = None
        self.cannon_patch = None
        self.cannon_wheel = None
        self.ball_artist = None
        self.path_artist = None
        self.max_height_artist = None
        self.impact_artist = None

        self.result_vars = {}
        self.realtime_vars = {}

        self._configure_style()
        self._build_interface()
        self.reset_simulation()

    def _configure_style(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 20, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Helvetica", 12, "bold"))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10, "bold"))
        style.configure("Accent.TButton", font=("Helvetica", 11, "bold"))

    def _build_interface(self):
        main_container = ttk.Frame(self.root, padding=12)
        main_container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main_container,
            text="Simulador de movimiento parabólico",
            style="Title.TLabel",
        )
        title.pack(anchor=tk.CENTER, pady=(0, 12))

        content = ttk.Frame(main_container)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(content)
        sidebar.grid(row=0, column=0, sticky="ns", padx=(0, 12))

        graph_panel = ttk.Frame(content)
        graph_panel.grid(row=0, column=1, sticky="nsew")
        graph_panel.rowconfigure(0, weight=1)
        graph_panel.columnconfigure(0, weight=1)

        self._build_input_panel(sidebar)
        self._build_results_panel(sidebar)
        self._build_realtime_panel(sidebar)
        self._build_plot(graph_panel)

    def _build_input_panel(self, parent):
        input_frame = ttk.LabelFrame(
            parent,
            text="Panel de entrada de datos",
            padding=12,
            style="Section.TLabelframe",
        )
        input_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(input_frame, text="Velocidad inicial (m/s):").grid(
            row=0, column=0, sticky=tk.W, pady=4
        )
        self.velocity_entry = ttk.Entry(input_frame, width=18)
        self.velocity_entry.grid(row=0, column=1, sticky=tk.EW, pady=4)
        self.velocity_entry.insert(0, "13")

        ttk.Label(input_frame, text="Ángulo de lanzamiento (°):").grid(
            row=1, column=0, sticky=tk.W, pady=4
        )
        self.angle_entry = ttk.Entry(input_frame, width=18)
        self.angle_entry.grid(row=1, column=1, sticky=tk.EW, pady=4)
        self.angle_entry.insert(0, "30")

        ttk.Label(input_frame, text="Velocidad de reproducción:").grid(
            row=2, column=0, sticky=tk.W, pady=4
        )
        self.playback_var = tk.DoubleVar(value=1.0)
        self.playback_scale = ttk.Scale(
            input_frame,
            from_=0.25,
            to=3.0,
            variable=self.playback_var,
            orient=tk.HORIZONTAL,
            command=self._update_playback_label,
        )
        self.playback_scale.grid(row=2, column=1, sticky=tk.EW, pady=4)
        self.playback_label = ttk.Label(input_frame, text="1.00x")
        self.playback_label.grid(row=3, column=1, sticky=tk.E, pady=(0, 8))

        launch_button = ttk.Button(
            input_frame,
            text="Lanzar proyectil",
            command=self.start_animation,
            style="Accent.TButton",
        )
        launch_button.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=(8, 4))

        reset_button = ttk.Button(
            input_frame,
            text="Reiniciar",
            command=self.reset_simulation,
        )
        reset_button.grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=4)

        input_frame.columnconfigure(1, weight=1)

    def _build_results_panel(self, parent):
        results_frame = ttk.LabelFrame(
            parent,
            text="Panel de resultados",
            padding=12,
            style="Section.TLabelframe",
        )
        results_frame.pack(fill=tk.X, pady=(0, 10))

        result_labels = [
            ("v0x", "Velocidad horizontal:"),
            ("v0y", "Velocidad vertical inicial:"),
            ("time_total", "Tiempo total de vuelo:"),
            ("height_max", "Altura máxima:"),
            ("range_x", "Alcance horizontal:"),
            ("current_vy", "Velocidad vertical actual:"),
            ("current_speed", "Rapidez total actual:"),
        ]

        for row, (key, label_text) in enumerate(result_labels):
            ttk.Label(results_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
            self.result_vars[key] = tk.StringVar(value="--")
            ttk.Label(results_frame, textvariable=self.result_vars[key]).grid(
                row=row, column=1, sticky=tk.E, pady=2
            )

        results_frame.columnconfigure(1, weight=1)

    def _build_realtime_panel(self, parent):
        realtime_frame = ttk.LabelFrame(
            parent,
            text="Visualización en tiempo real",
            padding=12,
            style="Section.TLabelframe",
        )
        realtime_frame.pack(fill=tk.X)

        realtime_labels = [
            ("time", "Tiempo transcurrido:"),
            ("x", "Posición horizontal:"),
            ("y", "Posición vertical:"),
            ("vx", "Velocidad horizontal:"),
            ("vy", "Velocidad vertical:"),
        ]

        for row, (key, label_text) in enumerate(realtime_labels):
            ttk.Label(realtime_frame, text=label_text).grid(row=row, column=0, sticky=tk.W, pady=2)
            self.realtime_vars[key] = tk.StringVar(value="--")
            ttk.Label(realtime_frame, textvariable=self.realtime_vars[key]).grid(
                row=row, column=1, sticky=tk.E, pady=2
            )

        realtime_frame.columnconfigure(1, weight=1)

    def _build_plot(self, parent):
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._setup_empty_plot()

    def _update_playback_label(self, _event=None):
        self.playback_label.config(text=f"{self.playback_var.get():.2f}x")

    def validate_inputs(self):
        """Valida y convierte las entradas del usuario."""
        try:
            velocity = float(self.velocity_entry.get())
            angle = float(self.angle_entry.get())
            playback_speed = float(self.playback_var.get())
        except ValueError:
            messagebox.showerror(
                "Entrada inválida",
                "La velocidad y el ángulo deben contener valores numéricos.",
            )
            return None

        if velocity <= 0:
            messagebox.showerror(
                "Entrada inválida",
                "La velocidad inicial debe ser mayor que cero.",
            )
            return None

        if angle <= 0 or angle >= 90:
            messagebox.showerror(
                "Entrada inválida",
                "El ángulo debe ser mayor que 0° y menor que 90°.",
            )
            return None

        return velocity, angle, max(playback_speed, 0.25)

    def calculate_trajectory(self, initial_velocity, angle_degrees):
        """Calcula trayectoria, velocidades y magnitudes principales."""
        angle_radians = np.radians(angle_degrees)
        v0x = initial_velocity * np.cos(angle_radians)
        v0y = initial_velocity * np.sin(angle_radians)

        time_total = 2 * v0y / self.GRAVITY
        height_max = v0y**2 / (2 * self.GRAVITY)
        range_x = v0x * time_total

        frames = max(90, int(time_total * 90))
        time_values = np.linspace(0, time_total, frames)
        x_values = v0x * time_values
        y_values = v0y * time_values - 0.5 * self.GRAVITY * time_values**2
        y_values[-1] = 0.0

        vy_values = v0y - self.GRAVITY * time_values
        vx_values = np.full_like(time_values, v0x)
        speed_values = np.sqrt(vx_values**2 + vy_values**2)

        return {
            "v0x": v0x,
            "v0y": v0y,
            "time_total": time_total,
            "height_max": height_max,
            "range_x": range_x,
            "time": time_values,
            "x": x_values,
            "y": y_values,
            "vx": vx_values,
            "vy": vy_values,
            "speed": speed_values,
        }

    def start_animation(self):
        """Inicia una nueva simulacion y animacion."""
        validated_values = self.validate_inputs()
        if validated_values is None:
            return

        initial_velocity, angle, playback_speed = validated_values
        self._stop_animation()

        self.trajectory = self.calculate_trajectory(initial_velocity, angle)
        self._update_general_results(frame=0)
        self._setup_simulation_plot(angle)

        interval_ms = max(8, int(25 / playback_speed))
        frame_count = len(self.trajectory["time"])
        self.animation = animation.FuncAnimation(
            self.figure,
            self.update_frame,
            frames=frame_count,
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        self.canvas.draw_idle()

    def update_frame(self, frame):
        """Actualiza cada cuadro de la animacion."""
        if self.trajectory is None:
            return []

        x_values = self.trajectory["x"]
        y_values = self.trajectory["y"]

        current_x = x_values[frame]
        current_y = y_values[frame]

        self.ball_artist.set_data([current_x], [current_y])
        self.path_artist.set_data(x_values[: frame + 1], y_values[: frame + 1])
        self._update_general_results(frame)
        self._update_realtime_values(frame)

        if frame >= len(x_values) - 1:
            self._stop_animation()

        return [self.ball_artist, self.path_artist]

    def reset_simulation(self):
        """Limpia la grafica y los paneles para una nueva simulacion."""
        self._stop_animation()
        self.trajectory = None
        self._setup_empty_plot()

        for variable in self.result_vars.values():
            variable.set("--")
        for variable in self.realtime_vars.values():
            variable.set("--")

        self.canvas.draw_idle()

    def _stop_animation(self):
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()
        self.animation = None

    def _setup_empty_plot(self):
        self.ax.clear()
        self.ax.set_title("Trayectoria de la bala de cañón")
        self.ax.set_xlabel("Posición horizontal x (m)")
        self.ax.set_ylabel("Posición vertical y (m)")
        self.ax.grid(True, linestyle="--", alpha=0.35)
        self.ax.axhline(0, color="saddlebrown", linewidth=2, label="Suelo")
        self.ax.set_xlim(-1, 16)
        self.ax.set_ylim(-0.5, 5)
        self._draw_cannon(angle_degrees=30)
        self.ax.legend(loc="upper right")

    def _setup_simulation_plot(self, angle_degrees):
        self.ax.clear()

        range_x = self.trajectory["range_x"]
        height_max = self.trajectory["height_max"]
        x_values = self.trajectory["x"]
        y_values = self.trajectory["y"]

        x_margin = max(range_x * 0.1, 1.0)
        y_margin = max(height_max * 0.25, 1.0)

        self.ax.set_xlim(-x_margin, range_x + x_margin)
        self.ax.set_ylim(-y_margin * 0.35, height_max + y_margin)
        self.ax.set_title("Trayectoria de la bala de cañón")
        self.ax.set_xlabel("Posición horizontal x (m)")
        self.ax.set_ylabel("Posición vertical y (m)")
        self.ax.grid(True, linestyle="--", alpha=0.35)
        self.ax.axhline(0, color="saddlebrown", linewidth=2, label="Suelo")

        self._draw_cannon(angle_degrees)

        (self.path_artist,) = self.ax.plot(
            [],
            [],
            color="royalblue",
            linewidth=2,
            label="Trayectoria recorrida",
        )
        (self.ball_artist,) = self.ax.plot(
            [0],
            [0],
            marker="o",
            color="black",
            markersize=11,
            linestyle="None",
            label="Bala de cañón",
        )

        max_index = int(np.argmax(y_values))
        self.max_height_artist = self.ax.scatter(
            [x_values[max_index]],
            [y_values[max_index]],
            color="crimson",
            marker="^",
            s=70,
            label="Altura máxima",
            zorder=4,
        )
        self.impact_artist = self.ax.scatter(
            [range_x],
            [0],
            color="darkorange",
            marker="x",
            s=90,
            linewidths=2,
            label="Impacto",
            zorder=4,
        )

        self.ax.legend(loc="upper right")
        self.canvas.draw_idle()

    def _draw_cannon(self, angle_degrees):
        barrel_length = 0.9
        barrel_width = 0.18
        cannon_angle = max(8, min(angle_degrees, 75))

        barrel = patches.Rectangle(
            (0, 0),
            barrel_length,
            barrel_width,
            angle=cannon_angle,
            color="dimgray",
            zorder=5,
        )
        wheel = patches.Circle(
            (0, -0.08),
            0.18,
            color="gray",
            ec="black",
            linewidth=1,
            zorder=6,
        )
        base = patches.Rectangle(
            (-0.22, -0.16),
            0.5,
            0.08,
            color="black",
            zorder=5,
        )

        self.ax.add_patch(barrel)
        self.ax.add_patch(wheel)
        self.ax.add_patch(base)

    def _update_general_results(self, frame):
        if self.trajectory is None:
            return

        self.result_vars["v0x"].set(f"{self.trajectory['v0x']:.2f} m/s")
        self.result_vars["v0y"].set(f"{self.trajectory['v0y']:.2f} m/s")
        self.result_vars["time_total"].set(f"{self.trajectory['time_total']:.2f} s")
        self.result_vars["height_max"].set(f"{self.trajectory['height_max']:.2f} m")
        self.result_vars["range_x"].set(f"{self.trajectory['range_x']:.2f} m")
        self.result_vars["current_vy"].set(f"{self.trajectory['vy'][frame]:.2f} m/s")
        self.result_vars["current_speed"].set(f"{self.trajectory['speed'][frame]:.2f} m/s")

    def _update_realtime_values(self, frame):
        self.realtime_vars["time"].set(f"{self.trajectory['time'][frame]:.2f} s")
        self.realtime_vars["x"].set(f"{self.trajectory['x'][frame]:.2f} m")
        self.realtime_vars["y"].set(f"{self.trajectory['y'][frame]:.2f} m")
        self.realtime_vars["vx"].set(f"{self.trajectory['vx'][frame]:.2f} m/s")
        self.realtime_vars["vy"].set(f"{self.trajectory['vy'][frame]:.2f} m/s")


def main():
    root = tk.Tk()
    ProjectileSimulator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
