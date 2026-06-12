"""Simulador de movimiento parabólico para una bala de cañón.

La aplicación usa Tkinter para la interfaz, Matplotlib para la gráfica y
FuncAnimation para animar un proyectil ideal sin resistencia del aire.
"""

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib.animation as animation
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage


class ProjectileSimulator:
    """Coordina la interfaz, los cálculos físicos y la animación."""

    GRAVITY = 9.81
    WINDOW_TITLE = "Simulador de movimiento parabólico"
    WINDOW_SIZE = "1180x760"
    MIN_WINDOW_SIZE = (980, 680)
    CANNON_IMAGE_PATH = Path(__file__).with_name("toy-style-cannon-e9cf.png")
    DEFAULT_MANUAL_VELOCITY = "13"
    DEFAULT_MANUAL_ANGLE = "30"
    DEFAULT_TARGET_RANGE = "14.92"
    DEFAULT_PLAYBACK_SPEED = 1.0
    IDEAL_ANGLE_DEGREES = 45.0
    MIN_PLAYBACK_SPEED = 0.25
    MAX_PLAYBACK_SPEED = 3.0
    BASE_ANIMATION_INTERVAL_MS = 25
    MIN_ANIMATION_INTERVAL_MS = 8
    MIN_TRAJECTORY_FRAMES = 90
    FRAMES_PER_SECOND_OF_FLIGHT = 90
    EARTH_RADIUS_KM = 6371.0
    FLIGHT_MAX_ALTITUDE_KM = 10.0
    EARTH_ROUTE_FRAMES = 160
    DEFAULT_ORIGIN_LAT = "9.9281"
    DEFAULT_ORIGIN_LON = "-84.0907"
    DEFAULT_DEST_LAT = "40.7128"
    DEFAULT_DEST_LON = "-74.0060"
    INITIAL_MATH_EXPLANATION = (
        "Lanza una simulación para ver el desarrollo matemático con "
        "sustituciones numéricas."
    )
    EARTH_MATH_EXPLANATION = (
        "La pestaña Math Explained muestra el desarrollo de las simulaciones "
        "parabólicas 2D. Para la ruta GPS 3D se usa una esfera terrestre en km "
        "y una altura máxima fija de 10 km."
    )
    INITIAL_MATH_BLOCKS = [
        ("note", INITIAL_MATH_EXPLANATION),
    ]
    RESULT_LABELS = [
        ("mode", "Modo de simulación:"),
        ("target_range", "Distancia objetivo:"),
        ("initial_velocity", "Velocidad inicial usada:"),
        ("launch_angle", "Ángulo usado:"),
        ("v0x", "Velocidad horizontal:"),
        ("v0y", "Velocidad vertical inicial:"),
        ("time_total", "Tiempo total de vuelo:"),
        ("height_max", "Altura máxima:"),
        ("range_x", "Alcance horizontal:"),
        ("current_vy", "Velocidad vertical actual:"),
        ("current_speed", "Rapidez total actual:"),
        ("gps_origin", "Origen GPS:"),
        ("gps_destination", "Destino GPS:"),
        ("earth_radius", "Radio terrestre:"),
        ("surface_distance", "Distancia sobre superficie:"),
        ("flight_altitude", "Altura máxima de vuelo:"),
    ]
    REALTIME_LABELS = [
        ("time", "Tiempo transcurrido:"),
        ("x", "Posición horizontal:"),
        ("y", "Posición vertical:"),
        ("vx", "Velocidad horizontal:"),
        ("vy", "Velocidad vertical:"),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title(self.WINDOW_TITLE)
        self.root.geometry(self.WINDOW_SIZE)
        self.root.minsize(*self.MIN_WINDOW_SIZE)

        self.animation = None
        self.trajectory = None
        self.simulation_context = {}
        self.cannon_image = self._load_cannon_image()
        self.ball_artist = None
        self.path_artist = None
        self.max_height_artist = None
        self.impact_artist = None
        self.earth_figure = None
        self.earth_ax = None
        self.earth_canvas = None
        self.earth_animation = None
        self.earth_route = None
        self.plane_artist = None
        self.earth_route_artist = None
        self.output_notebook = None

        self.result_vars = {}
        self.result_labels = {}
        self.realtime_vars = {}
        self.math_text_widget = None
        self.current_math_explanation = self.INITIAL_MATH_EXPLANATION

        self._configure_style()
        self._build_interface()
        self.reset_simulation()

    def _configure_style(self):
        """Define una apariencia consistente para los widgets principales."""
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Helvetica", 20, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Helvetica", 12, "bold"))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("TButton", font=("Helvetica", 10, "bold"))
        style.configure("Accent.TButton", font=("Helvetica", 11, "bold"))

    def _build_interface(self):
        """Construye el contenedor principal con panel lateral y gráfica."""
        main_container = ttk.Frame(self.root, padding=12)
        main_container.pack(fill=tk.BOTH, expand=True)

        title = ttk.Label(
            main_container,
            text=self.WINDOW_TITLE,
            style="Title.TLabel",
        )
        title.pack(anchor=tk.CENTER, pady=(0, 12))

        content = ttk.Frame(main_container)
        content.pack(fill=tk.BOTH, expand=True)
        content.columnconfigure(0, weight=0)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        sidebar_container = ttk.Frame(content)
        sidebar_container.grid(row=0, column=0, sticky="ns", padx=(0, 12))
        sidebar_container.rowconfigure(0, weight=1)
        sidebar_container.columnconfigure(0, weight=1)

        graph_panel = ttk.Frame(content)
        graph_panel.grid(row=0, column=1, sticky="nsew")
        graph_panel.rowconfigure(0, weight=1)
        graph_panel.columnconfigure(0, weight=1)

        sidebar = self._build_scrollable_sidebar(sidebar_container)
        self._build_input_panel(sidebar)
        self._build_results_panel(sidebar)
        self._build_realtime_panel(sidebar)
        self._build_output_notebook(graph_panel)

    def _build_scrollable_sidebar(self, parent):
        """Crea un panel lateral con scroll vertical."""
        canvas = tk.Canvas(parent, width=390, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        sidebar = ttk.Frame(canvas)
        sidebar_window = canvas.create_window((0, 0), window=sidebar, anchor=tk.NW)

        canvas.grid(row=0, column=0, sticky="ns")
        scrollbar.grid(row=0, column=1, sticky="ns")
        canvas.configure(yscrollcommand=scrollbar.set)

        def update_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def match_canvas_width(event):
            canvas.itemconfigure(sidebar_window, width=event.width)

        def scroll_with_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        sidebar.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", match_canvas_width)
        canvas.bind(
            "<Enter>",
            lambda _event: canvas.bind_all("<MouseWheel>", scroll_with_mousewheel),
        )
        canvas.bind("<Leave>", lambda _event: canvas.unbind_all("<MouseWheel>"))

        return sidebar

    def _build_input_panel(self, parent):
        """Construye los campos de entrada y controles de simulación."""
        input_frame = ttk.LabelFrame(
            parent,
            text="Panel de entrada de datos",
            padding=12,
            style="Section.TLabelframe",
        )
        input_frame.pack(fill=tk.X, pady=(0, 10))

        self.simulation_notebook = ttk.Notebook(input_frame)
        self.simulation_notebook.grid(row=0, column=0, columnspan=2, sticky=tk.EW)

        manual_tab = ttk.Frame(self.simulation_notebook, padding=10)
        ideal_tab = ttk.Frame(self.simulation_notebook, padding=10)
        earth_tab = ttk.Frame(self.simulation_notebook, padding=10)
        self.simulation_notebook.add(manual_tab, text="Velocidad y ángulo")
        self.simulation_notebook.add(ideal_tab, text="Distancia objetivo")
        self.simulation_notebook.add(earth_tab, text="Ruta GPS 3D")

        ttk.Label(manual_tab, text="Velocidad inicial (m/s):").grid(
            row=0, column=0, sticky=tk.W, pady=4
        )
        self.velocity_entry = ttk.Entry(manual_tab, width=18)
        self.velocity_entry.grid(row=0, column=1, sticky=tk.EW, pady=4)
        self.velocity_entry.insert(0, self.DEFAULT_MANUAL_VELOCITY)

        ttk.Label(manual_tab, text="Ángulo de lanzamiento (°):").grid(
            row=1, column=0, sticky=tk.W, pady=4
        )
        self.angle_entry = ttk.Entry(manual_tab, width=18)
        self.angle_entry.grid(row=1, column=1, sticky=tk.EW, pady=4)
        self.angle_entry.insert(0, self.DEFAULT_MANUAL_ANGLE)
        manual_tab.columnconfigure(1, weight=1)

        ttk.Label(ideal_tab, text="Distancia objetivo (m):").grid(
            row=0, column=0, sticky=tk.W, pady=4
        )
        self.target_range_entry = ttk.Entry(ideal_tab, width=18)
        self.target_range_entry.grid(row=0, column=1, sticky=tk.EW, pady=4)
        self.target_range_entry.insert(0, self.DEFAULT_TARGET_RANGE)

        ttk.Label(
            ideal_tab,
            text="Usa 45° y calcula la velocidad mínima ideal.",
            foreground="dimgray",
        ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        ideal_tab.columnconfigure(1, weight=1)

        gps_fields = [
            ("Latitud origen (°):", "origin_lat_entry", self.DEFAULT_ORIGIN_LAT),
            ("Longitud origen (°):", "origin_lon_entry", self.DEFAULT_ORIGIN_LON),
            ("Latitud destino (°):", "dest_lat_entry", self.DEFAULT_DEST_LAT),
            ("Longitud destino (°):", "dest_lon_entry", self.DEFAULT_DEST_LON),
        ]
        for row, (label_text, attr_name, default_value) in enumerate(gps_fields):
            ttk.Label(earth_tab, text=label_text).grid(
                row=row,
                column=0,
                sticky=tk.W,
                pady=4,
            )
            entry = ttk.Entry(earth_tab, width=18)
            entry.grid(row=row, column=1, sticky=tk.EW, pady=4)
            entry.insert(0, default_value)
            setattr(self, attr_name, entry)

        ttk.Label(
            earth_tab,
            text="Escala en km; altura máxima fija de 10 km.",
            foreground="dimgray",
        ).grid(row=len(gps_fields), column=0, columnspan=2, sticky=tk.W, pady=(4, 0))
        earth_tab.columnconfigure(1, weight=1)

        ttk.Label(input_frame, text="Velocidad de reproducción:").grid(
            row=1, column=0, sticky=tk.W, pady=(12, 4)
        )
        self.playback_var = tk.DoubleVar(value=self.DEFAULT_PLAYBACK_SPEED)
        self.playback_scale = ttk.Scale(
            input_frame,
            from_=self.MIN_PLAYBACK_SPEED,
            to=self.MAX_PLAYBACK_SPEED,
            variable=self.playback_var,
            orient=tk.HORIZONTAL,
            command=self._update_playback_label,
        )
        self.playback_scale.grid(row=1, column=1, sticky=tk.EW, pady=(12, 4))
        self.playback_label = ttk.Label(input_frame, text="1.00x")
        self.playback_label.grid(row=2, column=1, sticky=tk.E, pady=(0, 8))

        launch_button = ttk.Button(
            input_frame,
            text="Lanzar proyectil",
            command=self.start_animation,
            style="Accent.TButton",
        )
        launch_button.grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(8, 4))

        reset_button = ttk.Button(
            input_frame,
            text="Reiniciar",
            command=self.reset_simulation,
        )
        reset_button.grid(row=4, column=0, columnspan=2, sticky=tk.EW, pady=4)

        input_frame.columnconfigure(1, weight=1)

    def _build_results_panel(self, parent):
        """Construye el panel de resultados generales calculados."""
        results_frame = ttk.LabelFrame(
            parent,
            text="Panel de resultados",
            padding=12,
            style="Section.TLabelframe",
        )
        results_frame.pack(fill=tk.X, pady=(0, 10))

        for row, (key, label_text) in enumerate(self.RESULT_LABELS):
            ttk.Label(results_frame, text=label_text).grid(
                row=row,
                column=0,
                sticky=tk.W,
                pady=2,
            )
            self.result_vars[key] = tk.StringVar(value="--")
            self.result_labels[key] = label_text
            ttk.Label(results_frame, textvariable=self.result_vars[key]).grid(
                row=row, column=1, sticky=tk.E, pady=2
            )

        ttk.Button(
            results_frame,
            text="Copiar resultados",
            command=self.copy_all_results,
        ).grid(
            row=len(self.RESULT_LABELS),
            column=0,
            columnspan=2,
            sticky=tk.EW,
            pady=(10, 0),
        )

        results_frame.columnconfigure(1, weight=1)

    def _build_realtime_panel(self, parent):
        """Construye el panel con los valores actualizados durante la animación."""
        realtime_frame = ttk.LabelFrame(
            parent,
            text="Visualización en tiempo real",
            padding=12,
            style="Section.TLabelframe",
        )
        realtime_frame.pack(fill=tk.X)

        for row, (key, label_text) in enumerate(self.REALTIME_LABELS):
            ttk.Label(realtime_frame, text=label_text).grid(
                row=row,
                column=0,
                sticky=tk.W,
                pady=2,
            )
            self.realtime_vars[key] = tk.StringVar(value="--")
            ttk.Label(realtime_frame, textvariable=self.realtime_vars[key]).grid(
                row=row, column=1, sticky=tk.E, pady=2
            )

        realtime_frame.columnconfigure(1, weight=1)

    def _build_plot(self, parent):
        """Inicializa la figura de Matplotlib embebida en Tkinter."""
        self.figure = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=parent)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._setup_empty_plot()

    def _build_output_notebook(self, parent):
        """Construye las pestañas de gráfica y explicación matemática."""
        self.output_notebook = ttk.Notebook(parent)
        self.output_notebook.grid(row=0, column=0, sticky="nsew")

        trajectory_tab = ttk.Frame(self.output_notebook)
        math_tab = ttk.Frame(self.output_notebook)
        earth_tab = ttk.Frame(self.output_notebook)
        self.output_notebook.add(trajectory_tab, text="Trajectory")
        self.output_notebook.add(math_tab, text="Math Explained")
        self.output_notebook.add(earth_tab, text="Earth 3D")

        trajectory_tab.rowconfigure(0, weight=1)
        trajectory_tab.columnconfigure(0, weight=1)
        math_tab.rowconfigure(1, weight=1)
        math_tab.columnconfigure(0, weight=1)
        earth_tab.rowconfigure(0, weight=1)
        earth_tab.columnconfigure(0, weight=1)

        self._build_plot(trajectory_tab)
        self._build_math_panel(math_tab)
        self._build_earth_plot(earth_tab)

    def _build_math_panel(self, parent):
        """Construye la vista scrollable del desarrollo matemático."""
        toolbar = ttk.Frame(parent, padding=(8, 8, 8, 4))
        toolbar.grid(row=0, column=0, columnspan=2, sticky="ew")
        toolbar.columnconfigure(0, weight=1)

        ttk.Button(
            toolbar,
            text="Copiar",
            command=self.copy_math_explanation,
        ).grid(row=0, column=1, sticky=tk.E)

        self.math_text_widget = tk.Text(
            parent,
            wrap=tk.WORD,
            font=("Helvetica", 12),
            padx=16,
            pady=12,
            state=tk.DISABLED,
            background="#fbfbfb",
            borderwidth=0,
            highlightthickness=0,
        )
        self.math_text_widget.grid(row=1, column=0, sticky="nsew")
        self._configure_math_text_tags()

        scrollbar = ttk.Scrollbar(
            parent,
            orient=tk.VERTICAL,
            command=self.math_text_widget.yview,
        )
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.math_text_widget.configure(yscrollcommand=scrollbar.set)

    def _build_earth_plot(self, parent):
        """Inicializa la figura 3D para la ruta aérea sobre la Tierra."""
        self.earth_figure = Figure(figsize=(8, 5), dpi=100)
        self.earth_ax = self.earth_figure.add_subplot(111, projection="3d")
        self.earth_canvas = FigureCanvasTkAgg(self.earth_figure, master=parent)
        self.earth_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._setup_earth_plot()

    def _configure_math_text_tags(self):
        """Define estilos para la explicación matemática."""
        self.math_text_widget.tag_configure(
            "title",
            font=("Helvetica", 20, "bold"),
            foreground="#1f2933",
            spacing1=6,
            spacing3=12,
        )
        self.math_text_widget.tag_configure(
            "section",
            font=("Helvetica", 15, "bold"),
            foreground="#1f4e79",
            spacing1=16,
            spacing3=8,
        )
        self.math_text_widget.tag_configure(
            "label",
            font=("Helvetica", 11, "bold"),
            foreground="#5f6368",
            spacing1=6,
            spacing3=4,
        )
        self.math_text_widget.tag_configure(
            "formula",
            font=("Menlo", 13),
            background="#eef5ff",
            foreground="#111827",
            lmargin1=18,
            lmargin2=18,
            rmargin=18,
            spacing1=4,
            spacing3=6,
        )
        self.math_text_widget.tag_configure(
            "substitution",
            font=("Menlo", 13),
            background="#f6f8fa",
            foreground="#1f2933",
            lmargin1=18,
            lmargin2=18,
            rmargin=18,
            spacing1=4,
            spacing3=6,
        )
        self.math_text_widget.tag_configure(
            "result",
            font=("Helvetica", 12, "bold"),
            background="#e8f5e9",
            foreground="#1b5e20",
            lmargin1=18,
            lmargin2=18,
            rmargin=18,
            spacing1=4,
            spacing3=10,
        )
        self.math_text_widget.tag_configure(
            "data",
            font=("Menlo", 12),
            background="#fff7e6",
            foreground="#2f3136",
            lmargin1=18,
            lmargin2=18,
            rmargin=18,
            spacing1=4,
            spacing3=6,
        )
        self.math_text_widget.tag_configure(
            "note",
            font=("Helvetica", 13, "italic"),
            foreground="#5f6368",
            spacing1=12,
        )

    def _update_playback_label(self, _event=None):
        """Sincroniza la etiqueta visible con el valor del control deslizante."""
        self.playback_label.config(text=f"{self.playback_var.get():.2f}x")

    def _load_cannon_image(self):
        """Carga el símbolo del cañón si el archivo está disponible."""
        if not self.CANNON_IMAGE_PATH.exists():
            return None
        return mpimg.imread(self.CANNON_IMAGE_PATH)

    def copy_to_clipboard(self, text, confirmation):
        """Copia texto al portapapeles del sistema y confirma la acción."""
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.root.update()
        messagebox.showinfo("Copiado", confirmation)

    def copy_all_results(self):
        """Copia los resultados generales visibles en formato de texto."""
        lines = []
        for key, label in self.result_labels.items():
            lines.append(f"{label.rstrip(':')}: {self.result_vars[key].get()}")

        self.copy_to_clipboard(
            "\n".join(lines),
            "Resultados copiados al portapapeles.",
        )

    def copy_math_explanation(self):
        """Copia el desarrollo matemático visible."""
        self.copy_to_clipboard(
            self.current_math_explanation,
            "Desarrollo matemático copiado al portapapeles.",
        )

    def build_math_explanation(self):
        """Construye el desarrollo matemático con los valores actuales."""
        if self.simulation_context.get("mode") == "Ruta GPS 3D":
            return self.EARTH_MATH_EXPLANATION

        if self.trajectory is None:
            return self.INITIAL_MATH_EXPLANATION

        initial_velocity = self.simulation_context["initial_velocity"]
        angle = self.simulation_context["angle"]
        v0x = self.trajectory["v0x"]
        v0y = self.trajectory["v0y"]
        time_total = self.trajectory["time_total"]
        height_max = self.trajectory["height_max"]
        range_x = self.trajectory["range_x"]

        return f"""Math Explained
================

Datos de la simulación
----------------------
v_0 = {initial_velocity:.2f} m/s
theta = {angle:.2f}°
g = {self.GRAVITY:.2f} m/s²
x_0 = 0 m
y_0 = 0 m

1. Componentes de la velocidad inicial
--------------------------------------
Fórmula general:
v_{{0x}} = v_0 · cos(theta)
v_{{0y}} = v_0 · sin(theta)

Sustitución:
v_{{0x}} = {initial_velocity:.2f} · cos({angle:.2f}°) = {v0x:.2f} m/s
v_{{0y}} = {initial_velocity:.2f} · sin({angle:.2f}°) = {v0y:.2f} m/s

2. Posición horizontal en función del tiempo
--------------------------------------------
Fórmula general:
x(t) = x_0 + v_{{0x}} · t

Sustitución:
x(t) = 0 + {v0x:.2f} · t

3. Posición vertical en función del tiempo
------------------------------------------
Fórmula general:
y(t) = y_0 + v_{{0y}} · t - 1/2 · g · t²

Sustitución:
y(t) = 0 + {v0y:.2f} · t - 1/2 · {self.GRAVITY:.2f} · t²

4. Trayectoria independiente del tiempo: y(x)
---------------------------------------------
Fórmula general:
y(x) = x · tan(theta) - (g · x²) / (2 · v_0² · cos²(theta))

Sustitución:
y(x) = x · tan({angle:.2f}°) - ({self.GRAVITY:.2f} · x²) /
       (2 · {initial_velocity:.2f}² · cos²({angle:.2f}°))

5. Puntos clave del movimiento
------------------------------
Tiempo de vuelo total:
t_total = 2 · v_{{0y}} / g
t_total = 2 · {v0y:.2f} / {self.GRAVITY:.2f} = {time_total:.2f} s

Altura máxima alcanzada:
Y_max = v_{{0y}}² / (2g)
Y_max = {v0y:.2f}² / (2 · {self.GRAVITY:.2f}) = {height_max:.2f} m

Alcance horizontal máximo:
X_max = v_0² · sin(2theta) / g
X_max = {initial_velocity:.2f}² · sin(2 · {angle:.2f}°) /
        {self.GRAVITY:.2f} = {range_x:.2f} m
"""

    def build_math_blocks(self):
        """Construye bloques etiquetados para renderizar la explicación."""
        if self.simulation_context.get("mode") == "Ruta GPS 3D":
            return [("note", self.EARTH_MATH_EXPLANATION)]

        if self.trajectory is None:
            return self.INITIAL_MATH_BLOCKS

        initial_velocity = self.simulation_context["initial_velocity"]
        angle = self.simulation_context["angle"]
        v0x = self.trajectory["v0x"]
        v0y = self.trajectory["v0y"]
        time_total = self.trajectory["time_total"]
        height_max = self.trajectory["height_max"]
        range_x = self.trajectory["range_x"]

        return [
            ("title", "Math Explained"),
            ("section", "Datos de la simulación"),
            (
                "data",
                "\n".join(
                    [
                        f"v_0 = {initial_velocity:.2f} m/s",
                        f"theta = {angle:.2f}°",
                        f"g = {self.GRAVITY:.2f} m/s²",
                        "x_0 = 0 m",
                        "y_0 = 0 m",
                    ]
                ),
            ),
            ("section", "1. Componentes de la velocidad inicial"),
            ("label", "Fórmula general"),
            ("formula", "v_{0x} = v_0 · cos(theta)\nv_{0y} = v_0 · sin(theta)"),
            ("label", "Sustitución"),
            (
                "substitution",
                "\n".join(
                    [
                        (
                            "v_{0x} = "
                            f"{initial_velocity:.2f} · cos({angle:.2f}°)"
                            f" = {v0x:.2f} m/s"
                        ),
                        (
                            "v_{0y} = "
                            f"{initial_velocity:.2f} · sin({angle:.2f}°)"
                            f" = {v0y:.2f} m/s"
                        ),
                    ]
                ),
            ),
            ("section", "2. Posición horizontal en función del tiempo"),
            ("label", "Fórmula general"),
            ("formula", "x(t) = x_0 + v_{0x} · t"),
            ("label", "Sustitución"),
            ("substitution", f"x(t) = 0 + {v0x:.2f} · t"),
            ("section", "3. Posición vertical en función del tiempo"),
            ("label", "Fórmula general"),
            ("formula", "y(t) = y_0 + v_{0y} · t - 1/2 · g · t²"),
            ("label", "Sustitución"),
            (
                "substitution",
                f"y(t) = 0 + {v0y:.2f} · t - 1/2 · {self.GRAVITY:.2f} · t²",
            ),
            ("section", "4. Trayectoria independiente del tiempo: y(x)"),
            ("label", "Fórmula general"),
            (
                "formula",
                "y(x) = x · tan(theta) - (g · x²) / "
                "(2 · v_0² · cos²(theta))",
            ),
            ("label", "Sustitución"),
            (
                "substitution",
                "y(x) = x · tan({angle:.2f}°) - ({gravity:.2f} · x²) / "
                "(2 · {velocity:.2f}² · cos²({angle:.2f}°))".format(
                    angle=angle,
                    gravity=self.GRAVITY,
                    velocity=initial_velocity,
                ),
            ),
            ("section", "5. Puntos clave del movimiento"),
            ("label", "Tiempo de vuelo total"),
            ("formula", "t_total = 2 · v_{0y} / g"),
            (
                "result",
                f"t_total = 2 · {v0y:.2f} / {self.GRAVITY:.2f} = {time_total:.2f} s",
            ),
            ("label", "Altura máxima alcanzada"),
            ("formula", "Y_max = v_{0y}² / (2g)"),
            (
                "result",
                f"Y_max = {v0y:.2f}² / (2 · {self.GRAVITY:.2f}) = "
                f"{height_max:.2f} m",
            ),
            ("label", "Alcance horizontal máximo"),
            ("formula", "X_max = v_0² · sin(2theta) / g"),
            (
                "result",
                f"X_max = {initial_velocity:.2f}² · sin(2 · {angle:.2f}°) / "
                f"{self.GRAVITY:.2f} = {range_x:.2f} m",
            ),
        ]

    def update_math_explanation(self):
        """Refresca la pestaña matemática con el texto actual."""
        self.current_math_explanation = self.build_math_explanation()
        if self.math_text_widget is None:
            return

        self.math_text_widget.configure(state=tk.NORMAL)
        self.math_text_widget.delete("1.0", tk.END)
        for tag, text in self.build_math_blocks():
            self.math_text_widget.insert(tk.END, text + "\n\n", tag)
        self.math_text_widget.configure(state=tk.DISABLED)

    def validate_manual_inputs(self):
        """Valida velocidad y ángulo para el modo manual."""
        try:
            velocity = float(self.velocity_entry.get())
            angle = float(self.angle_entry.get())
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

        return {
            "initial_velocity": velocity,
            "angle": angle,
            "mode": "Velocidad y ángulo",
            "target_range": None,
        }

    def validate_distance_inputs(self):
        """Valida la distancia objetivo para el modo de alcance ideal."""
        try:
            target_range = float(self.target_range_entry.get())
        except ValueError:
            messagebox.showerror(
                "Entrada inválida",
                "La distancia objetivo debe contener un valor numérico.",
            )
            return None

        if target_range <= 0:
            messagebox.showerror(
                "Entrada inválida",
                "La distancia objetivo debe ser mayor que cero.",
            )
            return None

        initial_velocity, angle = self.calculate_ideal_launch_for_range(target_range)
        return {
            "initial_velocity": initial_velocity,
            "angle": angle,
            "mode": "Distancia objetivo",
            "target_range": target_range,
        }

    def validate_gps_inputs(self):
        """Valida coordenadas GPS para la simulación 3D."""
        try:
            origin_lat = float(self.origin_lat_entry.get())
            origin_lon = float(self.origin_lon_entry.get())
            dest_lat = float(self.dest_lat_entry.get())
            dest_lon = float(self.dest_lon_entry.get())
        except ValueError:
            messagebox.showerror(
                "Entrada inválida",
                "Todas las coordenadas GPS deben contener valores numéricos.",
            )
            return None

        coordinates = [
            ("Latitud origen", origin_lat, -90, 90),
            ("Latitud destino", dest_lat, -90, 90),
            ("Longitud origen", origin_lon, -180, 180),
            ("Longitud destino", dest_lon, -180, 180),
        ]
        for label, value, minimum, maximum in coordinates:
            if value < minimum or value > maximum:
                messagebox.showerror(
                    "Entrada inválida",
                    f"{label} debe estar entre {minimum}° y {maximum}°.",
                )
                return None

        if np.isclose(origin_lat, dest_lat) and np.isclose(origin_lon, dest_lon):
            messagebox.showerror(
                "Entrada inválida",
                "El origen y el destino GPS no pueden ser idénticos.",
            )
            return None

        return origin_lat, origin_lon, dest_lat, dest_lon

    def validate_playback_speed(self):
        """Valida el factor de reproducción compartido por ambos modos."""
        try:
            playback_speed = float(self.playback_var.get())
        except ValueError:
            return self.DEFAULT_PLAYBACK_SPEED

        return max(playback_speed, self.MIN_PLAYBACK_SPEED)

    def get_launch_settings(self):
        """Obtiene los valores de lanzamiento según la pestaña activa."""
        active_tab = self.simulation_notebook.index("current")
        if active_tab == 0:
            return self.validate_manual_inputs()
        if active_tab == 1:
            return self.validate_distance_inputs()
        return None

    def gps_to_cartesian(self, latitude, longitude, radius):
        """Convierte coordenadas GPS a coordenadas cartesianas 3D."""
        lat_rad = np.radians(latitude)
        lon_rad = np.radians(longitude)
        x_value = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y_value = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z_value = radius * np.sin(lat_rad)
        return np.array([x_value, y_value, z_value])

    def calculate_earth_route(self, origin_lat, origin_lon, dest_lat, dest_lon):
        """Calcula una ruta aérea arqueada sobre una Tierra esférica."""
        origin = self.gps_to_cartesian(origin_lat, origin_lon, self.EARTH_RADIUS_KM)
        destination = self.gps_to_cartesian(dest_lat, dest_lon, self.EARTH_RADIUS_KM)
        origin_unit = origin / np.linalg.norm(origin)
        destination_unit = destination / np.linalg.norm(destination)

        dot_product = np.clip(np.dot(origin_unit, destination_unit), -1.0, 1.0)
        central_angle = np.arccos(dot_product)
        sin_angle = np.sin(central_angle)

        progress = np.linspace(0.0, 1.0, self.EARTH_ROUTE_FRAMES)
        if np.isclose(sin_angle, 0.0):
            directions = np.outer(1.0 - progress, origin_unit) + np.outer(
                progress,
                destination_unit,
            )
            directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
        else:
            start_weights = np.sin((1.0 - progress) * central_angle) / sin_angle
            end_weights = np.sin(progress * central_angle) / sin_angle
            directions = (
                start_weights[:, np.newaxis] * origin_unit
                + end_weights[:, np.newaxis] * destination_unit
            )

        altitude = 4 * self.FLIGHT_MAX_ALTITUDE_KM * progress * (1 - progress)
        radius_values = self.EARTH_RADIUS_KM + altitude
        route_points = directions * radius_values[:, np.newaxis]

        return {
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "dest_lat": dest_lat,
            "dest_lon": dest_lon,
            "origin": origin,
            "destination": destination,
            "points": route_points,
            "altitude": altitude,
            "surface_distance": self.EARTH_RADIUS_KM * central_angle,
        }

    def calculate_ideal_launch_for_range(self, target_range):
        """Calcula la velocidad mínima ideal para un alcance horizontal dado."""
        angle_degrees = self.IDEAL_ANGLE_DEGREES
        initial_velocity = np.sqrt(target_range * self.GRAVITY)
        return initial_velocity, angle_degrees

    def calculate_trajectory(self, initial_velocity, angle_degrees):
        """Calcula posiciones, velocidades y magnitudes de la trayectoria."""
        angle_radians = np.radians(angle_degrees)
        v0x = initial_velocity * np.cos(angle_radians)
        v0y = initial_velocity * np.sin(angle_radians)

        time_total = 2 * v0y / self.GRAVITY
        height_max = v0y**2 / (2 * self.GRAVITY)
        range_x = v0x * time_total

        frames = max(
            self.MIN_TRAJECTORY_FRAMES,
            int(time_total * self.FRAMES_PER_SECOND_OF_FLIGHT),
        )
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
        """Inicia una nueva simulación con los valores de la pestaña activa."""
        if self.simulation_notebook.index("current") == 2:
            gps_values = self.validate_gps_inputs()
            if gps_values is None:
                return

            route_data = self.calculate_earth_route(*gps_values)
            self.start_earth_animation(route_data)
            return

        launch_settings = self.get_launch_settings()
        if launch_settings is None:
            return

        initial_velocity = launch_settings["initial_velocity"]
        angle = launch_settings["angle"]
        playback_speed = self.validate_playback_speed()
        self._stop_animation()
        self._stop_earth_animation()
        self.earth_route = None
        self._setup_earth_plot()
        self.output_notebook.select(0)

        self.trajectory = self.calculate_trajectory(initial_velocity, angle)
        self.simulation_context = {
            "mode": launch_settings["mode"],
            "target_range": launch_settings["target_range"],
            "initial_velocity": initial_velocity,
            "angle": angle,
        }
        self._update_general_results(frame=0)
        self._setup_simulation_plot(angle)
        self.update_math_explanation()

        interval_ms = max(
            self.MIN_ANIMATION_INTERVAL_MS,
            int(self.BASE_ANIMATION_INTERVAL_MS / playback_speed),
        )
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
        """Actualiza la bala, la trayectoria recorrida y los datos en pantalla."""
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
        """Limpia la gráfica y los paneles para una nueva simulación."""
        self._stop_animation()
        self._stop_earth_animation()
        self.trajectory = None
        self.earth_route = None
        self.simulation_context = {}
        self._setup_empty_plot()
        self._setup_earth_plot()

        for variable in self.result_vars.values():
            variable.set("--")
        for variable in self.realtime_vars.values():
            variable.set("--")

        self.update_math_explanation()
        self.canvas.draw_idle()
        self.earth_canvas.draw_idle()

    def _stop_animation(self):
        """Detiene la animación activa, si existe."""
        if self.animation is not None and self.animation.event_source is not None:
            self.animation.event_source.stop()
        self.animation = None

    def _stop_earth_animation(self):
        """Detiene la animación 3D activa, si existe."""
        if self.earth_animation is not None and self.earth_animation.event_source is not None:
            self.earth_animation.event_source.stop()
        self.earth_animation = None

    def _setup_empty_plot(self):
        """Dibuja el estado inicial de la gráfica antes de una simulación."""
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
        """Prepara la gráfica con límites y marcadores de la trayectoria."""
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

    def _setup_earth_plot(self):
        """Dibuja una Tierra esférica en escala de kilómetros."""
        if self.earth_ax is None:
            return

        self.earth_ax.clear()
        radius = self.EARTH_RADIUS_KM
        longitude_values = np.linspace(0, 2 * np.pi, 72)
        latitude_values = np.linspace(0, np.pi, 36)
        sphere_x = radius * np.outer(np.cos(longitude_values), np.sin(latitude_values))
        sphere_y = radius * np.outer(np.sin(longitude_values), np.sin(latitude_values))
        sphere_z = radius * np.outer(
            np.ones_like(longitude_values),
            np.cos(latitude_values),
        )

        self.earth_ax.plot_surface(
            sphere_x,
            sphere_y,
            sphere_z,
            color="deepskyblue",
            alpha=0.24,
            linewidth=0,
            shade=True,
        )
        self.earth_ax.set_title("Ruta aérea 3D sobre la Tierra")
        self.earth_ax.set_xlabel("X (km)")
        self.earth_ax.set_ylabel("Y (km)")
        self.earth_ax.set_zlabel("Z (km)")

        axis_limit = radius + 250
        self.earth_ax.set_xlim(-axis_limit, axis_limit)
        self.earth_ax.set_ylim(-axis_limit, axis_limit)
        self.earth_ax.set_zlim(-axis_limit, axis_limit)
        self.earth_ax.set_box_aspect((1, 1, 1))
        self.earth_ax.view_init(elev=24, azim=-60)

    def start_earth_animation(self, route_data):
        """Inicia la animación 3D de una ruta aérea."""
        self._stop_animation()
        self._stop_earth_animation()
        self.trajectory = None
        self.earth_route = route_data
        self.output_notebook.select(2)
        self.simulation_context = {
            "mode": "Ruta GPS 3D",
            "target_range": None,
            "initial_velocity": 0.0,
            "angle": 0.0,
        }

        points = route_data["points"]
        self._setup_earth_plot()
        (self.earth_route_artist,) = self.earth_ax.plot(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            color="crimson",
            linewidth=2.8,
            label="Ruta aérea",
        )
        self.earth_ax.scatter(
            *route_data["origin"],
            color="limegreen",
            s=60,
            label="Origen",
            depthshade=True,
        )
        self.earth_ax.scatter(
            *route_data["destination"],
            color="darkorange",
            s=60,
            label="Destino",
            depthshade=True,
        )
        (self.plane_artist,) = self.earth_ax.plot(
            [points[0, 0]],
            [points[0, 1]],
            [points[0, 2]],
            marker="^",
            color="black",
            markersize=9,
            linestyle="None",
            label="Avión",
        )
        self.earth_ax.legend(loc="upper right")
        self._update_earth_results(frame=0)
        self.current_math_explanation = self.EARTH_MATH_EXPLANATION
        self.update_math_explanation()

        playback_speed = self.validate_playback_speed()
        interval_ms = max(
            self.MIN_ANIMATION_INTERVAL_MS,
            int(self.BASE_ANIMATION_INTERVAL_MS / playback_speed),
        )
        self.earth_animation = animation.FuncAnimation(
            self.earth_figure,
            self.update_earth_frame,
            frames=len(points),
            interval=interval_ms,
            blit=False,
            repeat=False,
        )
        self.earth_canvas.draw_idle()

    def update_earth_frame(self, frame):
        """Actualiza la posición del avión en la ruta 3D."""
        if self.earth_route is None or self.plane_artist is None:
            return []

        points = self.earth_route["points"]
        x_value, y_value, z_value = points[frame]
        self.plane_artist.set_data([x_value], [y_value])
        self.plane_artist.set_3d_properties([z_value])
        self._update_earth_results(frame)

        if frame >= len(points) - 1:
            self._stop_earth_animation()

        return [self.plane_artist]

    def _draw_cannon(self, angle_degrees):
        """Dibuja el cañón en el origen usando imagen o figuras simples."""
        if self.cannon_image is not None:
            image = OffsetImage(self.cannon_image, zoom=0.82)
            cannon_symbol = AnnotationBbox(
                image,
                (0, 0),
                xybox=(12, 12),
                xycoords="data",
                boxcoords="offset points",
                frameon=False,
                zorder=6,
            )
            self.ax.add_artist(cannon_symbol)
            return

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
        """Actualiza las magnitudes generales mostradas en el panel."""
        if self.trajectory is None:
            return

        target_range = self.simulation_context.get("target_range")
        target_text = "--" if target_range is None else f"{target_range:.2f} m"

        self.result_vars["mode"].set(self.simulation_context.get("mode", "--"))
        self.result_vars["target_range"].set(target_text)
        self.result_vars["initial_velocity"].set(
            f"{self.simulation_context.get('initial_velocity', 0.0):.2f} m/s"
        )
        self.result_vars["launch_angle"].set(
            f"{self.simulation_context.get('angle', 0.0):.2f}°"
        )
        self.result_vars["v0x"].set(f"{self.trajectory['v0x']:.2f} m/s")
        self.result_vars["v0y"].set(f"{self.trajectory['v0y']:.2f} m/s")
        self.result_vars["time_total"].set(f"{self.trajectory['time_total']:.2f} s")
        self.result_vars["height_max"].set(f"{self.trajectory['height_max']:.2f} m")
        self.result_vars["range_x"].set(f"{self.trajectory['range_x']:.2f} m")
        self.result_vars["current_vy"].set(f"{self.trajectory['vy'][frame]:.2f} m/s")
        self.result_vars["current_speed"].set(f"{self.trajectory['speed'][frame]:.2f} m/s")
        self.result_vars["gps_origin"].set("--")
        self.result_vars["gps_destination"].set("--")
        self.result_vars["earth_radius"].set("--")
        self.result_vars["surface_distance"].set("--")
        self.result_vars["flight_altitude"].set("--")

    def _update_earth_results(self, frame):
        """Actualiza los resultados del modo de ruta GPS 3D."""
        if self.earth_route is None:
            return

        progress = frame / max(len(self.earth_route["points"]) - 1, 1)
        current_point = self.earth_route["points"][frame]

        self.result_vars["mode"].set("Ruta GPS 3D")
        self.result_vars["target_range"].set("--")
        self.result_vars["initial_velocity"].set("--")
        self.result_vars["launch_angle"].set("--")
        self.result_vars["v0x"].set("--")
        self.result_vars["v0y"].set("--")
        self.result_vars["time_total"].set("--")
        self.result_vars["height_max"].set("--")
        self.result_vars["range_x"].set("--")
        self.result_vars["current_vy"].set("--")
        self.result_vars["current_speed"].set("--")
        self.result_vars["gps_origin"].set(
            f"{self.earth_route['origin_lat']:.4f}°, {self.earth_route['origin_lon']:.4f}°"
        )
        self.result_vars["gps_destination"].set(
            f"{self.earth_route['dest_lat']:.4f}°, {self.earth_route['dest_lon']:.4f}°"
        )
        self.result_vars["earth_radius"].set(f"{self.EARTH_RADIUS_KM:.0f} km")
        self.result_vars["surface_distance"].set(
            f"{self.earth_route['surface_distance']:.2f} km"
        )
        self.result_vars["flight_altitude"].set(f"{self.FLIGHT_MAX_ALTITUDE_KM:.2f} km")

        self.realtime_vars["time"].set(f"{progress * 100:.1f} %")
        self.realtime_vars["x"].set(f"{current_point[0]:.2f} km")
        self.realtime_vars["y"].set(f"{current_point[1]:.2f} km")
        self.realtime_vars["vx"].set(f"{current_point[2]:.2f} km")
        self.realtime_vars["vy"].set(f"{self.earth_route['altitude'][frame]:.2f} km")

    def _update_realtime_values(self, frame):
        """Actualiza los valores instantáneos del cuadro actual."""
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
