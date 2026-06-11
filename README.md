# Simulador de movimiento parabólico

Proyecto en Python que simula y anima el movimiento parabólico ideal de una bala de cañón, sin resistencia del aire. La aplicación incluye una interfaz gráfica con Tkinter, una gráfica embebida con Matplotlib y cálculos numéricos con NumPy.

## Requisitos

- Python 3
- Tkinter disponible en la instalación de Python
- Matplotlib
- NumPy

## Instalación

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En macOS con Python instalado mediante Homebrew, si aparece el error
`ModuleNotFoundError: No module named '_tkinter'`, instale Tkinter para la
misma versión de Python:

```bash
brew install python-tk@3.14
```

Luego cree de nuevo el entorno virtual para que use el Python con soporte de
Tkinter:

```bash
deactivate
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Ejecución

```bash
python projectile_simulator.py
```

## Uso

1. Seleccione una de las dos pestañas de simulación:
   - **Velocidad y ángulo**: introduzca la velocidad inicial y el ángulo de lanzamiento.
   - **Distancia objetivo**: introduzca la distancia horizontal que debe alcanzar la bala.
2. Ajuste la velocidad de reproducción si desea una animación más lenta o rápida.
3. Presione **Lanzar proyectil**.
4. Use **Reiniciar** para limpiar la simulación e ingresar nuevos valores.

La interfaz valida que los campos contengan valores numéricos, que la velocidad
sea mayor que cero, que el ángulo sea mayor que 0° y menor que 90°, y que la
distancia objetivo sea mayor que cero.

## Modelo físico

El simulador utiliza el modelo ideal de movimiento parabólico desde el origen:

```text
x0 = 0
y0 = 0
g = 9.81 m/s^2
```

Primero se descompone la velocidad inicial:

```text
v0x = v0 cos(theta)
v0y = v0 sin(theta)
```

La posición del proyectil en cada instante se calcula con:

```text
x(t) = v0 cos(theta) t
y(t) = v0 sin(theta) t - 1/2 g t^2
```

La velocidad vertical instantánea y la rapidez total se calculan con:

```text
vx = v0x
vy(t) = v0y - g t
rapidez(t) = sqrt(vx^2 + vy(t)^2)
```

También se muestran:

```text
Tiempo total de vuelo = 2 v0y / g
Altura máxima = v0y^2 / (2g)
Alcance horizontal = v0x * tiempo_total
```

## Simulación por distancia objetivo

En el modo **Distancia objetivo**, el usuario introduce el alcance horizontal
deseado. El programa calcula la trayectoria ideal de alcance máximo usando:

```text
theta = 45°
R = v0^2 sin(2 theta) / g
```

Como `theta = 45°`, entonces `sin(90°) = 1`, por lo que:

```text
v0 = sqrt(Rg)
```

Con esa velocidad inicial calculada y el ángulo ideal de 45°, el simulador
genera la misma animación de trayectoria parabólica.

## Organización del código

El archivo principal `projectile_simulator.py` contiene la clase
`ProjectileSimulator`, que centraliza la interfaz, los cálculos físicos y la
animación. El código está separado por responsabilidades:

- Construcción de interfaz: paneles de entrada, resultados, datos en tiempo real
  y gráfica.
- Validación de entradas: modo manual, modo por distancia objetivo y velocidad de
  reproducción.
- Cálculos físicos: trayectoria parabólica y lanzamiento ideal para un alcance
  dado.
- Animación y visualización: actualización de la bala, trayectoria recorrida,
  marcadores y resultados visibles.

## Ejemplo de prueba

Modo **Velocidad y ángulo**:

```text
Velocidad inicial: 13 m/s
Ángulo: 30°
```

El programa debe mostrar aproximadamente:

```text
Velocidad horizontal: 11.26 m/s
Velocidad vertical inicial: 6.50 m/s
Tiempo de vuelo: 1.33 s
Altura máxima: 2.15 m
Alcance horizontal: 14.92 m
```

Modo **Distancia objetivo**:

```text
Distancia objetivo: 14.92 m
```

El programa debe calcular aproximadamente:

```text
Ángulo usado: 45°
Velocidad inicial usada: 12.10 m/s
Alcance horizontal: 14.92 m
```

## Archivos del proyecto

```text
README.md
requirements.txt
projectile_simulator.py
```
