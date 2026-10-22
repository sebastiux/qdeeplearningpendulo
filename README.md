# Péndulo invertido con Deep Q-Learning

Red neuronal profunda entrenada con **Deep Q-Learning (DQN)** para mantener de pie un péndulo invertido sobre un
carro. Resuelve los ejercicios de *Aprendizaje reforzado · Deep Q-learning* (Universidad Iberoamericana).

**Sitio:** https://sebastiux.github.io/qdeeplearningpendulo/

En el sitio están la explicación paso a paso, las ecuaciones, las curvas de aprendizaje, la comparación entre la
red superficial y la profunda, una demostración interactiva (puedes empujar el péndulo, controlarlo con el teclado
o cambiar de controlador) y un modo para **entrenar tu propia red en el navegador** con un Web Worker.

## El problema

| Elemento | Especificación |
| --- | --- |
| Estado | $s = [\theta,\ \dot\theta]$ (continuo) |
| Acciones | 41 fuerzas: $F \in \{-10, -9.5, \dots, +10\}$ N |
| Recompensa | $+1$ en la vertical, $-10$ al caer ($\lvert\theta\rvert > 12^\circ$) |
| Dinámica | Ecuaciones no lineales acopladas de $\ddot\theta$ y $\ddot x$, integración de Euler con $\Delta T = 0.02$ s |
| Red Q | $2 \to 64 \to 64 \to 41$ (dos capas ocultas ReLU, salida lineal) |
| Entrenamiento | Adam con $\alpha = 0.0005$, $\gamma = 0.99$, 2,000 episodios de hasta 300 pasos |

Parámetros físicos: $M = 1$ kg, $m = 0.1$ kg, $L = 0.5$ m, $g = 9.8$ m/s².

### Sobre la recompensa

El $+1$ binario del enunciado no distingue $0^\circ$ de $11^\circ$. Al entrenar con él, el agente aprendió a
sobrevivir los 300 pasos **inclinado ~11°**, acelerando el carro sin parar. Para que "vertical" signifique
$\theta \approx 0$ y que las 41 acciones sirvan para el control fino, la recompensa por paso es

$$r_t = 1 - \left(\tfrac{\theta_t}{12^\circ}\right)^2 - c\left(\tfrac{F_t}{10\,\text{N}}\right)^2, \qquad r = -10 \text{ al caer}$$

con $c = 0.1$. El máximo sigue siendo $+1$ por paso, o sea 300 por episodio. Sin el costo de esfuerzo la red
profunda alternaba entre $+9.5$ y $-5.5$ N en cada paso y la superficial rendía igual. Con él, las 41 acciones
sirven para el control fino.

## Resultados

Respuesta greedy desde $\theta_0 = 10^\circ$; "régimen" son los últimos 3 s de una simulación de 5 s.

| Métrica | Superficial (2→24→3) | Profunda (2→64→64→41) |
| --- | ---: | ---: |
| Error RMS de θ en régimen | 0.65° | **0.26°** |
| \|θ\| máximo en régimen | 1.30° | **0.61°** |
| Fuerza media \|F\| | 0.47 N | **0.36 N** |
| Niveles de fuerza usados | 3 | 5 |
| Episodios hasta promedio ≥ 250 | **100** | 341 |
| Mejor evaluación greedy | 296.4 | **297.7** |

Ambas redes equilibran el péndulo. La superficial converge antes porque explora 3 acciones en lugar de 41, pero sólo
puede corregir con golpes de ±10 N. La profunda modula la fuerza y es unas 2.5 veces más precisa.

## Estructura

```
training/dqn_pendulo.py   DQN desde cero en NumPy: ambiente, red, retropropagación, Adam, replay, red objetivo
web/                      Sitio estático (GitHub Pages)
  index.html, css/        Página y estilos
  js/physics.js           Simulador del carro-péndulo (idéntico al de Python)
  js/nn.js, js/dqn.js     Red neuronal, Adam y agente DQN en JavaScript
  js/train-worker.js      Entrenamiento en el navegador (Web Worker)
  js/render.js, main.js   Animaciones, gráficas y demo
  data/                   Pesos e historial exportados por el entrenamiento en Python
  vendor/                 Chart.js y KaTeX (incluidos localmente, sin CDN)
tests/smoke.mjs           Verifica que el modelo exportado equilibra el péndulo
.github/workflows/        Despliegue automático a GitHub Pages
```

## Reproducir

```bash
pip install -r training/requirements.txt
python training/dqn_pendulo.py          # entrena la red profunda y la superficial (~6 min), exporta a web/data
npm test                                # prueba de humo
npm run serve                           # http://localhost:8000
```

## Despliegue

Cada `push` a `main` o a `claude/nifty-archimedes-rmrxlx` ejecuta `.github/workflows/deploy-pages.yml`, que corre
la prueba de humo y publica la carpeta `web/` en GitHub Pages. La primera vez hay que activar Pages en
**Settings → Pages → Build and deployment → Source: GitHub Actions**.
