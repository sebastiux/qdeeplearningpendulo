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
| Estado | $s = [x,\ \dot x,\ \theta,\ \dot\theta]$ (continuo) |
| Acciones | 41 fuerzas: $F \in \{-10, -9.5, \dots, +10\}$ N |
| Recompensa | $+1$ en la vertical, $-10$ al caer ($\lvert\theta\rvert > 12^\circ$) o al chocar con un tope ($\lvert x\rvert > 2.4$ m) |
| Dinámica | Ecuaciones no lineales acopladas de $\ddot\theta$ y $\ddot x$, integración de Euler con $\Delta T = 0.02$ s |
| Riel | Finito, con topes en $\pm 2.4$ m |
| Red Q | $4 \to 64 \to 64 \to 41$ (dos capas ocultas ReLU, salida lineal) |
| Entrenamiento | Adam con $\alpha = 0.0005$, $\gamma = 0.99$, 2,000 episodios de hasta 300 pasos |

Parámetros físicos: $M = 1$ kg, $m = 0.1$ kg, $L = 0.5$ m, $g = 9.8$ m/s².

### Sobre el estado

El PDF usa $s = [\theta,\ \dot\theta]$. Con un riel finito eso no alcanza: la red no sabe dónde está el carro.
Entrenada así, lo dejaba desplazarse 2.85 m en 10 s, más allá del tope. Por eso el estado incluye la posición
$x$ y la velocidad $\dot x$ del carro.

### Sobre la recompensa

$$r_t = 1 - \left(\tfrac{\theta_t}{12^\circ}\right)^2 - 0.1\left(\tfrac{F_t}{10\,\text{N}}\right)^2 - 0.25\left(\tfrac{x_t}{2.4\,\text{m}}\right)^2, \qquad r = -10 \text{ al caer o chocar}$$

El máximo sigue siendo $+1$ por paso, o sea 300 por episodio. Cada término corrige algo que pasó al entrenar con la
recompensa literal:

- **$\theta^2$:** con el $+1$ binario el agente sobrevivía **inclinado ~11°**, acelerando el carro sin parar.
- **$F^2$:** sin costo de esfuerzo, la red profunda alternaba entre $+9.5$ y $-5.5$ N en cada paso y la
  superficial rendía igual.
- **$x^2$:** sin costo de posición, el carro solía quedar cerca de un tope.

## Resultados

Respuesta greedy desde $\theta_0 = 10^\circ$ y $x_0 = 0$; "régimen" son los últimos 3 s de una simulación de 5 s.

| Métrica | Superficial (4→24→3) | Profunda (4→64→64→41) |
| --- | ---: | ---: |
| Error RMS de θ en régimen | 1.01° | **0.28°** |
| \|θ\| máximo en régimen | 1.99° | **0.73°** |
| \|x\| del carro al final | 0.40 m | **0.02 m** |
| Fuerza media \|F\| | 0.33 N | **0.25 N** |
| Niveles de fuerza usados | 3 | 6 |
| Episodios hasta promedio ≥ 250 | **112** | 414 |
| Mejor evaluación greedy | 294.7 | **297.0** |

Con 135 arranques difíciles ($\theta_0$ de −10° a 10°, $\dot\theta_0$ de −1 a 1 rad/s y $x_0 \in \{-1.5, 0, 1.5\}$ m,
30 s cada uno), la profunda se recupera en el 92.6 % de los casos y la superficial en el 89.6 %.

Ambas redes equilibran el péndulo. La superficial converge antes porque explora 3 acciones en lugar de 41, pero sólo
puede corregir con golpes de ±10 N. La profunda modula la fuerza, es unas 3.6 veces más precisa y regresa el carro
al centro del riel.

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
python training/dqn_pendulo.py          # entrena la red profunda y la superficial (~5 min), exporta a web/data
npm test                                # prueba de humo
npm run serve                           # http://localhost:8000
```

## Despliegue

Cada `push` a `main` o a `claude/nifty-archimedes-rmrxlx` ejecuta `.github/workflows/deploy-pages.yml`, que corre
la prueba de humo y publica la carpeta `web/` en GitHub Pages. La primera vez hay que activar Pages en
**Settings → Pages → Build and deployment → Source: GitHub Actions**.
