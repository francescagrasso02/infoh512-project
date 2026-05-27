---
Authors:
  - Loukas Kavouras
  - Eleni Psaroudaki
  - Konstantinos Tsopelas
  - Dimitrios Rontogiannis
  - Nikolaos Theologitis
  - Dimitris Sacharidis
  - Giorgos Giannopoulos
  - Dimitrios Tomaras
  - Kleopatra Markou
  - Dimitrios Gunopulos
  - Dimitris Fotakis
  - Ioaniis Emiris
---
*PDF*
[[GLANCE.pdf]]

```cardlink
url: https://arxiv.org/abs/2405.18921
title: "GLANCE: Global Actions in a Nutshell for Counterfactual Explainability"
description: "The widespread deployment of machine learning systems in critical real-world decision-making applications has highlighted the urgent need for counterfactual explainability methods that operate effectively. Global counterfactual explanations, expressed as actions to offer recourse, aim to provide succinct explanations and insights applicable to large population subgroups. High effectiveness, measured by the fraction of the population that is provided recourse, ensures that the actions benefit as many individuals as possible. Keeping the cost of actions low ensures the proposed recourse actions remain practical and actionable. Limiting the number of actions that provide global counterfactuals is essential to maximizing interpretability. The primary challenge, therefore, is to balance these trade-offs--maximizing effectiveness, minimizing cost, while maintaining a small number of actions. We introduce $\\texttt{GLANCE}$, a versatile and adaptive algorithm that employs a novel agglomerative approach, jointly considering both the feature space and the space of counterfactual actions, thereby accounting for the distribution of points in a way that aligns with the model's structure. This design enables the careful balancing of the trade-offs among the three key objectives, with the size objective functioning as a tunable parameter to keep the actions few and easy to interpret. Our extensive experimental evaluation demonstrates that $\\texttt{GLANCE}$ consistently shows greater robustness and performance compared to existing methods across various datasets and models."
host: arxiv.org
favicon: https://arxiv.org/static/browse/0.3.4/images/icons/favicon-32x32.png
image: https://arxiv.org/static/browse/0.3.4/images/arxiv-logo-fb.png
```

---

### **Ideas Principales**

El paper introduce **GLANCE**, un algoritmo diseñado para generar **Explicaciones Contrafactuales Globales (GCE)**.

- **De lo local a lo global:** Mientras que las explicaciones tradicionales son "locales" (explican un solo caso), GLANCE busca un conjunto pequeño de acciones que funcionen para **grandes grupos de la población**.
- **Optimización Multiobjetivo:** El sistema busca equilibrar tres metas críticas:
    1. **Tamaño pequeño:** Pocas acciones para que sean fáciles de entender (interpretabilidad).
    2. **Bajo costo:** Que los cambios sugeridos sean fáciles de realizar para el usuario.
    3. **Alta efectividad:** Que las acciones ayuden a la mayor cantidad de personas posible a cambiar un resultado negativo.

### **Problemas que aborda**

- **Sobrecarga de información:** Proporcionar explicaciones individuales para cada usuario es ineficiente y no permite entender el comportamiento general del modelo.
- **Limitaciones de métodos previos:** Otros algoritmos, como GLOBE-CE, a menudo sugieren "direcciones" en lugar de acciones concretas, lo que genera incertidumbre sobre cuánto cambio es necesario y produce demasiadas "micro-acciones" que dificultan la interpretación.
- **Complejidad computacional:** El problema de encontrar el conjunto óptimo de acciones globales es **NP-duro**, lo que requiere algoritmos inteligentes y eficientes.

### **¿Qué hicieron los autores? (Metodología)**

Diseñaron un algoritmo que opera en dos fases principales mediante un enfoque de **agrupamiento (clustering) conjunto**:

1. **Generación de Acciones Diversas:** Dividen a la población en grupos (clusters) según sus características y generan múltiples opciones de cambio para cada centro de grupo.
2. **Fusión de Grupos (Agglomerative Approach):** Combinan grupos basándose en la similitud tanto en sus **características** como en las **acciones** necesarias para mejorar su situación. Esto permite agrupar a personas que, aunque sean diferentes, pueden beneficiarse de la misma acción de bajo costo.

### **Logros y Resultados**

- **Dominancia de Pareto:** GLANCE superó a sus competidores (como AReS, CET y GLOBE-CE) en el **57% de los casos**, logrando mayor efectividad con menor costo.
- **Practicidad:** A diferencia de otros métodos, GLANCE nunca produjo soluciones "imprácticas" (con menos del 80% de efectividad) en sus pruebas.
- **Robustez:** Demostró ser altamente estable, manteniendo resultados consistentes a través de diferentes modelos (redes neuronales, regresión logística, XGBoost) y conjuntos de datos.
- **Validación con usuarios:** Un estudio con 55 participantes confirmó que los humanos prefieren conjuntos de acciones **pequeños y estables**, incluso si otros métodos ofrecen costos ligeramente menores pero con mayor complejidad.

En resumen, GLANCE logra destilar el complejo comportamiento de un modelo de IA en unas pocas **acciones representativas** que son útiles, baratas y altamente efectivas para la mayoría de los usuarios afectados.


---

- I will be interesting to compare Dice vs GLANCE.
