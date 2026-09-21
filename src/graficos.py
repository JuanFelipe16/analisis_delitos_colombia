"""Helpers de estilo compartidos entre los notebooks de visualización."""

import matplotlib.pyplot as plt
import seaborn as sns


def aplicar_estilo():
    """Aplica el estilo visual consistente usado en todo el portafolio."""
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.autolayout"] = True


def cerrar_figura():
    """Ajusta el layout y muestra la figura actual (evita repetir 3 líneas por gráfico)."""
    plt.tight_layout()
    plt.show()
