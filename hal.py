"""
hal.py — Hardware Abstraction Layer (Patrón Adapter).

Desacopla la lógica de negocio del hardware físico.
La interfaz genérica `trigger(motivo)` permite intercambiar implementaciones
sin modificar los workers.

Fase 1 (actual): ActuadorMock → sonido Windows + consola
Fase 2 (futuro): ArduinoActuador → señal UART via pyserial

El sonido se maneja dentro del HAL para que en Fase 2, el Arduino
reproduzca su propio sonido al recibir la señal UART (b'1').
"""

import logging
from abc import ABC, abstractmethod
from config import (
    ARDUINO_PORT, ARDUINO_BAUDRATE,
    ALERTA_FRECUENCIA_HZ, ALERTA_DURACION_MS
)

logger = logging.getLogger(__name__)


class AbstractActuador(ABC):
    """Interfaz base para actuadores del sistema."""

    @abstractmethod
    def trigger(self, motivo: str) -> None:
        """
        Dispara una alerta/acción en el actuador.

        Args:
            motivo: Descripción de la causa de la alerta.
        """
        ...

    def cleanup(self) -> None:
        """Libera recursos del actuador. Override en subclases si es necesario."""
        pass


class ActuadorMock(AbstractActuador):
    """
    Fase 1: Actuador simulado.

    Reproduce un beep en Windows y loguea en consola.
    En Fase 2, el Arduino manejará su propio buzzer al recibir la señal,
    por lo que el sonido se elimina de aquí y se mueve al firmware.
    """

    def trigger(self, motivo: str) -> None:
        logger.warning("[ACTUADOR MOCK] ALERTA: %s", motivo)
        print(f"\n{'='*50}")
        print(f"  [!] ALERTA ACTUADOR: {motivo}")
        print(f"{'='*50}\n")
        self._reproducir_sonido()

    @staticmethod
    def _reproducir_sonido() -> None:
        """Beep via winsound (solo Windows). Falla silenciosamente en otros OS."""
        try:
            import winsound
            winsound.Beep(ALERTA_FRECUENCIA_HZ, ALERTA_DURACION_MS)
        except ImportError:
            # No estamos en Windows; intentar alternativa
            logger.debug("winsound no disponible (no Windows)")
        except Exception as e:
            logger.debug(f"Error reproduciendo sonido: {e}")


class ArduinoActuador(AbstractActuador):
    """
    Fase 2: Actuador real via Arduino UART.

    Envía b'1' por serial para activar el actuador físico.
    El Arduino debe tener firmware que:
    1. Lea el byte por Serial
    2. Active buzzer/LED/solenoide según corresponda
    3. Responda con ACK (opcional)
    """

    def __init__(self, port: str = None, baudrate: int = None):
        self._port = port or ARDUINO_PORT
        self._baudrate = baudrate or ARDUINO_BAUDRATE
        self._serial = None
        self._conectar()

    def _conectar(self) -> None:
        """Establece conexión serial con el Arduino."""
        try:
            import serial
            self._serial = serial.Serial(
                self._port,
                self._baudrate,
                timeout=1
            )
            logger.info(
                f"[ARDUINO] Conectado en {self._port} @ {self._baudrate} baud"
            )
        except ImportError:
            raise RuntimeError(
                "pyserial no instalado. Ejecutar: pip install pyserial"
            )
        except Exception as e:
            raise RuntimeError(
                f"No se pudo conectar al Arduino en {self._port}: {e}"
            )

    def trigger(self, motivo: str) -> None:
        if self._serial and self._serial.is_open:
            self._serial.write(b'1')
            logger.warning(f"[ARDUINO] ⚠️  Señal enviada: {motivo}")
        else:
            logger.error("[ARDUINO] Puerto serial no disponible")

    def cleanup(self) -> None:
        """Cierra la conexión serial."""
        if self._serial and self._serial.is_open:
            self._serial.close()
            logger.info("[ARDUINO] Conexión serial cerrada")


# ─────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────

def crear_actuador(modo: str = None) -> AbstractActuador:
    """
    Factory function para crear el actuador apropiado.

    Args:
        modo: "mock" o "arduino". Si None, usa ACTUADOR_MODO de config.

    Returns:
        Instancia de AbstractActuador.
    """
    from config import ACTUADOR_MODO

    modo = modo or ACTUADOR_MODO

    if modo == "arduino":
        return ArduinoActuador()
    elif modo == "mock":
        return ActuadorMock()
    else:
        logger.warning(f"Modo '{modo}' desconocido, usando mock")
        return ActuadorMock()
