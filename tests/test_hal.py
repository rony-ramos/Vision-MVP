"""
tests/test_hal.py — Tests unitarios para el Hardware Abstraction Layer.

Verifica:
- ActuadorMock funciona sin excepciones
- Factory crear_actuador() devuelve la clase correcta
- Interfaz AbstractActuador no se puede instanciar directamente
- Cleanup no falla en mock
"""

import os
import sys
import unittest

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hal import AbstractActuador, ActuadorMock, crear_actuador


class TestAbstractActuador(unittest.TestCase):
    """Tests de la interfaz abstracta."""

    def test_cannot_instantiate_abstract(self):
        """No se puede instanciar AbstractActuador directamente."""
        with self.assertRaises(TypeError):
            AbstractActuador()


class TestActuadorMock(unittest.TestCase):
    """Tests del actuador simulado."""

    def test_trigger_no_exception(self):
        """trigger() no debe lanzar excepciones."""
        actuador = ActuadorMock()
        try:
            actuador.trigger("Test de alerta")
        except Exception as e:
            self.fail(f"trigger() lanzó excepción: {e}")

    def test_trigger_with_special_chars(self):
        """trigger() maneja caracteres especiales sin error."""
        actuador = ActuadorMock()
        actuador.trigger("Alerta: angulo=25.3 > 20 max")
        actuador.trigger("")
        actuador.trigger("DEFECTO critico detectado")

    def test_cleanup_no_exception(self):
        """cleanup() no debe lanzar excepciones."""
        actuador = ActuadorMock()
        try:
            actuador.cleanup()
        except Exception as e:
            self.fail(f"cleanup() lanzó excepción: {e}")

    def test_is_abstract_actuador(self):
        """ActuadorMock es una instancia de AbstractActuador."""
        actuador = ActuadorMock()
        self.assertIsInstance(actuador, AbstractActuador)


class TestFactory(unittest.TestCase):
    """Tests de la factory function."""

    def test_crear_mock(self):
        """crear_actuador('mock') devuelve ActuadorMock."""
        actuador = crear_actuador("mock")
        self.assertIsInstance(actuador, ActuadorMock)

    def test_crear_default_is_mock(self):
        """crear_actuador() sin argumentos devuelve mock (default de config)."""
        actuador = crear_actuador()
        self.assertIsInstance(actuador, ActuadorMock)

    def test_crear_unknown_falls_back_to_mock(self):
        """Modo desconocido cae a mock con warning."""
        actuador = crear_actuador("desconocido")
        self.assertIsInstance(actuador, ActuadorMock)

    def test_crear_arduino_requires_serial(self):
        """crear_actuador('arduino') falla si no hay puerto serial."""
        # En entorno de test, no hay Arduino conectado
        with self.assertRaises(RuntimeError):
            crear_actuador("arduino")


if __name__ == '__main__':
    unittest.main()
