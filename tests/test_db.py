"""
tests/test_db.py — Tests unitarios para la capa de persistencia.

Verifica:
- Creación de tablas
- WAL mode activo
- CRUD de eventos (tabla unificada)
- Heartbeat upsert
- Estadísticas agregadas
- Concurrencia (2 threads escribiendo)
"""

import os
import sys
import sqlite3
import threading
import tempfile
import unittest

# Agregar raíz del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import db


class TestDBSetup(unittest.TestCase):
    """Tests de configuración de la base de datos."""

    def setUp(self):
        """Usar DB temporal para cada test."""
        self._original_db_path = config.DB_PATH
        self._temp_dir = tempfile.mkdtemp()
        config.DB_PATH = os.path.join(self._temp_dir, "test_vision.db")

    def tearDown(self):
        """Restaurar path original y limpiar."""
        try:
            if os.path.exists(config.DB_PATH):
                os.remove(config.DB_PATH)
            # WAL y SHM files
            for ext in ['-wal', '-shm']:
                p = config.DB_PATH + ext
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(self._temp_dir)
        except OSError:
            pass
        config.DB_PATH = self._original_db_path

    def test_init_db_creates_tables(self):
        """init_db() debe crear las tablas eventos y estado_sistema."""
        db.init_db()
        conn = db.get_connection()
        try:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            table_names = {t['name'] for t in tables}
            self.assertIn('eventos', table_names)
            self.assertIn('estado_sistema', table_names)
        finally:
            conn.close()

    def test_init_db_is_idempotent(self):
        """init_db() puede llamarse múltiples veces sin error."""
        db.init_db()
        db.init_db()  # No debe lanzar excepción
        db.init_db()

    def test_wal_mode_active(self):
        """La conexión debe usar WAL journal mode."""
        db.init_db()
        conn = db.get_connection()
        try:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(mode.lower(), 'wal')
        finally:
            conn.close()

    def test_busy_timeout_set(self):
        """busy_timeout debe estar configurado."""
        db.init_db()
        conn = db.get_connection()
        try:
            timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            self.assertEqual(timeout, 5000)
        finally:
            conn.close()


class TestEventosCRUD(unittest.TestCase):
    """Tests de operaciones CRUD sobre la tabla unificada de eventos."""

    def setUp(self):
        self._original_db_path = config.DB_PATH
        self._temp_dir = tempfile.mkdtemp()
        config.DB_PATH = os.path.join(self._temp_dir, "test_vision.db")
        db.init_db()

    def tearDown(self):
        try:
            if os.path.exists(config.DB_PATH):
                os.remove(config.DB_PATH)
            for ext in ['-wal', '-shm']:
                p = config.DB_PATH + ext
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(self._temp_dir)
        except OSError:
            pass
        config.DB_PATH = self._original_db_path

    def test_insertar_evento_calidad_ok(self):
        """Insertar evento de calidad OK."""
        db.insertar_evento_calidad('OK', area=15000.0, detalle='Posición correcta')
        eventos = db.obtener_ultimos_eventos('calidad')
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0]['resultado'], 'OK')
        self.assertEqual(eventos[0]['tipo_evento'], 'calidad')
        self.assertAlmostEqual(eventos[0]['valor_numerico'], 15000.0)

    def test_insertar_evento_calidad_defecto(self):
        """Insertar evento de calidad DEFECTO."""
        db.insertar_evento_calidad('DEFECTO', area=3000.0, detalle='Bandeja descentrada')
        eventos = db.obtener_ultimos_eventos('calidad')
        self.assertEqual(eventos[0]['resultado'], 'DEFECTO')

    def test_insertar_evento_postura_alerta(self):
        """Insertar evento de postura con alerta."""
        db.insertar_evento_postura(
            alerta=True,
            angulo_espalda=25.3,
            angulo_cuello=35.1,
            detalle='Espalda inclinada'
        )
        eventos = db.obtener_ultimos_eventos('postura')
        self.assertEqual(len(eventos), 1)
        self.assertEqual(eventos[0]['resultado'], 'ALERTA')
        self.assertAlmostEqual(eventos[0]['valor_numerico'], 25.3)
        self.assertAlmostEqual(eventos[0]['valor_numerico2'], 35.1)

    def test_insertar_evento_postura_ok(self):
        """Insertar evento de postura sin alerta."""
        db.insertar_evento_postura(alerta=False, angulo_espalda=10.0, angulo_cuello=15.0)
        eventos = db.obtener_ultimos_eventos('postura')
        self.assertEqual(eventos[0]['resultado'], 'OK')

    def test_obtener_ultimos_eventos_limit(self):
        """El limit de eventos funciona correctamente."""
        for i in range(20):
            db.insertar_evento_calidad('OK', area=float(i))
        eventos = db.obtener_ultimos_eventos('calidad', limit=5)
        self.assertEqual(len(eventos), 5)

    def test_obtener_ultimos_eventos_sin_filtro(self):
        """Obtener todos los tipos de evento sin filtro."""
        db.insertar_evento_calidad('OK', area=100.0)
        db.insertar_evento_postura(alerta=False, angulo_espalda=10.0)
        eventos = db.obtener_ultimos_eventos()
        self.assertEqual(len(eventos), 2)

    def test_obtener_ultimos_eventos_orden_desc(self):
        """Los eventos se devuelven en orden descendente (más recientes primero)."""
        db.insertar_evento_calidad('OK', area=1.0, detalle='primero')
        db.insertar_evento_calidad('DEFECTO', area=2.0, detalle='segundo')
        eventos = db.obtener_ultimos_eventos('calidad')
        self.assertEqual(eventos[0]['detalle'], 'segundo')
        self.assertEqual(eventos[1]['detalle'], 'primero')


class TestHeartbeat(unittest.TestCase):
    """Tests del sistema de heartbeat."""

    def setUp(self):
        self._original_db_path = config.DB_PATH
        self._temp_dir = tempfile.mkdtemp()
        config.DB_PATH = os.path.join(self._temp_dir, "test_vision.db")
        db.init_db()

    def tearDown(self):
        try:
            if os.path.exists(config.DB_PATH):
                os.remove(config.DB_PATH)
            for ext in ['-wal', '-shm']:
                p = config.DB_PATH + ext
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(self._temp_dir)
        except OSError:
            pass
        config.DB_PATH = self._original_db_path

    def test_heartbeat_insert(self):
        """Primer heartbeat crea registro."""
        db.actualizar_heartbeat("worker_test")
        workers = db.obtener_estado_workers()
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0]['worker_name'], 'worker_test')
        self.assertEqual(workers[0]['status'], 'activo')

    def test_heartbeat_upsert(self):
        """Heartbeats sucesivos actualizan el timestamp sin duplicar."""
        db.actualizar_heartbeat("worker_test")
        db.actualizar_heartbeat("worker_test")
        db.actualizar_heartbeat("worker_test")
        workers = db.obtener_estado_workers()
        self.assertEqual(len(workers), 1)  # Solo 1 registro

    def test_multiple_workers(self):
        """Múltiples workers se registran independientemente."""
        db.actualizar_heartbeat("worker_bandejas")
        db.actualizar_heartbeat("worker_postura")
        workers = db.obtener_estado_workers()
        self.assertEqual(len(workers), 2)


class TestEstadisticas(unittest.TestCase):
    """Tests de estadísticas agregadas."""

    def setUp(self):
        self._original_db_path = config.DB_PATH
        self._temp_dir = tempfile.mkdtemp()
        config.DB_PATH = os.path.join(self._temp_dir, "test_vision.db")
        db.init_db()

    def tearDown(self):
        try:
            if os.path.exists(config.DB_PATH):
                os.remove(config.DB_PATH)
            for ext in ['-wal', '-shm']:
                p = config.DB_PATH + ext
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(self._temp_dir)
        except OSError:
            pass
        config.DB_PATH = self._original_db_path

    def test_estadisticas_vacias(self):
        """Sin datos, las estadísticas deben ser cero."""
        stats = db.obtener_estadisticas()
        self.assertEqual(stats['calidad_total'], 0)
        self.assertEqual(stats['calidad_ok'], 0)
        self.assertEqual(stats['calidad_defecto'], 0)
        self.assertEqual(stats['calidad_pct_merma'], 0.0)
        self.assertEqual(stats['postura_total'], 0)
        self.assertEqual(stats['postura_alertas'], 0)

    def test_estadisticas_calidad(self):
        """Estadísticas de calidad se calculan correctamente."""
        for _ in range(8):
            db.insertar_evento_calidad('OK', area=10000.0)
        for _ in range(2):
            db.insertar_evento_calidad('DEFECTO', area=3000.0)

        stats = db.obtener_estadisticas()
        self.assertEqual(stats['calidad_total'], 10)
        self.assertEqual(stats['calidad_ok'], 8)
        self.assertEqual(stats['calidad_defecto'], 2)
        self.assertAlmostEqual(stats['calidad_pct_merma'], 20.0)

    def test_estadisticas_postura(self):
        """Estadísticas de postura se calculan correctamente."""
        for _ in range(5):
            db.insertar_evento_postura(alerta=False, angulo_espalda=10.0)
        for _ in range(3):
            db.insertar_evento_postura(alerta=True, angulo_espalda=25.0)

        stats = db.obtener_estadisticas()
        self.assertEqual(stats['postura_total'], 8)
        self.assertEqual(stats['postura_alertas'], 3)


class TestConcurrencia(unittest.TestCase):
    """Tests de concurrencia con WAL mode."""

    def setUp(self):
        self._original_db_path = config.DB_PATH
        self._temp_dir = tempfile.mkdtemp()
        config.DB_PATH = os.path.join(self._temp_dir, "test_vision.db")
        db.init_db()

    def tearDown(self):
        try:
            if os.path.exists(config.DB_PATH):
                os.remove(config.DB_PATH)
            for ext in ['-wal', '-shm']:
                p = config.DB_PATH + ext
                if os.path.exists(p):
                    os.remove(p)
            os.rmdir(self._temp_dir)
        except OSError:
            pass
        config.DB_PATH = self._original_db_path

    def test_concurrent_writes(self):
        """2 threads escribiendo simultáneamente no deben causar 'database locked'."""
        errors = []

        def writer_calidad():
            try:
                for i in range(50):
                    db.insertar_evento_calidad('OK', area=float(i))
            except Exception as e:
                errors.append(f"calidad: {e}")

        def writer_postura():
            try:
                for i in range(50):
                    db.insertar_evento_postura(
                        alerta=(i % 5 == 0),
                        angulo_espalda=float(i)
                    )
            except Exception as e:
                errors.append(f"postura: {e}")

        t1 = threading.Thread(target=writer_calidad)
        t2 = threading.Thread(target=writer_postura)

        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [], f"Errores de concurrencia: {errors}")

        # Verificar que todos los registros se insertaron
        stats = db.obtener_estadisticas()
        self.assertEqual(stats['calidad_total'], 50)
        self.assertEqual(stats['postura_total'], 50)


if __name__ == '__main__':
    unittest.main()
