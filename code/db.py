"""
db.py — Capa de persistencia SQLite con WAL mode.

Tabla unificada `eventos` con columna `tipo_evento` para calidad y postura.
Tabla `estado_sistema` para heartbeat de workers.

CRITICAL: Toda conexión DEBE usar get_connection() que activa WAL y busy_timeout.
Esto permite 2 writers (workers) y 1 reader (dashboard) sin 'database locked'.
"""

import sqlite3
import datetime
import config


def get_connection() -> sqlite3.Connection:
    """
    Crea una conexión a SQLite con WAL mode activado.

    WAL (Write-Ahead Logging) permite lecturas concurrentes con escrituras,
    eliminando el error 'database locked' en escenarios multi-proceso.
    """
    conn = sqlite3.connect(config.DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Crea las tablas si no existen. Idempotente."""
    conn = get_connection()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS eventos (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL DEFAULT (datetime('now', 'localtime')),
                tipo_evento     TEXT    NOT NULL CHECK(tipo_evento IN ('calidad', 'postura')),
                resultado       TEXT    NOT NULL,
                valor_numerico  REAL,
                valor_numerico2 REAL,
                detalle         TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_eventos_tipo
                ON eventos(tipo_evento, timestamp DESC);

            CREATE TABLE IF NOT EXISTS estado_sistema (
                worker_name     TEXT PRIMARY KEY,
                last_heartbeat  TEXT NOT NULL,
                status          TEXT NOT NULL DEFAULT 'activo'
            );

            CREATE TABLE IF NOT EXISTS configuracion (
                clave TEXT PRIMARY KEY,
                valor TEXT NOT NULL
            );

            -- Valores por defecto basados en config.py si no existen
            INSERT OR IGNORE INTO configuracion (clave, valor) VALUES ('cam_bandejas_index', '?');
            INSERT OR IGNORE INTO configuracion (clave, valor) VALUES ('cam_postura_index', '?');
        """)
        # Reemplazamos los parámetros de INSERT OR IGNORE
        conn.execute(
            "UPDATE configuracion SET valor = ? WHERE clave = 'cam_bandejas_index' AND valor = '?'",
            (str(config.CAM_BANDEJAS_INDEX),)
        )
        conn.execute(
            "UPDATE configuracion SET valor = ? WHERE clave = 'cam_postura_index' AND valor = '?'",
            (str(config.CAM_POSTURA_INDEX),)
        )
        conn.commit()
    finally:
        conn.close()


# =============================================
# Escritura (usado por workers)
# =============================================

def insertar_evento_calidad(resultado: str, area: float = None,
                            detalle: str = None) -> None:
    """
    Registra un evento de inspección de bandeja.

    Args:
        resultado: 'OK' o 'DEFECTO'
        area: Área del contorno detectado (px²)
        detalle: Descripción adicional del evento
    """
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO eventos (tipo_evento, resultado, valor_numerico, detalle)
               VALUES ('calidad', ?, ?, ?)""",
            (resultado, area, detalle)
        )
        conn.commit()
    finally:
        conn.close()


def insertar_evento_postura(alerta: bool, angulo_espalda: float = None,
                            angulo_cuello: float = None,
                            detalle: str = None) -> None:
    """
    Registra un evento de monitoreo ergonómico.

    Args:
        alerta: True si se detectó postura riesgosa
        angulo_espalda: Ángulo de inclinación de espalda (grados)
        angulo_cuello: Ángulo de flexión cervical (grados)
        detalle: Descripción adicional del evento
    """
    conn = get_connection()
    try:
        conn.execute(
            """INSERT INTO eventos (tipo_evento, resultado, valor_numerico,
               valor_numerico2, detalle)
               VALUES ('postura', ?, ?, ?, ?)""",
            ('ALERTA' if alerta else 'OK', angulo_espalda, angulo_cuello, detalle)
        )
        conn.commit()
    finally:
        conn.close()


def actualizar_heartbeat(worker_name: str) -> None:
    """Actualiza el heartbeat de un worker (upsert)."""
    conn = get_connection()
    try:
        now = datetime.datetime.now().isoformat()
        conn.execute(
            """INSERT INTO estado_sistema (worker_name, last_heartbeat, status)
               VALUES (?, ?, 'activo')
               ON CONFLICT(worker_name)
               DO UPDATE SET last_heartbeat = excluded.last_heartbeat,
                             status = 'activo'""",
            (worker_name, now)
        )
        conn.commit()
    finally:
        conn.close()


# =============================================
# Lectura (usado por dashboard)
# =============================================

def obtener_ultimos_eventos(tipo_evento: str = None, limit: int = 50) -> list:
    """
    Obtiene los últimos eventos, opcionalmente filtrados por tipo.

    Returns:
        Lista de dicts con los campos del evento.
    """
    conn = get_connection()
    try:
        if tipo_evento:
            cursor = conn.execute(
                """SELECT id, timestamp, tipo_evento, resultado,
                          valor_numerico, valor_numerico2, detalle
                   FROM eventos
                   WHERE tipo_evento = ?
                   ORDER BY id DESC LIMIT ?""",
                (tipo_evento, limit)
            )
        else:
            cursor = conn.execute(
                """SELECT id, timestamp, tipo_evento, resultado,
                          valor_numerico, valor_numerico2, detalle
                   FROM eventos
                   ORDER BY id DESC LIMIT ?""",
                (limit,)
            )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def obtener_estadisticas() -> dict:
    """
    Obtiene estadísticas agregadas del sistema.

    Returns:
        Dict con conteos de calidad (OK/DEFECTO) y postura (OK/ALERTA).
    """
    conn = get_connection()
    try:
        stats = {}

        # Estadísticas de calidad
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN resultado = 'OK' THEN 1 ELSE 0 END) as ok,
                SUM(CASE WHEN resultado = 'DEFECTO' THEN 1 ELSE 0 END) as defecto
               FROM eventos WHERE tipo_evento = 'calidad'"""
        ).fetchone()
        stats['calidad_total'] = row['total'] or 0
        stats['calidad_ok'] = row['ok'] or 0
        stats['calidad_defecto'] = row['defecto'] or 0
        stats['calidad_pct_merma'] = (
            round(stats['calidad_defecto'] / stats['calidad_total'] * 100, 1)
            if stats['calidad_total'] > 0 else 0.0
        )

        # Estadísticas de postura
        row = conn.execute(
            """SELECT
                COUNT(*) as total,
                SUM(CASE WHEN resultado = 'ALERTA' THEN 1 ELSE 0 END) as alertas
               FROM eventos WHERE tipo_evento = 'postura'"""
        ).fetchone()
        stats['postura_total'] = row['total'] or 0
        stats['postura_alertas'] = row['alertas'] or 0

        return stats
    finally:
        conn.close()


def obtener_estado_workers() -> list:
    """Obtiene el estado de todos los workers registrados."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT worker_name, last_heartbeat, status FROM estado_sistema"
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def obtener_config_camaras() -> dict:
    """
    Obtiene los índices de las cámaras desde la base de datos.
    Returns: dict con 'cam_bandejas_index' y 'cam_postura_index' (enteros).
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT clave, valor FROM configuracion WHERE clave IN ('cam_bandejas_index', 'cam_postura_index')"
        )
        filas = cursor.fetchall()
        # Valores por defecto en caso de algún error extraño
        configs = {
            'cam_bandejas_index': config.CAM_BANDEJAS_INDEX,
            'cam_postura_index': config.CAM_POSTURA_INDEX
        }
        for row in filas:
            try:
                configs[row['clave']] = int(row['valor'])
            except ValueError:
                pass
        return configs
    finally:
        conn.close()


def actualizar_config_camaras(idx_bandejas: int, idx_postura: int) -> None:
    """Actualiza los índices de las cámaras en la base de datos."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE configuracion SET valor = ? WHERE clave = 'cam_bandejas_index'",
            (str(idx_bandejas),)
        )
        conn.execute(
            "UPDATE configuracion SET valor = ? WHERE clave = 'cam_postura_index'",
            (str(idx_postura),)
        )
        conn.commit()
    finally:
        conn.close()
