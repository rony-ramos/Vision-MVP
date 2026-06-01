"""
dashboard.py — HMI: Dashboard de monitoreo en tiempo real.

Interfaz Streamlit con modelo Pull/Polling via st_autorefresh.
Cero estado en el frontend — puro visor de SQLite.

Ejecución: streamlit run dashboard.py
"""

import datetime
import streamlit as st
from streamlit_autorefresh import st_autorefresh

import config
import db


# ─────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Vision-MVP · Monitor de Producción",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-refresh (polling cada 2 segundos)
st_autorefresh(interval=config.DASHBOARD_REFRESH_MS, key="data_refresh")

# Inicializar DB (idempotente)
db.init_db()


# ─────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Header */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1E88E5;
        margin-bottom: 1.5rem;
    }

    /* KPI cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Status indicators */
    .status-active {
        color: #4CAF50;
        font-weight: bold;
    }
    .status-inactive {
        color: #f44336;
        font-weight: bold;
    }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #90CAF9;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    /* Alert banner */
    .alert-banner {
        background: linear-gradient(135deg, #b71c1c 0%, #e53935 100%);
        color: white;
        padding: 0.8rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1rem;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<div class="main-header">🏭 VISION-MVP · Monitor de Producción</div>',
            unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Estado del Sistema (Workers)
# ─────────────────────────────────────────────
workers = db.obtener_estado_workers()
stats = db.obtener_estadisticas()

# Banner de alerta si algún worker está caído
now = datetime.datetime.now()
workers_caidos = []
for w in workers:
    try:
        last_hb = datetime.datetime.fromisoformat(w['last_heartbeat'])
        if (now - last_hb).total_seconds() > config.HEARTBEAT_TIMEOUT_S:
            workers_caidos.append(w['worker_name'])
    except (ValueError, TypeError):
        workers_caidos.append(w['worker_name'])

if workers_caidos:
    st.markdown(
        f'<div class="alert-banner">⚠️ WORKERS SIN RESPUESTA: '
        f'{", ".join(workers_caidos)}</div>',
        unsafe_allow_html=True
    )

# Layout principal
col_kpi, col_status = st.columns([3, 1])

with col_kpi:
    st.markdown('<div class="section-header">📊 KPIs de Producción</div>',
                unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.metric(
            label="✅ Bandejas OK",
            value=stats.get('calidad_ok', 0)
        )
    with k2:
        st.metric(
            label="❌ Bandejas Defectuosas",
            value=stats.get('calidad_defecto', 0)
        )
    with k3:
        merma = stats.get('calidad_pct_merma', 0)
        st.metric(
            label="📉 % Merma",
            value=f"{merma}%",
            delta=f"{merma - 8:.1f}% vs objetivo" if merma > 0 else None,
            delta_color="inverse"  # Negativo es bueno para merma
        )
    with k4:
        st.metric(
            label="🧍 Alertas Ergonómicas",
            value=stats.get('postura_alertas', 0)
        )

with col_status:
    st.markdown('<div class="section-header">⚙️ Estado del Sistema</div>',
                unsafe_allow_html=True)

    expected_workers = ['worker_bandejas', 'worker_postura']

    for wname in expected_workers:
        worker_info = next((w for w in workers if w['worker_name'] == wname), None)

        if worker_info:
            try:
                last_hb = datetime.datetime.fromisoformat(
                    worker_info['last_heartbeat']
                )
                delta = (now - last_hb).total_seconds()
                is_active = delta < config.HEARTBEAT_TIMEOUT_S

                icon = "🟢" if is_active else "🔴"
                time_str = f"{delta:.0f}s ago"
                css_class = "status-active" if is_active else "status-inactive"
            except (ValueError, TypeError):
                icon = "🔴"
                time_str = "Error"
                css_class = "status-inactive"
        else:
            icon = "⚪"
            time_str = "No registrado"
            css_class = "status-inactive"

        st.markdown(
            f'{icon} <span class="{css_class}">{wname}</span> '
            f'<small>({time_str})</small>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────
# Tablas de Eventos Recientes
# ─────────────────────────────────────────────
st.markdown("---")

col_calidad, col_postura = st.columns(2)

with col_calidad:
    st.markdown(
        '<div class="section-header">📦 Eventos de Calidad (Bandejas)</div>',
        unsafe_allow_html=True
    )

    eventos_calidad = db.obtener_ultimos_eventos(
        tipo_evento='calidad',
        limit=config.DASHBOARD_EVENTOS_LIMIT
    )

    if eventos_calidad:
        # Formatear para display
        display_data = []
        for e in eventos_calidad:
            display_data.append({
                'Hora': e['timestamp'].split(' ')[-1] if ' ' in (e['timestamp'] or '') else e['timestamp'],
                'Resultado': '✅ OK' if e['resultado'] == 'OK' else '❌ DEFECTO',
                'Área (px²)': f"{e['valor_numerico']:.0f}" if e['valor_numerico'] else '-',
                'Detalle': e['detalle'] or '-'
            })
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.info("No hay eventos de calidad registrados aún.")

with col_postura:
    st.markdown(
        '<div class="section-header">🧍 Eventos de Postura (Ergonomía)</div>',
        unsafe_allow_html=True
    )

    eventos_postura = db.obtener_ultimos_eventos(
        tipo_evento='postura',
        limit=config.DASHBOARD_EVENTOS_LIMIT
    )

    if eventos_postura:
        display_data = []
        for e in eventos_postura:
            display_data.append({
                'Hora': e['timestamp'].split(' ')[-1] if ' ' in (e['timestamp'] or '') else e['timestamp'],
                'Estado': '⚠️ ALERTA' if e['resultado'] == 'ALERTA' else '✅ OK',
                'Espalda (°)': f"{e['valor_numerico']:.1f}" if e['valor_numerico'] else '-',
                'Cuello (°)': f"{e['valor_numerico2']:.1f}" if e['valor_numerico2'] else '-',
                'Detalle': e['detalle'] or '-'
            })
        st.dataframe(
            display_data,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.info("No hay eventos de postura registrados aún.")


# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f"<small>🔄 Auto-refresh: {config.DASHBOARD_REFRESH_MS/1000:.0f}s "
    f"| Última actualización: {now.strftime('%H:%M:%S')} "
    f"| Total inspecciones: {stats.get('calidad_total', 0)}</small>",
    unsafe_allow_html=True
)
