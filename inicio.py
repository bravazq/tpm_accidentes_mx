import streamlit as st

st.set_page_config(
    page_title="Accidentes viales MX",
    page_icon="👋",
)

st.write("# Análisis de accidentes de tráfico y seguridad vial en México 🇲🇽")

st.markdown(
    """
    Proyecto TPM desarrollado por el alumno Brandon Vázquez Hernández de la Maestría en Análisis y Visuzalición
    de Datos Masivos que imparte la UNIR.
    
    ### Resumen
    Mediante análisis predictivo de datos abiertos de accidentes viales urbanos/suburbanos en México,
    este estudio identifica zonas/horarios de alto riesgo y factores determinantes 
    (infraestructura, tipo de vehículo, etc.). Se implementa el uso de Google Cloud Storage para
    alojar los archivos de datos obtenidos de: Instituto Nacional de Estadística y Geografía (INEGI), así como los modelado de clasificación (gravedad de
    accidentes), series de tiempo (frecuencias) y clusterización, integrando resultados en una aplicación web
    interactiva con mapas de calor y pronósticos.
    El objetivo es generar conciencia ciudadana y proveer evidencia para políticas preventivas
    que reduzcan la siniestralidad.
    
    ### Introducción
    La seguridad vial representa un desafío para México. Según el Instituto Nacional de Estadística
    y Geografía (INEGI), en 2022 ocurrieron 362,586 accidentes de tránsito en zonas urbanas, con 4,085
    muertes y 45,758 heridos graves. Estos eventos no solo implican tragedias humanas, sino costos
    económicos equivalentes al 1.7% del PIB nacional. Pese a esfuerzos institucionales, persisten
    vacíos analíticos: falta de identificación sistemática de patrones predictivos y escasez de
    herramientas accesibles para prevención proactiva.
    
    Los estudios existentes se concentran en:
    - Análisis descriptivos ex-post (INEGI. SESNSP).
    - Modelos estadísticos no integrados con variables contextuales (meteorología, diseño vial y tipo de vehículo).
    - Visualizaciones estáticas sin capacidad interactiva para la toma de decisiones.

    Esto dificulta la priorización eficiente de recursos en infraestructura y la generación de alertas 
    preventivas basadas en riesgo dinámico.
    Este trabajo aborda estas limitaciones mediante un análisis multidimensional de datos masivos que 
    integra:
    
    - Fuentes de datos:
        * APIs gubernamentales y datos abiertos.
    - Procesamiento:
        * Pipeline ETL con geocodificación.
    - Modelado predictivo:
        * Clasificación (gravedad de accidentes).
        * Series temporales (tendencias horarias).
        * Clusterización.
    - Difusión:
        * Dashboard interactivo con capacidad de pronóstico de riesgo y simulador de intervenciones.

    
    
    ### Contribución del proyecto
    Con este proyecto se contribuye a la implementación integrada de Machine Learning explicable (XIA)
    para diagnóstico de factores de siniestralidad vial en el contexto mexicano. El impacto que se
    desea alcanzar con este estudio es a la ciudadanía (planificación de rutas seguras), autoridades
    (optimización de operativos y mantenimiento vial) y legisladores (evidencia para reformas de
    movilidad).
"""
)