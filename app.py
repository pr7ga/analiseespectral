import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-dark')
import numpy as np
import io
import math
from datetime import datetime
import locale

st.set_page_config(page_title="Análise Espectral com Harmônicos", layout="wide")
st.title("Análise Espectral com Detecção de Harmônicos")

uploaded_file = st.file_uploader("Arraste e solte um arquivo CSV ou clique para selecionar", type="csv")

st.sidebar.header("Parâmetros")
equipamento_usuario = st.sidebar.text_input("Nome do Equipamento", value="Equipamento")
limiar_potencia = st.sidebar.slider(
    "Limiar de Potência para Detecção (dBm)",
    min_value=-100.0,
    max_value=0.0,
    value=-30.0,
    step=0.02,
    format="%.1f"
)
atenuacao_db = st.sidebar.number_input("Atenuação aplicada na medição (dB)", value=20, step=1)
exibir_limites = st.sidebar.checkbox("Exibir linhas de limite", value=True)
exibir_pseudo = st.sidebar.checkbox("Exibir THD e Harmônico dominante", value=True)

if uploaded_file:
    linhas = uploaded_file.read().decode('utf-8').splitlines()

    metadata = {}
    for linha in linhas:
        if linha.startswith('!'):
            if 'TIMESTAMP' in linha and 'Trace' not in linha:
                metadata['Timestamp'] = linha.split('TIMESTAMP')[1].strip()
            elif 'NAME' in linha:
                metadata['Name'] = linha.split('NAME')[1].strip()
            elif 'MODEL' in linha:
                metadata['Model'] = linha.split('MODEL')[1].strip()
            elif 'SERIAL' in linha:
                metadata['Serial'] = linha.split('SERIAL')[1].strip()
            elif 'FIRMWARE_VERSION' in linha:
                metadata['Firmware'] = linha.split('FIRMWARE_VERSION')[1].strip()
        if linha.strip() == 'BEGIN':
            break

    linha_inicio = linhas.index('BEGIN') + 1
    dados = []
    for linha in linhas[linha_inicio:]:
        partes = linha.strip().split(',')
        if len(partes) >= 2:
            dados.append([partes[0], partes[1]])

    df = pd.DataFrame(dados, columns=['Frequência (Hz)', 'Potência (dBm)'])
    df['Frequência (MHz)'] = pd.to_numeric(df['Frequência (Hz)'], errors='coerce') / 1e6
    df['Potência (dBm)'] = pd.to_numeric(df['Potência (dBm)'], errors='coerce') + atenuacao_db
    df = df.dropna()

    df_filtrado = df[df['Potência (dBm)'] > limiar_potencia].reset_index(drop=True)
    idx_max = df_filtrado['Potência (dBm)'].idxmax()
    freq_fundamental = df_filtrado.loc[idx_max, 'Frequência (MHz)']
    pot_fundamental = df_filtrado.loc[idx_max, 'Potência (dBm)']

    if 145 <= freq_fundamental <= 147:
        freq_fundamental_exibida = 146
    elif 434 <= freq_fundamental <= 437:
        freq_fundamental_exibida = 435
    else:
        freq_fundamental_exibida = round(freq_fundamental)

    resultados = [{
        'Ordem': 1,
        'Frequência': freq_fundamental,
        'Frequência_exibida': freq_fundamental_exibida,
        'Potência': pot_fundamental,
        'Label': 'Fundamental'
    }]

    max_ordem = min(10, int(df['Frequência (MHz)'].max() / freq_fundamental))
    for ordem in range(2, max_ordem + 1):
        freq_teorica = ordem * freq_fundamental
        tolerancia = 0.05 * freq_teorica
        candidatos = df_filtrado[(df_filtrado['Frequência (MHz)'] >= freq_teorica - tolerancia) &
                                  (df_filtrado['Frequência (MHz)'] <= freq_teorica + tolerancia)]
        if not candidatos.empty:
            idx_max = candidatos['Potência (dBm)'].idxmax()
            resultados.append({
                'Ordem': ordem,
                'Frequência': df_filtrado.loc[idx_max, 'Frequência (MHz)'],
                'Frequência_exibida': ordem * freq_fundamental_exibida,
                'Potência': df_filtrado.loc[idx_max, 'Potência (dBm)'],
                'Label': f'{ordem}º Harmônico'
            })

    pot_fund_watt = 10 ** (pot_fundamental / 10)
    pot_harmonicos_watt = sum(10 ** (item['Potência'] / 10) for item in resultados[1:])
    thd = 100 * (pot_harmonicos_watt ** 0.5) / pot_fund_watt
    thd_frac = thd / 100
    dBc_total = 20 * np.log10(thd_frac) if thd_frac > 0 else float('-inf')

    if len(resultados) > 1:
        pot_max_harmonico = max(item['Potência'] for item in resultados[1:])
        dBc_maior = pot_max_harmonico - pot_fundamental
    else:
        dBc_maior = float('-inf')

    if exibir_pseudo:
        resultados.append({
            'Ordem': None,
            'Frequência': None,
            'Potência': None,
            'Label': f"Distorção harmônica total (THD): {thd:.2f}".replace('.', ',') + "%"
        })

        if math.isfinite(dBc_maior):
            resultados.append({
                'Ordem': None,
                'Frequência': None,
                'Potência': None,
                'Label': f"Harmônico dominante: {dBc_maior:.2f}".replace('.', ',') + " dBc"
            })

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(df['Frequência (MHz)'], df['Potência (dBm)'], color='blue')
    cores = plt.colormaps.get_cmap('tab10')

    harmonicos_visuais = [item for item in resultados if item['Ordem'] is not None]
    pseudoharmonicos = [item for item in resultados if item['Ordem'] is None]

    for idx, item in enumerate(harmonicos_visuais):
        freq_str = f"{int(round(item['Frequência_exibida']))} MHz"
        pot_str = f"{item['Potência']:.2f}".replace('.', ',') + " dBm"
        label = f"{item['Label']} ({freq_str}, {pot_str})"
        ax.scatter(item['Frequência'], item['Potência'], color=cores(idx % 10), marker='x', s=50, zorder=5, label=label)
        
    limites_visuais = []
    if exibir_limites:
        ax.axhline(-16, color='red', linestyle='--', linewidth=1)
        ax.axhline(pot_fundamental - 40, color='saddlebrown', linestyle='--', linewidth=1)

        limites_visuais = [
            plt.Line2D([], [], color='red', linestyle='--', label='Limite absoluto (-16 dBm)'),
            plt.Line2D([], [], color='saddlebrown', linestyle='--', label='Limite relativo à fundamental (-40 dBc)')
        ]

    nivel_40db = pot_fundamental - 40  # necessário para definir eixo y mesmo com limites ocultos



    ymin = min(df['Potência (dBm)'].min(), -16, nivel_40db) - 5
    ymax = df['Potência (dBm)'].max() + 5
    ax.set_ylim(ymin, ymax)

    legendas_marcadas = [plt.Line2D([0], [0], marker='x', color='w',
                                    markeredgecolor=cores(i % 10), markersize=8,
                                    label=f"{item['Label']} ({int(round(item['Frequência_exibida']))} MHz, {item['Potência']:.2f}".replace('.', ',') + " dBm)")
                         for i, item in enumerate(harmonicos_visuais)]

    linha_em_branco = plt.Line2D([], [], color='#f7f7f7', label='')
    legendas_vermelhas = [plt.Line2D([], [], color='none', label=item['Label'],
                                     marker='', linestyle='', linewidth=0)
                          for item in pseudoharmonicos]

    legenda_completa = (
        legendas_marcadas +
        limites_visuais +
        ([linha_em_branco] + legendas_vermelhas if pseudoharmonicos else [])
    )
    leg = ax.legend(handles=legenda_completa, loc='upper right', fontsize=10)

    for text in leg.get_texts():
        if 'THD' in text.get_text() or 'Harmônico dominante' in text.get_text():
            text.set_color('red')

    ax.set_title(
        f"{equipamento_usuario} - Análise Espectral de Transmissão - {freq_fundamental_exibida} MHz",
        fontsize=18  # ou 16, 18, etc.
    )
    ax.set_xlabel('Frequência (MHz)')
    ax.set_ylabel('Potência (dBm)')
    ax.grid(True, color='lightgray')

    # Formatação do timestamp para o formato brasileiro
    timestamp_raw = metadata.get('Timestamp', '')
    try:
        locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')
    except:
        try:
            locale.setlocale(locale.LC_TIME, 'en_US')
        except:
            pass
    try:
        dt_obj = datetime.strptime(timestamp_raw, "%A, %d %B %Y %H:%M:%S")
        timestamp_br = dt_obj.strftime("%d/%m/%Y %H:%M:%S")
    except:
        timestamp_br = "N/A"

    texto_metadata = (
        f"Data: {timestamp_br} | "
        f"Equipamento: {metadata.get('Name', 'N/A')} {metadata.get('Model', 'N/A')} | "
        f"Serial: {metadata.get('Serial', 'N/A')} | "
        f"Firmware: {metadata.get('Firmware', 'N/A')} | "
        f"Atenuação: {atenuacao_db} dB | "
        f"Medições feitas por PR7GA"
    )

    fig.subplots_adjust(bottom=0.12)
    # fig.patch.set_facecolor('#e5e5e5')  # fundo da figura
    ax.set_facecolor('#f7f7f7')         # fundo da área do gráfico

    fig.text(0.5, 0.045, texto_metadata, wrap=True, ha='center', fontsize=8)

    # Moldura do gráfico em cinza
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
        spine.set_linewidth(1)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=400, bbox_inches='tight')
    buffer.seek(0)
    nome_arquivo = f"{equipamento_usuario.strip().replace(' ', '_').lower()}_{freq_fundamental_exibida}_mhz.png"

    st.download_button(
        label="Baixar Gráfico em PNG (Alta Resolução)",
        data=buffer,
        file_name=nome_arquivo,
        mime="image/png"
    )
    # Exportação estruturada dos harmônicos e métricas para CSV
    linhas_csv = []

    # Harmônicos reais
    for item in resultados:
        if item['Ordem'] is not None:
            linhas_csv.append({
                "Ordem": item['Ordem'],
                "Frequência (MHz)": f"{item['Frequência_exibida']:.2f}".replace('.', ','),
                "Potência (dBm)": f"{item['Potência']:.2f}".replace('.', ','),
                "Descrição": item['Label'],
                "Tipo": "Harmônico",
                "Valor Numérico": ''
            })

    # Métricas finais: THD e Harmônico dominante
    linhas_csv.append({
        "Ordem": '',
        "Frequência (MHz)": '',
        "Potência (dBm)": '',
        "Descrição": f"Distorção harmônica total (THD)",
        "Tipo": "Métrica",
        "Valor Numérico": f"{thd:.2f}".replace('.', ',')  # percentual
    })

    linhas_csv.append({
        "Ordem": '',
        "Frequência (MHz)": '',
        "Potência (dBm)": '',
        "Descrição": f"Harmônico dominante (dBc)",
        "Tipo": "Métrica",
        "Valor Numérico": f"{dBc_maior:.2f}".replace('.', ',') if math.isfinite(dBc_maior) else '-∞'
    })

    df_harmonicos = pd.DataFrame(linhas_csv)

    csv_str = df_harmonicos.to_csv(sep=';', index=False)
    csv_str_bom = '\ufeff' + csv_str  # Adiciona BOM UTF-8
    csv_bytes = io.BytesIO(csv_str_bom.encode('utf-8'))

    st.download_button(
        label="Baixar Tabela de Harmônicos + Métricas (CSV)",
        data=csv_bytes,
        file_name=f"harmonicos_{freq_fundamental_exibida}MHz.csv",
        mime="text/csv",
        help="Baixa um arquivo CSV com harmônicos detectados e métricas estruturadas"
    )

    st.pyplot(fig)
