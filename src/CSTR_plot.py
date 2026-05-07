import pandas as pd
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_signals(cstr, mask=None, title='CSTR Variables', dropdir=None, block=False):
    """Plot multichannel signal using Plotly."""
    
    datafn = cstr.datafn
    print(f'Opening input data file: {datafn}')
    
    # Carrega os dados gerados
    df_dataset = pd.read_csv(datafn, sep=";")
    
    # Seleciona apenas as colunas numéricas
    df_num = df_dataset.select_dtypes(include=['number'])
    colunas = df_num.columns
    
    # Aplica a máscara se houver (para plotar apenas variáveis específicas)
    if mask is not None:
        colunas = [colunas[i-1] for i in mask if (i-1) < len(colunas)]
        df_num = df_num[colunas]

    num_variaveis = len(colunas)
    
    # Cria os subplots empilhados
    fig = make_subplots(
        rows=num_variaveis, 
        cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.005
    )

    # Adiciona cada variável
    for i, col in enumerate(colunas):
        fig.add_trace(go.Scatter(
            x=df_num.index, 
            y=df_num[col], 
            mode='lines', 
            name=col
        ), row=i+1, col=1)
        
        fig.update_yaxes(title_text=col, row=i+1, col=1)

    # Layout final
    fig.update_layout(
        height=120 * num_variaveis, 
        title=title,
        hovermode='x unified',
        showlegend=False
    )

    fig.show()

def plotscatter(cstr, feat1, feat2, feat3=None,
                standardize=False,
                title='CSTR: Condition in Feature Space'):
    """
    Gera um gráfico de dispersão (scatter plot) interativo 2D ou 3D das variáveis do CSTR.
    """
    datafn = cstr.datafn
    print(f'Opening input data file: {datafn}')
    
    # 1. Carrega os dados diretamente pelo Pandas
    df = pd.read_csv(datafn, sep=";")
    
    # Remove espaços em branco do nome das colunas (para evitar erros ao chamar a coluna CLASS)
    df.columns = df.columns.str.strip()
    
    # Garante que as features existem no DataFrame. Subtrai 1 se seus parâmetros (feat1, etc) usarem base 1.
    colunas_numericas = df.select_dtypes(include=['number']).columns
    var1 = colunas_numericas[feat1 - 1]
    var2 = colunas_numericas[feat2 - 1]
    
    colunas_filtro = [var1, var2]
    var3 = None
    if feat3 is not None:
        var3 = colunas_numericas[feat3 - 1]
        colunas_filtro.append(var3)

    # 2. Padronização (Opcional)
    if standardize:
        print('Standardizing data (Z-score)...')
        scaler = StandardScaler()
        # Aplica padronização apenas nas colunas numéricas selecionadas
        df[colunas_filtro] = scaler.fit_transform(df[colunas_filtro])
        
        # Atualiza os nomes das variáveis nos eixos para indicar a padronização
        var1_label = f"{var1} (Standardized)"
        var2_label = f"{var2} (Standardized)"
        var3_label = f"{var3} (Standardized)" if var3 else None
        
        df = df.rename(columns={var1: var1_label, var2: var2_label})
        if var3:
            df = df.rename(columns={var3: var3_label})
            
        var1, var2, var3 = var1_label, var2_label, var3_label

    # Adiciona a coluna de tempo/amostra para visualizar no hover
    df['Time_Index'] = df.index

    # 3. Geração do Gráfico Plotly
    # O parâmetro 'color' agrupa automaticamente os pontos pela classe de falha
    if feat3 is None:
        # Plot 2D
        fig = px.scatter(
            df, 
            x=var1, 
            y=var2, 
            color='CLASS', # Colore de acordo com a falha (normal, S2, 2+3, etc)
            hover_data=['Time_Index'], # Mostra a iteração do tempo no mouse
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set1 # Paleta de cores nítida
        )
    else:
        # Plot 3D
        fig = px.scatter_3d(
            df, 
            x=var1, 
            y=var2, 
            z=var3, 
            color='CLASS', 
            hover_data=['Time_Index'],
            title=title,
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        
        # Ajusta as proporções e remove fundo do gráfico 3D para ficar mais limpo
        fig.update_layout(
            scene=dict(
                xaxis_title=var1,
                yaxis_title=var2,
                zaxis_title=var3,
                aspectmode='cube'
            )
        )

    # Melhorias gerais no layout
    fig.update_layout(
        legend_title_text='Condição do Reator',
        height=700, # Gráfico maior para facilitar a visualização da dispersão
        hovermode='closest'
    )

    fig.show()
