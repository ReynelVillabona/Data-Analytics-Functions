def missing_values_per_column(dataframe):
    nan_values_per_column = dataframe.isna().sum()
    colun_with_null = {column: conteo for column, conteo in nan_values_per_column.items() if conteo > 0}
    return colun_with_null, len(colun_with_null)


def distribution_graph_column(dataframe, column):

            data_no_nan = dataframe[dataframe[column].notna()]

            sns.set(style="whitegrid")

            plt.figure(figsize=(40, 6))
            sns.histplot(data_no_nan[column], kde=True, color='skyblue')
            plt.title(f'Distribución de {column}')
            plt.xlabel(column)
            plt.ylabel('Frecuencia')
    
            return plt.show()


### latitude and longitude in decimals are necessary to call this function, only check the 6 more closest neighbors. (You can change this)
def assign_closer_neighbor_value(dataframe,column):

    array = dataframe.loc[dataframe[column].isna()].index
    

    for indice in array:
        defined_columns = dataframe[dataframe['Sub-Region'].notna()].copy()

        defined_columns['Distancia'] = defined_columns.apply(
            lambda x: geodesic((dataframe.loc[indice]['Decimal Latitude'], dataframe.loc[indice]['Decimal Longitude']),
                            (x['Decimal Latitude'], x['Decimal Longitude'])).km,
            axis=1
        )


        indices_valor_minimo = defined_columns.nsmallest(6, 'Distancia').index
        ##print(indices_valor_minimo)

        # Iterar sobre los índices de menor a mayor distancia
        for index in indices_valor_minimo:
            if not pd.isna(dataframe.at[index, column]):
               

                # Asignar el valor correspondiente al DataFrame
                dataframe.at[indice, column] = defined_columns.at[index, column]

                break

                
    
    return array


def nan_convertidos_a_valor_especifico(dataframe, column, value):
    Faltantes = dataframe.loc[dataframe[column].isna()].index
    for indice in Faltantes:
        dataframe.at[indice, column] = value
    return "Nan values were converted in ", value


### here you are assigning the median value of the 12 closest neighbors (you can change this)
def assign_closer_neighbor_median_value(column,dataframe):

    array = dataframe.loc[dataframe[column].isna()].index
    

    for indice in array:
        columnas_definidas = dataframe[dataframe[column].notna()].copy()

        columnas_definidas['Distancia'] = columnas_definidas.apply(
            lambda x: geodesic((dataframe.loc[indice]['Decimal Latitude'], dataframe.loc[indice]['Decimal Longitude']),
                            (x['Decimal Latitude'], x['Decimal Longitude'])).km,
            axis=1
        )


        indices_valor_minimo = columnas_definidas.nsmallest(12, 'Distancia').index

        # Calcular el promedio de los valores
        value = dataframe.loc[indices_valor_minimo, column].median()
       

        dataframe.at[indice, column] = value

                

                

    return array

def assigning_mean_std_values(dataframe, column):
    mean = dataframe[column].mean()
    std = dataframe[column].std()

    random_values = np.random.normal(mean, std, dataframe[column].isna().sum())

    dataframe.loc[dataframe[column].isna(), column] = random_values

    return None


def eliminating_columns_with_many_zeros(df, umbral=0.9):
    numerical_columns = df.select_dtypes(include=['number']).columns
    
    columns_to_be_eliminated = []
    for column in numerical_columns:
        porcentaje_ceros = (df[column] == 0).sum() / len(df)
        if porcentaje_ceros > umbral:
            columns_to_be_eliminated.append(column)
    
    df_resultado = df.drop(columns_to_be_eliminated, axis=1)
    return df_resultado