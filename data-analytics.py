datasets = ['numerical',
    'categorical',
    'concatenated']

### You must have defined "dataframe_numerical" (only numbers), "dataframe_categorical" (only one-hot encoding variables) y "dataframes_concatenated" (numbers and one-hot encoding dataframes)
def models_loop(datasets):
    for dataset_name in datasets:
        
        
        if dataset_name == 'numerical':
            
            
            X = dataframe_numerical.drop(["RDI v3.1"], axis=1)
            X = X.drop(["Year"], axis=1)


            y = dataframe_numerical["RDI v3.1"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, 
                                                        shuffle=True, random_state=2)
            
            numericalresults_rand = RandomForestModel(X_train,X_test,y_train,y_test)
            numericalresults_lgbm = lgbm(X_train,X_test,y_train,y_test)
            numericalresults_xgboost = xgboostmodel(X_train,X_test,y_train,y_test)
            numericalresults_svmmodel = svmmodel(X_train,X_test,y_train,y_test)

            


        elif dataset_name == 'categorical':
            

            X = dataframe_categorical


            y = dataframe_numerical["RDI v3.1"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, 
                                                        shuffle=True, random_state=2)
            
            categoricalresults_rand = RandomForestModel(X_train,X_test,y_train,y_test)
            categoricalresults_lgbm = lgbm(X_train,X_test,y_train,y_test)
            categoricalresults_xgboost = xgboostmodel(X_train,X_test,y_train,y_test)
            categoricalresults_svmmodel = svmmodel(X_train,X_test,y_train,y_test)




        elif dataset_name == 'concatenated':
            

            X = dataframes_concatenated


            y = dataframe_numerical["RDI v3.1"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, 
                                                        shuffle=True, random_state=2)
            
            concatenatedresults_rand = RandomForestModel(X_train,X_test,y_train,y_test)
            concatenatedresults_lgbm = lgbm(X_train,X_test,y_train,y_test)
            concatenatedresults_xgboost = xgboostmodel(X_train,X_test,y_train,y_test)
            concatenatedresults_svmmodel = svmmodel(X_train,X_test,y_train,y_test)
            

            
        else:
            print("Unknown")
        
        print("\n")
    return numericalresults_rand, numericalresults_lgbm,numericalresults_xgboost,numericalresults_svmmodel,categoricalresults_rand,categoricalresults_lgbm,categoricalresults_xgboost,categoricalresults_svmmodel,concatenatedresults_rand,concatenatedresults_lgbm,concatenatedresults_xgboost,concatenatedresults_svmmodel


def building_df(numericalresults_rand, numericalresults_lgbm,numericalresults_xgboost,numericalresults_svmmodel,categoricalresults_rand,categoricalresults_lgbm,categoricalresults_xgboost,categoricalresults_svmmodel,concatenatedresults_rand,concatenatedresults_lgbm,concatenatedresults_xgboost,concatenatedresults_svmmodel
):

    numerical_df = pd.DataFrame({
        'Modelo': ['Random Forest', 'LGBM', 'XGBoost', 'SVM'],
        'Train MSE': [numericalresults_rand[0], numericalresults_lgbm[0], numericalresults_xgboost[0], numericalresults_svmmodel[0]],
        'Test MSE': [numericalresults_rand[1], numericalresults_lgbm[1], numericalresults_xgboost[1], numericalresults_svmmodel[1]],
        'Train R2': [numericalresults_rand[2], numericalresults_lgbm[2], numericalresults_xgboost[2], numericalresults_svmmodel[2]],
        'Test R2': [numericalresults_rand[3], numericalresults_lgbm[3], numericalresults_xgboost[3], numericalresults_svmmodel[3]]
    })

    categorical_df = pd.DataFrame({
        'Modelo': ['Random Forest', 'LGBM', 'XGBoost', 'SVM'],
        'Train MSE': [categoricalresults_rand[0], categoricalresults_lgbm[0], categoricalresults_xgboost[0], categoricalresults_svmmodel[0]],
        'Test MSE': [categoricalresults_rand[1], categoricalresults_lgbm[1], categoricalresults_xgboost[1], categoricalresults_svmmodel[1]],
        'Train R2': [categoricalresults_rand[2], categoricalresults_lgbm[2], categoricalresults_xgboost[2], categoricalresults_svmmodel[2]],
        'Test R2': [categoricalresults_rand[3], categoricalresults_lgbm[3], categoricalresults_xgboost[3], categoricalresults_svmmodel[3]]
    })

    concatenated_df = pd.DataFrame({
        'Modelo': ['Random Forest', 'LGBM', 'XGBoost', 'SVM'],
        'Train MSE': [concatenatedresults_rand[0], concatenatedresults_lgbm[0], concatenatedresults_xgboost[0], concatenatedresults_svmmodel[0]],
        'Test MSE': [concatenatedresults_rand[1], concatenatedresults_lgbm[1], concatenatedresults_xgboost[1], concatenatedresults_svmmodel[1]],
        'Train R2': [concatenatedresults_rand[2], concatenatedresults_lgbm[2], concatenatedresults_xgboost[2], concatenatedresults_svmmodel[2]],
        'Test R2': [concatenatedresults_rand[3], concatenatedresults_lgbm[3], concatenatedresults_xgboost[3], concatenatedresults_svmmodel[3]]
    })

    final_df = pd.concat([numerical_df, categorical_df, concatenated_df], keys=['Numerical', 'Categorical', 'Concatenated'])

    return final_df


def scatter_plots(final_df,numericalresults_rand, numericalresults_lgbm, numericalresults_xgboost, numericalresults_svmmodel, categoricalresults_rand, categoricalresults_lgbm, categoricalresults_xgboost, categoricalresults_svmmodel,
                            concatenatedresults_rand, concatenatedresults_lgbm, concatenatedresults_xgboost, concatenatedresults_svmmodel):
    
    numerical_rows = final_df.xs('Numerical', level=0)

    columnas_a_visualizar = numerical_rows['Modelo']

    fig = make_subplots(rows=3, cols=4, subplot_titles=columnas_a_visualizar, shared_xaxes=False, shared_yaxes=True, row_titles=final_df.index.levels[0].tolist())

    numerical_results_list = [numericalresults_rand, numericalresults_lgbm, numericalresults_xgboost, numericalresults_svmmodel, categoricalresults_rand, categoricalresults_lgbm, categoricalresults_xgboost, categoricalresults_svmmodel,
                            concatenatedresults_rand, concatenatedresults_lgbm, concatenatedresults_xgboost, concatenatedresults_svmmodel]

    for i, numerical_results in enumerate(numerical_results_list, 1):
        fila = (i - 1) // 4 + 1
        columna_grafico = (i - 1) % 4 + 1

        trace = go.Scatter(x=numerical_results[4], y=numerical_results[5], mode='markers', name=numerical_results[0])
        fig.add_trace(trace, row=fila, col=columna_grafico)

    fig.update_layout(height=900, width=1500, title_text='Test data')

    return fig.show()


def RandomForestModel(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    Mse_train = mean_squared_error(y_train, y_pred_train)
    Mse_test = mean_squared_error(y_test, y_pred_test)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    return Mse_train, Mse_test, r2_train, r2_test, y_test, y_pred_test


def lgbm(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_data = lgb.Dataset(X_train_scaled, label=y_train)
    test_data = lgb.Dataset(X_test_scaled, label=y_test, reference=train_data)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
    }

    num_round = 100
    bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

    Y_train = bst.predict(X_train_scaled)
    y_val = bst.predict(X_test_scaled)

    rmse_train = mean_squared_error(y_train, Y_train, squared=False)
    rmse_test = mean_squared_error(y_test, y_val, squared=False)

    r2_train = r2_score(y_train, Y_train)
    r2_test = r2_score(y_test, y_val)

    return rmse_train, rmse_test, r2_train, r2_test, y_test, y_val



def xgboostmodel(X_train, X_test, y_train, y_test):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 5,
        'learning_rate': 0.1
    }

    bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')], early_stopping_rounds=10, verbose_eval=False)

    y_pred_train = bst.predict(dtrain)
    y_pred_test = bst.predict(dtest)

    mse_train = mean_squared_error(y_train, y_pred_train)
    r2_train = r2_score(y_train, y_pred_train)

    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_test = r2_score(y_test, y_pred_test)

    return mse_train, mse_test, r2_train, r2_test, y_test, y_pred_test



def svmmodel(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svr = SVR(kernel='linear', C=1.0)
    svr.fit(X_train_scaled, y_train)

    y_train_pred = svr.predict(X_train_scaled)
    y_test_pred = svr.predict(X_test_scaled)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return mse_train, mse_test, r2_train, r2_test, y_test, y_test_pred
