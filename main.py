#モジュールのインポート
import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import codecs
import pandas as pd
from matplotlib import pyplot as plt
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve, mean_squared_error, make_scorer, r2_score , mean_absolute_error,mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import math
from base64 import b64encode


# タイトルを表示

st.set_page_config(layout="centered",  page_title="線形モデル作成ツール")
st.title("GUI De Modeling")
st.markdown("created by Keisuke Kamata")
st.markdown("簡単に線形モデルを作成するためのツールです。データ加工はこのアプリに投入する前に行ってください。エラーが出た際は kamata.keisuke.kyoto@gmail.com までご連絡ください")

st.sidebar.markdown("### 1. データの読み込み")
uploaded_file = st.sidebar.file_uploader("CSVファイルをドラッグ&ドロップ、またはブラウザから選択してください", type='csv', key='train')

if uploaded_file is not None:
    #データの読込み
    df = pd.read_csv(uploaded_file)
    df_0 = df

    #object型をcategory型に変更

    @st.cache
    def change_categoryformat(df):
        df_output = df
        df_output.loc[:, df_output.dtypes == 'object'] = df.select_dtypes(['object']).apply(lambda x: x.astype('category'))
        return df_output

    df = change_categoryformat(df)


    #ID・目的変数の選択
    st.sidebar.markdown("### 2. ID・目的変数の選択")
    target = st.sidebar.selectbox(
        '目的変数を選択してください',
        df.columns
    )
    id = st.sidebar.selectbox(
        'IDを選択してください',
        df.columns
    )

    #st.dataframe(df)
    #st.markdown(df.shape)
    
    #説明変数の選択

    st.sidebar.markdown("### 3. 説明変数の選択")
    unique_columns = df.drop(columns = [target]).columns.values
    cols = st.sidebar.multiselect("",unique_columns,[])
    
    #モデル構築
    st.sidebar.markdown("### 4. モデル作成開始")
    if cols is not None:
        cv      = st.sidebar.slider("交差検定数", min_value =1, max_value=50, value=10)
        st.sidebar.write("※ホールドアウト用のデータが準備できる場合、別途準備してください")

        if st.sidebar.button('モデル作成開始'):

            # check process    
            if df[target].isnull().any() == "True":
                st.write("errorです: 目的変数に欠損が含まれております。欠損が内容にデータを準備してください")
                st.stop()
            
            if df[target].dtypes != "float64" and df[target].dtypes != "int64":
                try:
                    df[target] = df[target].astype("float64")
                except:
                    st.write("errorです: 目的変数にカテゴリが含まれている可能性があります。データを確認し、再度データを取り込んでください")
                    st.stop()
                df[target] = df[target].astype("float64")

            for c in cols:
                if df[c].dtypes != "float64" and df[c].dtypes != "int64":
                    try:
                        df[c] = df[c].astype("float64")
                    except:
                        st.write("errorです: ",c,"にカテゴリが含まれている可能性があります。データを確認して、再度データを取り込んでください")
                        st.stop()
                    df[c] = df[c].astype("float64")                

            with st.spinner('実行中...'):


                @st.cache
                def modeling(df,target,cols):
                    Y = df[target]
                    X = df[cols]

                    # 標準化する変数をリストアップ
                    standard_cols = []
                    for c in cols:
                        if len(df[c].unique()) > 2:
                            standard_cols.append(c)


                    # モデリング & 評価
                    folds = KFold(n_splits= cv, shuffle=True, random_state=0)
                    oof_preds = np.zeros(X.shape[0])
                    y_preds = []
                    y_vals = []
                    models = []
                    importance_list = []
                    perm_imp_list = []
                    val_inds = []
                    fold_label = np.zeros(X.shape[0])
                    def rmse(y_true, y_pred):
                        return np.sqrt(mean_squared_error(y_true, y_pred))
                    mse_scorer = make_scorer(rmse)

                    for train_ind, val_ind in folds.split(X):
                        # partition
                        x_train, x_val = X.iloc[train_ind], X.iloc[val_ind]
                        y_train, y_val = Y.iloc[train_ind], Y.iloc[val_ind]
                        val_inds+=val_ind.tolist()
                        
                        # missing value imputation (欠損flagも立てず、まずは中央値補完 <= もしデータの性質として欠損=0などがわかっている場合は、それを適用)
                        imp = SimpleImputer(missing_values=np.nan, strategy='median')

                        imp.feature_names_in_ = cols
                        imp.fit(x_train)
                        x_train.loc[:] = imp.transform(x_train)
                        x_val.loc[:] = imp.transform(x_val)
                        
                        # standardization / scaler 
                        scaler = StandardScaler()

                        scaler.fit(x_train[standard_cols])
                        x_train.loc[:,standard_cols], x_val.loc[:,standard_cols] = scaler.transform(x_train[standard_cols]), scaler.transform(x_val[standard_cols])
                        
                        # model
                        regr =ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000000000,random_state=0,normalize=False)
                        regr.fit(x_train, y_train)
                        
                        # prediction
                        y_pred   =regr.predict(x_val)
                        y_preds +=y_pred.tolist()
                        y_vals  +=y_val.tolist() 
                        
                        # run permutation importance
                        result = permutation_importance(regr, x_train, y_train, scoring=mse_scorer, n_repeats=10, n_jobs=-1, random_state=71)
                        perm_imp_df = pd.DataFrame({"importances_mean":result["importances_mean"], "importances_std":result["importances_std"]}, index=X.columns)
                        perm_imp_list += [perm_imp_df]
                    
                    Result = pd.DataFrame(columns=[id])
                    Result["id"] = df_0.iloc[val_inds][id].tolist()
                    Result["actual"] = df_0.iloc[val_inds][target].tolist()
                    Result["prediction"]   = y_preds
                    Result["actual-prediction"] = Result["actual"] - Result["prediction"]
                    for col in cols:
                        Result[col] = df_0.iloc[val_inds][col].tolist()
                    return Result, y_vals,y_preds,perm_imp_list

                Result,y_vals,y_preds,perm_imp_list= modeling(df,target,cols)
                

                ## evaluation
                # metric
                # st.markdown("#### 精度")

                col1, col2, col3 = st.columns(3)
                col1.metric(label="R2", value= round(r2_score(y_vals, y_preds),4))
                col2.metric(label="MAE", value= mean_absolute_error(y_vals, y_preds))
                col3.metric(label="MAPE", value= round(mean_absolute_percentage_error(y_vals, y_preds),4))
            
                # permutation importance
                # 参考: https://qiita.com/kenmatsu4/items/c49059f78c2b6fed0929
                @st.cache
                def permutation_importance_calc(df,cols,perm_imp_list):
                    permutation_imporatance_result = pd.DataFrame(columns = ["importances_mean"], index=df[cols].columns)
                    for col in df[cols].columns:
                        importance = 0
                        for i in range(0,5):
                            importance += perm_imp_list[i].loc[col,'importances_mean']
                        permutation_imporatance_result.loc[col,'importances_mean'] = importance/5
                    permutation_imporatance_result = permutation_imporatance_result*(-1)
                    permutation_imporatance_result.sort_values(by ='importances_mean', inplace=True, ascending=True)
                    return permutation_imporatance_result

                permutation_imporatance_result = permutation_importance_calc(df,cols,perm_imp_list)
                

            
                # figure
                # 精度
                fig = px.scatter(Result, x="actual",y="prediction",hover_name="id")
                reference_line = go.Scatter(x=[min(Result["actual"].append(Result["prediction"])),max(Result["actual"].append(Result["prediction"]))], y=[min(Result["actual"].append(Result["prediction"])),max(Result["actual"].append(Result["prediction"]))],mode = 'lines',
                    line=dict(color='gray', width=1,dash='dot'),showlegend=False)
                fig.add_trace(reference_line)

                layout = go.Layout(
                    title=dict(text='実測 vs 予測',),
                    title_x=0.5,
                    xaxis=dict(title='実測',),
                    yaxis=dict(title='予測',),
                    plot_bgcolor='white',
                )
                fig.update_xaxes(showline=True,
                                linewidth=1,
                                linecolor='lightgrey',
                                color='grey')
                fig.update_yaxes(showline=True,
                                linewidth=1,
                                linecolor='lightgrey',
                                color='grey')
                fig['layout'].update(layout)
                st.plotly_chart(fig, use_container_width=True)

                # 係数と変数の重要性
                col1, col2 = st.columns([3,2])

                # 変数の重要性
                permutation_imporatance_result.sort_values(by ='importances_mean', inplace=True, ascending=True)
                fig = go.Figure(go.Bar(
                            x = permutation_imporatance_result['importances_mean'],
                            y=permutation_imporatance_result.index.values,
                            orientation='h'))
                layout = go.Layout(
                    plot_bgcolor='white',
                )

                fig['layout'].update(layout)
                col1.write("変数の重要性 (Permutation importnace)")
                col1.plotly_chart(fig, use_container_width=True)
            
                # 全てのデータを使ってmodelingをし、係数を抽出
                def modeling_all(df,target,cols):
                    Y = df[target].copy()
                    X = df.drop(columns = [target]).copy()
                    # 特徴量セット Final
                    X = X[cols]

                    # missing value imputation
                    imp = SimpleImputer(missing_values=np.nan, strategy='median')
                    imp.feature_names_in_ = cols
                    imp.fit(X)
                    X.loc[:] = imp.transform(X)

                    # standadization / scaler

                    standard_cols = []
                    for c in cols:
                        if len(df[c].unique()) > 2:
                            standard_cols.append(c)

                    scaler = StandardScaler()
                    scaler.feature_names_in_ = standard_cols
                    scaler.fit(X[standard_cols])
                    X.loc[:,standard_cols] = scaler.transform(X[standard_cols])


                    # model
                    regr =ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=1000,random_state=0)
                    regr.fit(X, Y)


                    # coefficient & intercept

                    Result_coef = pd.DataFrame()

                    pred = 0 
                    coefs = []
                    cols_input = cols
                    intercept = regr.intercept_

                    permutation_imporatance_result.sort_values(by ='importances_mean', inplace=True, ascending=False)
                    Result_coef = pd.DataFrame(columns = ["coefficient"], index=permutation_imporatance_result.index.values)
                
                    for col in cols:
                        index_0 = cols.index(col)
                        coef = regr.coef_[index_0]
                        if col in scaler.feature_names_in_:
                            index = scaler.feature_names_in_.tolist().index(col)
                            std  = math.sqrt(scaler.var_[index])
                            mean = scaler.mean_[index]
                            intercept  += (-1)*coef*mean/std
                            pred += df.loc[3,col]*coef/std
                        
                            Result_coef.loc[col,"coefficient"] = coef/std
                        else:
                            Result_coef.loc[col,"coefficient"] = coef
                    Result_coef.loc["切片","coefficient"] = intercept
                    return Result_coef
                    
                Result_coef = modeling_all(df,target,cols)

                col2.write("係数")
                col2.dataframe(Result_coef)

                @st.cache
                def convert_df(df):
                    # IMPORTANT: Cache the conversion to prevent computation on every rerun
                    return df.to_csv().encode('utf-8')

                csv = convert_df(Result_coef)

                st.download_button(
                    label="Download 係数",
                    data=csv,
                    file_name='coeffient.csv',
                    mime='text/csv',
                )

                csv2 = convert_df(Result)

                st.download_button(
                    label="Download 交差検定予測結果",
                    data=csv2,
                    file_name='prediction.csv',
                    mime='text/csv',
                )

                

         
            



            