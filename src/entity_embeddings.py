import gc
from pickletools import optimize
import oc
import gc
import joblib
import numpy as np
import pandas as pd

from sklearn import metrics, preprocessing
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
from tensorflow.keras import backend as K
from tensorflow.keras import utils
from tensorflow.keras.models import Model, load_model

def create_model(data, catcols):
    """
    :param data:pandasデータフレーム
    :param catcols:質的変数の列リスト
    :return: tf.keras model
    """
    inputs = []
    outputs = []

    for c in catcols:
        num_unique_values = int(data[c].nunique())
        #埋め込み次元数の計算
        embed_dim = int(min(np.ceil((num_unique_values/2)),50))

        #kerasのサイズ1の入力
        inp = layers.Input(shape=(1,))

        #埋め込み層
        out = layers.Embedding(
            num_unique_values + 1,
            embed_dim,
            name = c
        )(inp)

        #なにこれ
        out = layers.SpatialDropout1D(0.3)(out)

        #出力用に変形
        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        inputs.append(inp)
        outputs.append(out)
    
    x = layers.Concatenate()(outputs)

    #backnorm層の追加
    x = layers.BatchNormalization()(x)

    #ドロップアウト付き全結合層の層数
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    #softmax関数を追加し二値分類を解く
    y = layers.Dense(2, activation="softmax")(x)

    #最終的なモデル
    model = Model(inputs=inputs, outputs=y)

    #モデルの作成
    model.compile(loss="binary_crossentropy", optimizer = "adam")
    print(model)
    return model


def run(fold):
    
    df = pd.read_csv("../input/cat_train_folds.csv")
    features = [f for f in df.columns if f not in ("id", "target", "kfold")]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    for col in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:,col] = lbl_enc.fit_transform(df[col].values)

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = create_model(df, features)

    x_train = [df_train[features].values[:,k] for k in range(len(features))]
    x_valid = [df_valid[features].values[:,k] for k in range(len(features))]

    y_train = df_train.target.values
    y_valid = df_valid.target.values

    y_train_cat = utils.to_categorical(y_train)
    y_valid_cat = utils.to_categorical(y_valid)

    model.fit(
        x_train,
        y_train_cat,
        validation_data = (x_valid, y_valid_cat),
        verbose = 1,
        batch_size = 1024,
        epochs = 3
    )

    valid_preds = model.predict(x_valid)[:,1]
    auc = metrics.roc_auc_score(y_valid, valid_preds)
    print(f"Fold={fold}, AUC={auc}")

    K.clear_session()

if __name__ == "__main__":
    for f in range(5):run(f)

