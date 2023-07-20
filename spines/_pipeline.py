"""用户预流失行为模型预测pipeline"""
# creator: birch.kwok 郭炳铭
# first release date: 2023-03-03
# TODO: 参考Keras新增函数式api：需要新增add方法，和按DAG图运行的执行flow，以便使用多个模型或者多路特征融合


import warnings

from typing import List, Tuple, Sized
from decimal import *
from functools import wraps
import re
from collections import OrderedDict
import gc

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import networkx as nx

from ._split_tools import df_block_split
from ._preprocessing import transform_dtypes_low_mem, inverse_transform_dtypes


warnings.filterwarnings('ignore')


class TransformerMixin:
    """ModelPipeline类Transformer内置类的父类

    """
    def set_name(self, name):
        self._name = name
        return
    def get_name(self):
        return self._name

    def get_params(self, *args, **kwargs):
        raise NotImplementedError

    def transform(self, *args, **kwargs):
        raise NotImplementedError


def check_dataframe(dataset):
    """dataframe检查函数，要求输入dataset必须为pandas dataframe，并且必须 dt 、userid 和 sensor_action 列
    :params:
    dataset: pandas dataframe

    :return:
    pandas dataframe

    """
    # 文件校验
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError("`dataset`必须为pandas DataFrame!")

    if 'dt' not in dataset.columns and 'userid' not in dataset.columns and 'sensor_action' not in dataset.columns:
        raise ValueError("`dataset`必须包含 dt 、userid 和 sensor_action 列!")
    return True


def check_name_has_symbol(name, symbol='-'):
    if not isinstance(symbol, str):
        raise TypeError("`symbol`必须为字符串!")

    if not isinstance(name, str) and name is not None:
        raise TypeError("`name`必须为字符串或None!")

    if name is not None:
        if re.search(symbol, name):
            raise ValueError(f"`name`中不能包含'{symbol}'符号")
    return name


class ForecastPipeline:
    """统一的模型封装pipeline类, 兼容具有predict、predict_proba方法的模型(已提前拟合)
    """

    def _init_(self) -> None:
        """
        :params:

        model: 具有predict 或 predict_proba方法实现的模型
        threshold: 模型概率预测阈值
        using_threshold: 是否使用阈值, 默认为True, 即启用阈值筛选方法, 为False时，输出模型的predict原始值

        :returns:
        None

        """
        self._pipeline = nx.DiGraph()
        self._current_sequences_list = []
        self._is_get_params_way = False

        self._params = self.get_params()

        self._global_sensor_cols = []

    def _set_model(self, model):
        if model is not None:
            if not any([hasattr(model, "predict"), hasattr(model, "predict_proba")]):
                raise ValueError("模型必须有predict或predict_proba方法实现!")

        self._fitted_model = model

    def _set_threshold(self, threshold, using_threshold):
        assert (threshold is None and using_threshold is False) or \
               (isinstance(threshold, (float, Decimal))
                and (0 < threshold < 1)), "`threshold`参数必须为浮点数，且大于0小于1!"

        self._threshold = Decimal(threshold)
        self._using_threshold = using_threshold

    def _check_name_append_to_seq(self, cls_, cls_name=None):
        """检查队列transformer名字，如与已有transformer重名则重命名为 {transformer_name}-{index}
        """
        assert cls_name is None or isinstance(cls_name, str)
        check_name_has_symbol(cls_name, '-')

        cls_name = cls_name or cls_.get_name()
        if cls_name is None:
            cls_name = cls_._name_

        if len(self._current_sequences_list) > 0:
            last_name_idx = 0
            for n in self._current_sequences_list:
                name = re.split('-', n[0])[0]

                if cls_name == name:
                    last_name_idx += 1

            cls_.set_name(cls_name + f'-{last_name_idx}' if last_name_idx > 0 else cls_name)

        self._pipeline.add_nodes_from()
        self._current_sequences_list.append((cls_.get_name(), cls_))

        return

    @staticmethod
    def _change_seq_to_dict(seq_list):
        """将pipeline队列转变为有序字典"""
        res = OrderedDict()
        for i in seq_list:
            res[i[0]] = i[1]
        return res

    def _check_operation(self, operation):
        # 20230603改，放开父类限制，但是要求此transformer类具有get_params和transform, set_name方法
        # 推荐统一继承自此脚本下的TransformerMixin类
        if not all([hasattr(operation, attr) for attr in ['get_params', 'transform', 'set_name', 'get_name']]):
            raise ValueError("transformer需要同时具有get_params, transform, set_name, get_name方法。")
        if not isinstance(operation.get_name(), str) or operation.get_name() is None:
            raise ValueError("transformer需要有默认名字, 并且名字为str类型或者为None")

        check_name_has_symbol(operation.get_name())

        return

    def _clean_seq(func):
        """清除原有队列"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self = args[0]
            self._current_sequences_list = []

            return func(*args, **kwargs)

        return wrapper

    # 清除原有队列
    @_clean_seq
    def sequences(self, list_of_operation: List[Tuple]):
        """设置pipeline处理顺序

        list_of_operation需要定义为list，其中元素需要为tuple，元祖首元素为transformer名称，第二个元素为transformer，
        形如：
        [
            ('fillna', model_pipeline.OutliersFiller(d)),
            ('feature_generate', model_pipeline.FeatureGenerator(func))
        ]
        """
        assert len(list_of_operation) > 0
        assert isinstance(list_of_operation, list)

        def check_one_step(step):
            if len(step) != 2:
                raise ValueError("`list_of_operation` 每个元素长度都需等于2")

            if not isinstance(step, tuple):
                raise ValueError("`list_of_operation` 每个元素都需要为 tuple 类型")

            if not isinstance(step[0], str):
                raise ValueError("`list_of_operation` 每个 tuple 的首元素必须为 str 类型")

            # 检查transformer是否符合规定
            self._check_operation(step[1])

        for i in list_of_operation:
            check_one_step(i)

            # 将实例化的对象存入pipeline列表
            self._check_name_append_to_seq(i[1], cls_name=i[0])

        # 刷新参数
        self._params = self.get_params()

        return self

    def add(self, operation, name=None):
        """顺序添加执行步骤"""
        self._check_operation(operation)

        self._check_name_append_to_seq(operation, cls_name=name)
        # 刷新参数
        self._params = self.get_params()

    def load_params(self, params):
        """使用参数字典快速重载pipeline
        params: dict

        # 使用方法
        import joblib

        # 导入旧pipeline
        loaded_model = joblib.load("/path/to/your/old_model.pkl")

        import unified_pipeline

        model_pipeline = unified_pipeline.ModelPipeline()

        model_pipline.load_params(loaded_model.get_params())

        # 保存新pipeline
        joblib.dump(model_pipline, "/path/to/your/new_model.pkl")
        """
        assert isinstance(params, dict), "params 必须为字典`dict`类型"
        assert params.get('sequences'), "params 必须有对应的处理pipeline"
        assert params.get('model'), "params 必须有对应的已拟合模型"

        for k in params.keys():
            if k not in self._params.keys() or k == 'model_pipeline_class_code':
                if k != 'model_pipeline_class_code':
                    print(f"`{k}`非有效键或不允许修改，已忽略。")
                continue

            self._params[k] = params[k]

        self._set_model(params['model'])
        self._set_threshold(threshold=params['threshold'],
                             using_threshold=params['using_threshold'])

        self.sequences(params['sequences'])

        return self

    def get_params(self):
        """获取当前pipeline参数

        """
        self._is_get_params_way = True
        self._params = {
            'model': self.get_model(),
            'threshold': self._threshold,
            'using_threshold': self._using_threshold,
            'sequences': self.get_current_sequences(),
            'model_pipeline_class_code': self.get_model_pipeline_class_code(),
            'feature_generator_func_code': self.get_feature_generator_func_code(),
        }

        self._is_get_params_way = False
        return self._params

    def get_feature_generator_func_code(self, transformer_name=None):
        """获取特征处理函数的源代码
        """
        if len(self._current_sequences_list) == 0:
            return {}

        if transformer_name is None:
            code_dict = {}
            for k, v in self._change_seq_to_dict(self._current_sequences_list).items():
                if 'source code' in v.get_params().keys():
                    code_dict[k] = v.get_params()['source code']
            return code_dict

        encoder = self._change_seq_to_dict(self._current_sequences_list).get(transformer_name)
        if encoder is not None and 'source code' in encoder.get_params().keys():
            return encoder.get_params()['source code']

        return {}

    def get_model_pipeline_class_code(self):
        """获取当前pipeline类的源代码
        """
        return self._model_pipeline_class_code

    def get_current_sequences(self):
        """获取当前pipeline列表
        """
        if len(self._current_sequences_list) == 0 and self._is_get_params_way is False:
            return
        else:
            return self._current_sequences_list

    def get_current_sequences_dict(self):
        """获取当前pipeline的有序字典
        """
        return self._change_seq_to_dict(self.get_current_sequences()) if len(
            self._current_sequences_list) > 0 else None

    def get_threshold(self):
        """获取当前pipeline的概率筛选阈值
        """
        return self._threshold

    def get_model(self):
        """获取当前pipeline的fitted model
        """
        return self._fitted_model

    def _clean_cache(self):
        self._global_sensor_cols = []

    def execute_pipeline(self, dataset):
        """执行当前pipeline预处理部分, 不包含模型预测步骤
        """
        # 清除环境缓存
        self._clean_cache()

        sensor_data, dataset = self._split_sensor_cols(dataset)

        self._global_sensor_cols.append(sensor_data)

        assert len(self._current_sequences_list) > 0  # 只有在初始化sequences后才能使用

        # 按顺序执行任务序列
        for op_name, op in self._change_seq_to_dict(self._current_sequences_list).items():
            print(f"Running {op_name} ...")
            dataset = op.transform(dataset)
            s_cols, dataset = self._split_sensor_cols(dataset)
            if len(s_cols) > 0:
                self._global_sensor_cols.append(s_cols)  # 每一步都将sensor开头的列筛选出来，并在下一个步骤中剔除，这点需要注意

        print(f"特征处理后，数据集空值为：{dataset.isna().sum().sum()} 个。")

        return dataset

    @staticmethod
    def _split_sensor_cols(dataset):
        """分割sensor开头的列
        """
        # 删除列名前面的空格符
        dataset.columns = dataset.columns.str.strip()

        if 'sensor_action' in dataset.columns:
            dataset.drop(columns='sensor_action', inplace=True)

        sensor_cols = dataset.columns[dataset.columns.str.strip().str.startswith('sensor_')]
        no_sensor_cols = dataset.columns[~dataset.columns.str.startswith('sensor_')]
        return dataset[sensor_cols], dataset[no_sensor_cols]

    @staticmethod
    def _concat_df_on_columns(df, dfs):
        _ = [df]
        for i in dfs:
            # 原地修改
            i.reset_index(drop=True, inplace=True)
            _.append(i)

        return pd.concat(_, axis=1)

    def _insert_to_result(self, userid, predict, predict_prob, sensor_action):
        """将固定列插入预测结果
        """
        results = pd.DataFrame(columns=self._const_results_columns)

        results['userid'] = userid
        results['predict'] = predict
        results['predict_prob'] = predict_prob
        results['sensor_action'] = sensor_action

        return results

    def predict(
            self,
            to_predict_data,
            return_proba=True,
            pos_label=1,
            model_predict_proba_args=None
    ):
        """将数据集切割后压缩处理"""

        check_dataframe(to_predict_data)
        assert to_predict_data.shape[0] > 0, "数据量必须大于0"

        transform_dtypes_low_mem(to_predict_data, verbose=True)
        dataset = to_predict_data.copy()
        df_idx = to_predict_data.index.copy()

        to_delete_columns = [
            column for column in to_predict_data.columns
            if not (column in ['userid', 'dt'] or str(column).strip().startswith('sensor_'))
        ]

        to_predict_data.drop(columns=to_delete_columns, inplace=True)

        # 重建索引
        dataset.reset_index(drop=True, inplace=True)

        into_model_data_list = df_block_split(dataset, rows_limit=10_0000)

        results = []
        for row_idx in into_model_data_list:
            # 分割样本
            df = dataset.loc[row_idx, :].reset_index(drop=True)
            # 删除掉已预测的条数，以节约内存
            dataset.drop(index=row_idx, inplace=True)

            inverse_transform_dtypes(df, int_dtypes=np.int32, float_dtypes=np.float32, verbose=False)
            res = self._predict(df, return_proba=return_proba, pos_label=pos_label,
                                 model_predict_proba_args=model_predict_proba_args)
            transform_dtypes_low_mem(res, verbose=False)

            results.append(res)

            gc.collect()

        predict_df = pd.concat(results, axis=0, ignore_index=True)
        predict_df.index = df_idx
        return predict_df

    def _predict(self,
                  dataset,
                  return_proba=True,
                  pos_label=1,
                  model_predict_proba_args=None):
        """
        :params:

        pos_label: 正类label
        dataset: pandas DataFrame
        return_proba: 是否需要返回样本的概率预测值, 仅当plain_predict为False时生效
        model_predict_proba_args: fitted model的predict_proba方法的关键字参数

        :returns:
        pandas DataFrame

        """

        userid = dataset['userid'].astype('category')
        sensor_action = dataset['sensor_action']

        into_model_data = self.execute_pipeline(dataset)

        # 兼容lightgbm.basic.Booster
        import lightgbm as lgb
        if not isinstance(self._fitted_model, lgb.basic.Booster):
            predictor = self._fitted_model.predict_proba

            def res_selector(s):
                res = s[:, pos_label]
                if not isinstance(res, Sized):
                    return np.array([res])
                return res
        else:
            predictor = self._fitted_model.predict

            def res_selector(s):
                res = s
                if not isinstance(res, Sized):
                    return np.array([res])
                return res

        # 分割样本
        into_model_data_list = df_block_split(into_model_data, rows_limit=10000)
        total_length = int(np.ceil(into_model_data.shape[0] / 10000))
        pred_res = []
        pred_res_prob = []

        for row_idx in tqdm(into_model_data_list, desc=f"model predicting...", total=total_length):
            imd = into_model_data.loc[row_idx, :]
            # 删除掉已预测的数据
            into_model_data.drop(index=row_idx, inplace=True)

            if model_predict_proba_args is not None and isinstance(model_predict_proba_args, dict):
                yp_prob = res_selector(
                    predictor(
                        imd, **model_predict_proba_args
                    )
                )
            else:
                yp_prob = res_selector(
                    predictor(
                        imd
                    )
                )

            pred_res_prob.append(yp_prob)

            if self._using_threshold:
                pred_res.append(
                    np.where(yp_prob >= self._threshold, 1, 0)
                )
            else:
                pred_res.append(
                    self._fitted_model.predict(imd)
                )

        if len(pred_res) > 1:
            predict = np.concatenate(pred_res, axis=0)
            predict_prob = np.concatenate(pred_res_prob, axis=0)
        else:
            predict = pred_res[0]
            predict_prob = pred_res_prob[0]

        if not return_proba:
            predict_prob = None

        results = self._insert_to_result(
            userid=userid,
            predict=predict,
            predict_prob=predict_prob,
            sensor_action=sensor_action
        )

        if len(self._global_sensor_cols) > 0:
            return self._concat_df_on_columns(results, self._global_sensor_cols)

        return results

    def _repr_(self):
        if len(self._current_sequences_list) == 0:
            return 'ModelPipeline Processing Flow: \n'
        else:
            _ = list(self._change_seq_to_dict(self._current_sequences_list).keys())

        return 'ModelPipeline Processing Flow: \n' + '\n'.join([
            '↓ ' + str(_[i]) if i < len(_) - 1 else '-- ' + str(_[i]) for i in range(len(_))
        ])

    def _str_(self):
        return self._repr_()


