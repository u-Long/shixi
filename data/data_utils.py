import numpy as np
import pandas as pd
import datetime
import time
import random
from tqdm import tqdm
import os
# from statsmodels.tsa.stattools import adfuller
from multiprocessing import Pool
from functools import partial
import torch
# from jqdatasdk import *
# auth('','')

# ts_pro 延迟初始化，避免 import 时触发 tushare 空 token 请求
_ts_pro = None

def _get_ts_pro():
    global _ts_pro
    if _ts_pro is None:
        import tushare as ts
        _ts_pro = ts.pro_api("")
    return _ts_pro

# 兼容旧代码直接用 ts_pro 的地方，在模块加载时不再执行网络请求
class _LazyTsPro:
    def __getattr__(self, name):
        return getattr(_get_ts_pro(), name)

ts_pro = _LazyTsPro()

# 特征截断dict
clip_dict = {
    "amount": [0, 10],
    "buy_elg_amount": [0, 30],
    "buy_elg_vol": [0, 30],
    "buy_lg_amount": [0, 50],
    "buy_lg_vol": [0, 50],
    "buy_md_amount": [0, 70],
    "buy_md_vol": [0, 70],
    "buy_sm_amount": [0, 100],
    "buy_sm_vol": [0, 100],
    "change": [-4, 4],
    "circ_mv": [-400, 400],
    "pb": [-0.1, 0.1],
    "pe_ttm": [-0.6, 0.6],
    "pe": [-0.6, 0.6],
    "ps_ttm": [-0.6, 0.6],
    "ps": [-0.6, 0.6],
    "sell_elg_amount": [0, 40],
    "sell_elg_vol": [0, 40],
    "sell_lg_amount": [0, 80],
    "sell_lg_vol": [0, 80],
    "sell_md_amount": [0, 80],
    "sell_md_vol": [0, 80],
    "sell_sm_amount": [0, 80],
    "sell_sm_vol": [0, 80],
    "total_mv": [-400, 400],
    "turnover_rate_f": [0, 20],
    "turnover_rate": [0, 20],
    "up_limit": [-0.1, 0.1],
    "down_limit":[-0.1,0.1],
    "vol": [0, 10],
    "volume_ratio": [0, 5],
    "open": [-0.2,0.2], ##
    "high": [-0.2,0.2], ##
    "low": [-0.2,0.2], ##
    "close": [-0.2,0.2], ##
    "pre_close": [-0.2,0.2], ##
    "ma5": [-0.2,0.2], ##
    "ma10": [-0.2,0.2], ##
    "ma15": [-0.2,0.2], ##
    "ma20": [-0.2,0.2], ##
    "ma25": [-0.2,0.2], ##
    "pct_chg": [-0.2,0.2], ##
    "return_1": [-0.5, 0.5], ##
    "return_5": [-0.5 ,0.5], ##
    "return_10": [-0.5 ,0.5], ##
}

    # "pct_chg": [-0.25,0.25],
    # "return_5": [- ,0.5],
    # "return_10": [-0.5 ,0.5],


def get_origin_data_from_tushare(
    save_folder_path, stock_list, start_date, end_date, reload_all=False
) -> None:
    """从tushare获取数据.

    Args:
        save_folder_path (str): 原始数据保存路径.
        stock_list (list): 股票列表或列表名称或空.
        start_date (str | None): 开始日期.
        end_date (str | None): 截止日期.
        reload_all (_type_, optional): 是否全部重新获取. Defaults to False:bool.
    """
    # get all stock basic info
    stock_info = ts_pro.stock_basic()
    # del stock name with "ST"&"退"
    # stock_info = stock_info[~stock_info.name.str.contains("ST")]
    # stock_info = stock_info[~stock_info.name.str.contains("退")]
    # set ts_code as index
    stock_info = stock_info.set_index("ts_code")
    # 今日停牌股list
    # suspend_list = list(
    #     ts_pro.suspend_d(
    #         trade_date=datetime.date.today().strftime("%Y%m%d"), suspend_type="S"
    #     ).ts_code
    # )
    # 去掉停牌股票
    # stock_info = stock_info.reindex(
    #     [i for i in stock_info.index if i not in suspend_list]
    # )

    # 对股票市场信息进行编码
    stock_industry = stock_info.industry.fillna("None")
    industry_list = list(set(stock_industry))
    industry_list.sort()
    industry_list = {
        i: (industry_list.index(i) + 1) / len(industry_list) for i in industry_list
    }
    for i in industry_list:
        stock_industry.replace(
            to_replace=i, value=industry_list[i], inplace=True)

    # 输入股票列表
    if isinstance(stock_list, list):
        print("Using the selected stocks.")
    # 全体股票
    elif stock_list == None:
        stock_list = list(stock_info.index)
        stock_list.sort()
        print("Using the whole range of Chinese stock market.")
    # 沪深300
    elif stock_list == "csi300":
        stock_list = (
            ts_pro.index_weight(
                index_code="399300.SZ").iloc[:300]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI300.")
    # 中证500
    elif stock_list == "csi500":
        stock_list = (
            ts_pro.index_weight(
                index_code="000905.SH").iloc[:500]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI500.")
    # 中证500+沪深300
    elif stock_list == "csi800":
        csi300_stock_list = (
            ts_pro.index_weight(
                index_code="399300.SZ").iloc[:300]["con_code"].tolist()
        )
        csi500_stock_list = (
            ts_pro.index_weight(
                index_code="000905.SH").iloc[:500]["con_code"].tolist()
        )
        stock_list = csi300_stock_list + csi500_stock_list
        stock_list.sort()
        print("Using the constituent stocks of market index CSI800.")
    # 中证1000
    elif stock_list == "csi1000":
        stock_list = (
            ts_pro.index_weight(
                index_code="000852.SH").iloc[:1000]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI1000.")

    else:
        print("Wrong type of stock list.")
        return
    print("stock list len is ", len(stock_list))
    time.sleep(0.3)

    # 数据存储文件夹
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 获取各股数据
    for s in tqdm(stock_list):
        # 如果不是重新加载全部，并且当前股票已存在，则跳过
        if not reload_all and os.path.exists(
            os.path.join(save_folder_path, f"{s}.pkl")
        ):
            continue

        # 获取原始数据
        df = []
        try:
            # 日频交易
            data_df = ts_pro.daily(
                ts_code=s, start_date=start_date, end_date=end_date
            ).set_index(["ts_code", "trade_date"])
            df.append(data_df)

            # 日频每日指标
            basic_df = ts_pro.daily_basic(
                ts_code=s, start_date=start_date, end_date=end_date
            ).set_index(["ts_code", "trade_date"])
            basic_df.drop("close", axis=1, inplace=True)
            df.append(basic_df)

            # 日频个股资金流向
            moneyflow_df = ts_pro.moneyflow(
                ts_code=s, start_date=start_date, end_date=end_date
            ).set_index(["ts_code", "trade_date"])
            df.append(moneyflow_df)

            # 每日涨跌停价格
            stk_limit_df = ts_pro.stk_limit(
                ts_code=s, start_date=start_date, end_date=end_date
            ).set_index(["ts_code", "trade_date"])
            df.append(stk_limit_df)

            df = pd.concat(df, axis=1).fillna(0).droplevel(0)
            df.index = pd.DatetimeIndex(df.index)
            df = df.sort_index()
            for w in [5, 10, 15, 20, 25]:
                df["ma{}".format(w)] = df["close"].rolling(
                    window=w).mean().fillna(0)
            df["industry"] = stock_industry[s]
            if end_date is not None:
                df = df[df.index <= end_date]
            if start_date is not None:
                df = df[df.index >= start_date]
            df.to_pickle(os.path.join(save_folder_path, f"{s}.pkl"))
        except Exception as e:
            print(f"Get Stock Data Failed. Error:{e}")

    print("Get Origin Data From Tushare Finished.")


def plus_get_origin_data_from_tushare(
    save_folder_path, stock_list, start_date, end_date
) -> None:
    """从tushare获取数据.

    Args:
        save_folder_path (str): 原始数据保存路径.
        stock_list (list): 股票列表或列表名称或空.
        start_date (str | None): 开始日期.
        end_date (str | None): 截止日期.
        reload_all (_type_, optional): 是否全部重新获取. Defaults to False:bool.
    """
    # get all stock basic info
    stock_info = ts_pro.stock_basic()
    # del stock name with "ST"&"退"
    # stock_info = stock_info[~stock_info.name.str.contains("ST")]
    # stock_info = stock_info[~stock_info.name.str.contains("退")]
    # set ts_code as index
    stock_info = stock_info.set_index("ts_code")
    # 今日停牌股list
    # suspend_list = list(
    #     ts_pro.suspend_d(
    #         trade_date=datetime.date.today().strftime("%Y%m%d"), suspend_type="S"
    #     ).ts_code
    # )
    # 去掉停牌股票
    # stock_info = stock_info.reindex(
    #     [i for i in stock_info.index if i not in suspend_list]
    # )

    # 对股票市场信息进行编码
    stock_industry = stock_info.industry.fillna("None")
    industry_list = list(set(stock_industry))
    industry_list.sort()
    industry_list = {
        i: (industry_list.index(i) + 1) / len(industry_list) for i in industry_list
    }
    for i in industry_list:
        stock_industry.replace(
            to_replace=i, value=industry_list[i], inplace=True)

    # 输入股票列表
    if isinstance(stock_list, list):
        print("Using the selected stocks.")
    # 全体股票
    elif stock_list == None:
        stock_list = list(stock_info.index)
        stock_list.sort()
        print("Using the whole range of Chinese stock market.")
    # 沪深300
    elif stock_list == "csi300":
        stock_list = (
            ts_pro.index_weight(
                index_code="399300.SZ").iloc[:300]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI300.")
    # 中证500
    elif stock_list == "csi500":
        stock_list = (
            ts_pro.index_weight(
                index_code="000905.SH").iloc[:500]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI500.")
    # 中证500+沪深300
    elif stock_list == "csi800":
        csi300_stock_list = (
            ts_pro.index_weight(
                index_code="399300.SZ").iloc[:300]["con_code"].tolist()
        )
        csi500_stock_list = (
            ts_pro.index_weight(
                index_code="000905.SH").iloc[:500]["con_code"].tolist()
        )
        stock_list = csi300_stock_list + csi500_stock_list
        stock_list.sort()
        print("Using the constituent stocks of market index CSI800.")
    # 中证1000
    elif stock_list == "csi1000":
        stock_list = (
            ts_pro.index_weight(
                index_code="000852.SH").iloc[:1000]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI1000.")
    else:
        print("Wrong type of stock list.")
        return
    print("stock list len is ", len(stock_list))
    time.sleep(0.3)

    # 数据存储文件夹
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 获取各股数据
    for s in tqdm(stock_list):
        
        save_path=os.path.join(save_folder_path, f"{s}.pkl")
        if os.path.exists(
           save_path
        ):
            df=pd.read_pickle(save_path)
            final_date=str(df.index.values[-1])[:10]

            
            if final_date>start_date:
                next_start_date=final_date
                
            else:
                next_start_date=start_date
        
        else:
            next_start_date=start_date

        
        # 获取原始数据
        df = []
        try:
            # 日频交易
            data_df = ts_pro.daily(
                ts_code=s, start_date=next_start_date, end_date=end_date
            ).set_index(["ts_code", "trade_date"])
            df.append(data_df)

            # 日频每日指标
            basic_df = ts_pro.daily_basic(
                ts_code=s, start_date=next_start_date, end_date=end_date
            ).set_index(["ts_code", "trade_date"])
            basic_df.drop("close", axis=1, inplace=True)
            df.append(basic_df)

            # 日频个股资金流向
            moneyflow_df = ts_pro.moneyflow(
                ts_code=s, start_date=next_start_date, end_date=end_date
            ).set_index(["ts_code", "trade_date"])
            df.append(moneyflow_df)

            # 每日涨跌停价格
            stk_limit_df = ts_pro.stk_limit(
                ts_code=s, start_date=next_start_date, end_date=end_date
            ).set_index(["ts_code", "trade_date"])
            df.append(stk_limit_df)

            df = pd.concat(df, axis=1).fillna(0).droplevel(0)
            df.index = pd.DatetimeIndex(df.index)
            df = df.sort_index()
            for w in [5, 10, 15, 20, 25]:
                df["ma{}".format(w)] = df["close"].rolling(
                    window=w).mean().fillna(0)
            df["industry"] = stock_industry[s]
            if end_date is not None:
                df = df[df.index <= end_date]
            if start_date is not None:
                df = df[df.index >= start_date]

            if os.path.exists(
                    save_path
            ):
                old_df=pd.read_pickle(save_path)
                new_df=pd.concat([old_df,df])    
                new_df=new_df.drop_duplicates(keep='first')
                print(new_df.duplicated().any())
                new_df.to_pickle(save_path)

            else:
                print('new df')
                print(df)
                df.to_pickle(save_path)
        except Exception as e:
            print(f"Get Stock Data Failed. Error:{e}")


    print("Get Origin Data From Tushare Finished.")


def process_single_stock(df) -> pd.DataFrame:
    """处理单个股票数据.

    Args:
        df (dataframe): 单只股票原始数据表.

    Returns:
        pd.DataFrame: 单只股票处理后的数据表.
    """
    new_df = pd.DataFrame()

    new_df["open"] = df["open"] / \
        df["open"].shift(5).fillna(method="backfill") - 1

    new_df["pct_chg"] = (df["close"] / \
        df["close"].shift(1).fillna(method="backfill") - 1).clip(-0.3,0.3)

    price_feature_list = [
        "high",
        "low",
        "close",
        "pre_close",
        "ma5",
        "ma10",
        "ma15",
        "ma20",
        "ma25",
    ]
    for col in price_feature_list:
        new_df[col] = df[col] / df["open"] - 1

    std_feature_list = ["change", "vol", "amount"]
    for col in std_feature_list:
        new_df[col] = df[col] / \
            df[col].rolling(window=10).std().fillna(method="bfill")

    keep_feature_list = ["turnover_rate", "turnover_rate_f", "volume_ratio"]
    for col in keep_feature_list:
        new_df[col] = df[col]

    first_order_difference_feature_list = [
        "pe", "pe_ttm", "pb", "ps", "ps_ttm"]
    for col in first_order_difference_feature_list:
        new_df[col] = df[col] - df[col].shift(1).fillna(method="bfill")

    clip_first_order_difference_feature_list = ["total_mv", "circ_mv"]
    for col in clip_first_order_difference_feature_list:
        df[col + "log"] = np.log(df[col] + 1)
        df[col + "_div_std"] = (
            df[col + "log"]
            / df[col + "log"].rolling(window=10).std().fillna(method="bfill")
        ).fillna(0)
        new_df[col] = df[col + "_div_std"] - df[col + "_div_std"].shift(1).fillna(
            method="bfill"
        )

    log_std_clip_feature_list = [
        "buy_sm_vol",
        "buy_sm_amount",
        "sell_sm_vol",
        "sell_sm_amount",
        "buy_md_vol",
        "buy_md_amount",
        "sell_md_vol",
        "sell_md_amount",
        "buy_lg_vol",
        "buy_lg_amount",
        "sell_lg_vol",
        "sell_lg_amount",
        "buy_elg_vol",
        "buy_elg_amount",
        "sell_elg_vol",
        "sell_elg_amount",
    ]
    for col in log_std_clip_feature_list:
        df[col + "_log"] = np.log(df[col] + 1)
        new_df[col] = (
            df[col + "_log"]
            / df[col + "_log"].rolling(window=10).std().fillna(method="bfill")
        ).fillna(0)

    std_clip_feature_list = ["net_mf_vol", "net_mf_amount"]
    for col in std_clip_feature_list:
        new_df[col] = (
            df[col] / df[col].rolling(window=10).std().fillna(method="bfill")
        ).fillna(0)

    limit_feature_list = ["up_limit", "down_limit"]
    for col in limit_feature_list:
        new_df[col] = df[col] / df["open"] - 1

    new_df["industry"] = df["industry"]

    new_df["return_10"] = (df["open"].shift(-11) /
                           df["open"].shift(-1) - 1).fillna(0)

    new_df["return_5"] = (df["open"].shift(-6) /
                          df["open"].shift(-1) - 1).fillna(0)
    
    new_df["return_1"] = (df["open"].shift(-2) /
                          df["open"].shift(-1) - 1).fillna(0)
    new_df["return_10"].loc[(new_df["return_10"] > 1.6) | (new_df["return_10"] < -0.66)] = 0.
    new_df["return_5"].loc[(new_df["return_5"] > 0.61) | (new_df["return_5"] < -0.41)] = 0.
    new_df["return_1"].loc[(new_df["return_1"] > 0.2) | (new_df["return_1"] < -0.2)] = 0.

    new_df["validity_label"] = 1

    for clip_col in clip_dict.keys():
        new_df[clip_col] = new_df[clip_col].clip(
            lower=clip_dict[clip_col][0], upper=clip_dict[clip_col][1]
        )

    new_df=new_df.fillna(0)
    new_df.replace([np.inf, -np.inf], 0, inplace=True)
    return new_df


def process_origin_data_sublist(
    file_name_list, folder_path, save_folder_path
) -> pd.DataFrame:
    """批处理子集.

    Args:
        file_name_list (list): 处理数据名称列表.
        folder_path (str): 原始数据路径.
        save_folder_path (str): 保存路径.

    Returns:
        df: 合并后的df.
    """
    df = pd.DataFrame()
    for file_name in tqdm(file_name_list):
        origin_df = pd.read_pickle(os.path.join(folder_path, file_name))
        process_df = process_single_stock(origin_df)
        process_df.to_pickle(os.path.join(save_folder_path, file_name))
        process_df = process_df.reset_index()
        process_df["ts_code"] = file_name[:9]
        df = pd.concat([df, process_df], axis=0, ignore_index=True)

    return df


def process_origin_data(file_name_list, folder_path, save_folder_path, k) -> None:
    """多进程处理原始数据.

    Args:
        file_name_list (str): 处理数据名称列表.
        folder_path (str): 原始数据路径.
        save_folder_path (str): 保存路径.
        k (_type_): 拆分数量.
    """
    file_name_list_sublists = seperate_k_fold(list_=file_name_list, k=k)
    process_num = len(file_name_list_sublists)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    par = partial(
        process_origin_data_sublist,
        folder_path=folder_path,
        save_folder_path=save_folder_path,
    )
    with Pool(processes=process_num) as p:
        result = p.map(par, file_name_list_sublists)
        print("multiprocess done.")
        df = pd.concat(result, axis=0, ignore_index=True)
        print(len(df))
        df.to_pickle(os.path.join(save_folder_path, "all.pkl"))


def plus_process_origin_data_sublist(
    file_name_list, folder_path, save_folder_path
) -> pd.DataFrame:
    """批处理子集.

    Args:
        file_name_list (list): 处理数据名称列表.
        folder_path (str): 原始数据路径.
        save_folder_path (str): 保存路径.

    Returns:
        df: 合并后的df.
    """
    df = pd.DataFrame()
    for file_name in tqdm(file_name_list):
        origin_df = pd.read_pickle(os.path.join(folder_path, file_name))
        process_df = process_single_stock(origin_df)
        processed_df=pd.read_pickle(os.path.join(save_folder_path,file_name))
        
        cdf=pd.concat([processed_df,process_df])
        print(cdf[cdf.duplicated()==True])
        process_df=cdf.drop_duplicates(keep='first')

        # process_df.to_pickle(os.path.join(save_folder_path, file_name))
        process_df = process_df.reset_index()
        process_df["ts_code"] = file_name[:9]
        df = pd.concat([df, process_df], axis=0, ignore_index=True)

    return df


def plus_process_origin_data(file_name_list, folder_path, save_folder_path, k) -> None:
    """多进程处理原始数据.

    Args:
        file_name_list (str): 处理数据名称列表.
        folder_path (str): 原始数据路径.
        save_folder_path (str): 保存路径.
        k (_type_): 拆分数量.
    """
    file_name_list_sublists = seperate_k_fold(list_=file_name_list, k=k)
    process_num = len(file_name_list_sublists)
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    par = partial(
        plus_process_origin_data_sublist,
        folder_path=folder_path,
        save_folder_path=save_folder_path,
    )
    with Pool(processes=process_num) as p:
        result = p.map(par, file_name_list_sublists)
        print("multiprocess done.")
        df = pd.concat(result, axis=0, ignore_index=True)
        print(len(df))
        df.to_pickle(os.path.join(save_folder_path, "all.pkl"))


def seperate_k_fold(list_, k):
    """对list进行拆分.

    Args:
        list_ (list): 带拆分list.
        k (int): 拆分数量.

    Returns:
        list: 拆分后的list
    """
    len_ = len(list_)
    step = int(len_ / k)
    return [list_[i: i + step] for i in range(0, len_, step)]


def ADF_test(file_name_list, folder_path) -> None:
    """ADF检测Unit root.

    Args:
        file_name_list (str): 文件名称.
        folder_path (str): 文件路径.
    """
    unit_root_dict = {}
    sub_file_name_list = random.sample(file_name_list, 300)
    for file_name in tqdm(sub_file_name_list):
        if file_name == "all.pkl":
            continue
        df = pd.read_pickle(os.path.join(folder_path, file_name)).fillna(0)
        if len(df) < 200:
            continue
        cols = df.columns
        unit_root_cols = []
        for col in cols:
            (adf, p_value, usedlag, nobs, critical_values, icbest) = adfuller(
                df[col].values, autolag="AIC"
            )
            adf_test_re_str = ""
            if p_value > 0.05:
                adf_test_re_str += f"p value {p_value} larger than 0.05. "
                unit_root_cols.append(col)
            for key in critical_values.keys():
                if adf > critical_values[key]:
                    adf_test_re_str += f"adf larger than {key} critical values. "
            if adf_test_re_str != "":
                print(col + ":" + adf_test_re_str)
        unit_root_dict[file_name] = unit_root_cols
    print(unit_root_dict)
    np.save(os.path.join(folder_path, "unit_root_dict.npy"), unit_root_dict)


def get_market_info_from_tushare(save_folder_path, save_file_name):
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    SSE_company_info = ts_pro.stock_company(
        fields=["ts_code", "reg_capital"], exchange='SSE')
    # print(len(SSE_company_info))
    SSE_company_info = SSE_company_info.set_index(
        "ts_code"
    )
    time.sleep(1)
    SZSE_company_info = ts_pro.stock_company(
        exchange="SZSE", fields=["ts_code",   "reg_capital"])
    # print(len(SZSE_company_info))
    SZSE_company_info = SZSE_company_info.set_index('ts_code')
    # print(SZSE_company_info.loc['002550.SZ'])

    BSE_company_info = ts_pro.stock_company(
        fields=["ts_code", "reg_capital"], exchange='BSE')
    # print(BSE_company_info.shape)
    BSE_company_info = BSE_company_info.set_index(
        "ts_code"
    )
    company_info = pd.concat(
        [SSE_company_info, SZSE_company_info, BSE_company_info])
    # print(company_info.shape)
    # print(company_info.loc[['002550.SZ', '002608.SZ', '002696.SZ', '002717.SZ', '002726.SZ', '002741.SZ', '002775.SZ', '002779.SZ', '002851.SZ', '002865.SZ', '002978.SZ', '003000.SZ', '003026.SZ', '003032.SZ', '003816.SZ', '300262.SZ', '300266.SZ', '300337.SZ', '300341.SZ', '300359.SZ', '300383.SZ', '300419.SZ', '300481.SZ', '300553.SZ', '300573.SZ', '300575.SZ', '300615.SZ', '300746.SZ', '300749.SZ', '300752.SZ', '300783.SZ', '300791.SZ', '300808.SZ', '300840.SZ', '300849.SZ', '300852.SZ', '300866.SZ', '300929.SZ', '300966.SZ', '601019.SH', '601086.SH', '601089.SH', '601595.SH', '601865.SH', '601921.SH', '603009.SH', '603013.SH', '603040.SH', '603132.SH', '603208.SH', '603213.SH', '603266.SH', '603389.SH', '603393.SH', '603707.SH', '603767.SH', '603789.SH', '603859.SH', '603936.SH', '603970.SH', '603983.SH', '603988.SH', '605155.SH', '688087.SH', '688165.SH', '688170.SH', '688190.SH', '688195.SH',     '688197.SH', '688260.SH', '688311.SH', '688313.SH', '688501.SH', '688560.SH', '688567.SH', '688596.SH', '688661.SH', '688718.SH']])

    # print(len(stock_list))
    company_info.to_pickle(os.path.join(save_folder_path, save_file_name))
    print("get market info finished.")


def load_market_info(load_folder_path, load_file_name, stock_list):
    if not os.path.exists(os.path.join(load_folder_path, load_file_name)):
        print(os.path.join(load_folder_path, load_file_name))
        print("File not exists.")
        return

    market_info = pd.read_pickle(
        os.path.join(load_folder_path, load_file_name))
    try:
        return_df = market_info.loc[stock_list]
    except:
        print(set(stock_list)-set(market_info.index)) 
        print("market_info not contain all stock in stock list.")
        return

    else:
        print("Load market info finished.")
        return return_df


def get_lob_origin_data_from_joinquant(save_folder_path, stock_list, start_date, end_date, reload_all=False):
    # 输入股票列表
    if isinstance(stock_list, list):
        print("Using the selected stocks.")
    # 全体股票
    elif stock_list == None or stock_list == 'all':
        stock_list = get_all_securities(types=['stock'], date=None).index.to_list()
        stock_list.sort()
        print("Using the whole range of Chinese stock market.")
    # 沪深300
    elif stock_list == "csi300":
        stock_list = get_index_stocks('000300.XSHG')
        print("Using the constituent stocks of market index CSI300.")
    # 中证500
    elif stock_list == "csi500":
        stock_list = get_index_stocks('000905.XSHG')
        print("Using the constituent stocks of market index CSI500.")
    # 中证500+沪深300
    elif stock_list == "csi800":
        csi300_stock_list = get_index_stocks('000300.XSHG')
        csi500_stock_list = get_index_stocks('000905.XSHG')
        stock_list = csi300_stock_list + csi500_stock_list
        stock_list.sort()
        print("Using the constituent stocks of market index CSI800.")
    # 中证1000
    elif stock_list == "csi1000":
        stock_list = get_index_stocks('000852.XSHG')
        print("Using the constituent stocks of market index CSI1000.")
    else:
        print("Wrong type of stock list.")
        return
    print("stock list len is ", len(stock_list))
    time.sleep(0.3)

    # 数据存储文件夹
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 获取各股数据
    for s in tqdm(stock_list):
        # 如果不是重新加载全部，并且当前股票已存在，则跳过
        if not reload_all and os.path.exists(
            os.path.join(save_folder_path, f"{s}.pkl")
        ):
            continue
        
        try:
            df=get_ticks(s, start_dt=start_date, end_dt=end_date,fields=None, skip=False)
            df.to_pickle(os.path.join(save_folder_path, f"{s}.pkl"))

        except Exception as e:
            print(f"Get Stock Data Failed. Error:{e}")


def get_minute_origin_data_from_joinquant(save_folder_path, stock_list, start_date, end_date, reload_all=False):
    # 输入股票列表
    if isinstance(stock_list, list):
        print("Using the selected stocks.")
    # 全体股票
    elif stock_list == None or stock_list == 'all':
        stock_list = get_all_securities(types=['stock'], date=None).index.to_list()
        stock_list.sort()
        print("Using the whole range of Chinese stock market.")
    # 沪深300
    elif stock_list == "csi300":
        stock_list = get_index_stocks('000300.XSHG')
        print("Using the constituent stocks of market index CSI300.")
    # 中证500
    elif stock_list == "csi500":
        stock_list = get_index_stocks('000905.XSHG')
        print("Using the constituent stocks of market index CSI500.")
    # 中证500+沪深300
    elif stock_list == "csi800":
        csi300_stock_list = get_index_stocks('000300.XSHG')
        csi500_stock_list = get_index_stocks('000905.XSHG')
        stock_list = csi300_stock_list + csi500_stock_list
        stock_list.sort()
        print("Using the constituent stocks of market index CSI800.")
    # 中证1000
    elif stock_list == "csi1000":
        stock_list = get_index_stocks('000852.XSHG')
        print("Using the constituent stocks of market index CSI1000.")
    else:
        print("Wrong type of stock list.")
        return
    print("stock list len is ", len(stock_list))
    time.sleep(0.3)

    # 数据存储文件夹
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)

    # 获取各股数据
    for s in tqdm(stock_list):
        # 如果不是重新加载全部，并且当前股票已存在，则跳过
        if not reload_all and os.path.exists(
            os.path.join(save_folder_path, f"{s}.pkl")
        ):
            continue
        
        try:
            df=get_price(s, start_date=start_date, end_date=end_date, frequency='1m',fields=['open','close','low','high','volume','money','factor',
                'high_limit','low_limit','avg','pre_close','paused'], skip_paused=False, fq='pre', count=None,round=True)
            df.to_pickle(os.path.join(save_folder_path, f"{s}.pkl"))

        except Exception as e:
            print(f"Get Stock Data Failed. Error:{e}")


def load_data(
        
    data_type, stock_list=None, start_date=None, end_date=None,  contain_bj=True, drop_short=True, short_len=200, col_name=None
):
    """读取数据.

    Args:
        data_type: 市场_频率_原始/处理后数据
        stock_list: 读取的股票列表
        start_date: 开始时间
        end_date: 结束时间
        contain_bj=True: 是否读取北交所数据
        drop_short=True: 是否去掉过短的股票
        short_len=200: 长度阈值(drop short=False时无效)

    Return:
        date_list: 交易日列表
        features: 特征(tensor)
        labels: 标签(tensor)
        industry_graph: 关系图(tensor)

    """
    current_file_path = os.path.abspath(__file__)
    parent_folder_path = os.path.dirname(current_file_path)

    # 特征列名
    feature_col = [
        "open",
        "pct_chg",
        "high",
        "low",
        "close",
        "pre_close",
        "ma5",
        "ma10",
        "ma15",
        "ma20",
        "ma25",
        "change",
        "vol",
        "amount",
        "turnover_rate",
        "turnover_rate_f",
        "volume_ratio",
        "pe",
        "pe_ttm",
        "pb",
        "ps",
        "ps_ttm",
        "total_mv",
        "circ_mv",
        "buy_sm_vol",
        "buy_sm_amount",
        "sell_sm_vol",
        "sell_sm_amount",
        "buy_md_vol",
        "buy_md_amount",
        "sell_md_vol",
        "sell_md_amount",
        "buy_lg_vol",
        "buy_lg_amount",
        "sell_lg_vol",
        "sell_lg_amount",
        "buy_elg_vol",
        "buy_elg_amount",
        "sell_elg_vol",
        "sell_elg_amount",
        "net_mf_vol",
        "net_mf_amount",
        "up_limit",
        "down_limit",
        "industry",
    ]

    # 标签列名
    # label_col = ["return_10", "return_5", "validity_label", "pct_chg"]
    label_col = ["return_10", "return_5", "validity_label", "pct_chg", "return_1"]

    if data_type == 'feature_col_index':
        try:
            pos = feature_col.index(col_name)
        except:
            print("wrong col name.")
            return None
        else:
            return pos

    elif data_type == 'label_col_index':
        try:
            pos = label_col.index(col_name)
        except:
            print("wrong col name.")
            return None
        else:
            return pos

    # 输入股票列表
    if isinstance(stock_list, list):
        print("Using the selected stocks.")
        # 全体股票
    elif stock_list == "all" or stock_list == None:
        # get all stock basic info
        stock_info = ts_pro.stock_basic()
        # del stock name with "ST"&"退"
        stock_info = stock_info[~stock_info.name.str.contains("ST")]
        stock_info = stock_info[~stock_info.name.str.contains("退")]
        # set ts_code as index
        stock_info = stock_info.set_index("ts_code")
        # 今日停牌股list
        suspend_list = list(
            ts_pro.suspend_d(
                trade_date=datetime.date.today().strftime("%Y%m%d"), suspend_type="S"
            ).ts_code
        )
        # 去掉停牌股票
        stock_info = stock_info.reindex(
            [i for i in stock_info.index if i not in suspend_list]
        )
        stock_list = list(stock_info.index)
        stock_list.sort()
        print("Using the whole range of Chinese stock market.")
        # 沪深300
    elif stock_list == "csi300":
        stock_list = (
            ts_pro.index_weight(
                index_code="399300.SZ").iloc[:300]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI300.")
        # 中证500
    elif stock_list == "csi500":
        stock_list = (
            ts_pro.index_weight(
                index_code="000905.SH").iloc[:500]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI500.")
        # 中证500+沪深300
    elif stock_list == "csi800":
        csi300_stock_list = (
            ts_pro.index_weight(
                index_code="399300.SZ").iloc[:300]["con_code"].tolist()
        )
        csi500_stock_list = (
            ts_pro.index_weight(
                index_code="000905.SH").iloc[:500]["con_code"].tolist()
        )
        stock_list = csi300_stock_list + csi500_stock_list
        stock_list.sort()
        print("Using the constituent stocks of market index CSI800.")
    # 中证1000
    elif stock_list == "csi1000":
        stock_list = (
            ts_pro.index_weight(
                index_code="000852.SH").iloc[:1000]["con_code"].tolist()
        )
        print("Using the constituent stocks of market index CSI1000.")
    else:
        print("Wrong type of stock list.")
        return
    print("stock list len is ", len(stock_list))
    time.sleep(0.3)

    if data_type == 'ch_daily_processed' or data_type == 'ch_daily_origin':
        # A股日线处理后数据
        if data_type == 'ch_daily_processed':
            load_folder_path = os.path.join(
                parent_folder_path, 'CH', 'Processed_data_2024')
        # A股日线原始数据
        elif data_type == 'ch_daily_origin':
            load_folder_path = os.path.join(
                parent_folder_path, 'CH', 'Origin_data')

        # 数据读取文件夹
        if not os.path.exists(load_folder_path):
            print("Load Folder Path not Exitst.")
            return

        del_stock_list = []
        data = {}
        for s in tqdm(stock_list):
            if not contain_bj and s[-2:] == 'BJ':
                del_stock_list.append(s)
                continue
            try:
                df = pd.read_pickle(os.path.join(load_folder_path, f"{s}.pkl"))

            except:
                print(f"Load stock {s} failed.")
                del_stock_list.append(s)
                continue
            else:
                if len(df) == 0:
                    continue
                if end_date is not None:
                    df = df[df.index <= end_date]
                if start_date is not None:
                    df = df[df.index >= start_date]
                if drop_short and len(df) < short_len:
                    del_stock_list.append(s)
                    continue
                data[s] = df.copy()

        stock_list = list(data.keys())
        print(f"Use {len(stock_list)} stock, del {len(del_stock_list)} stock")
        avg_ret = pd.DataFrame({s: data[s]["pct_chg"]
                                for s in data}).fillna(0).mean(axis=1)
        date_list = list(avg_ret.index)

        for s in tqdm(stock_list):
            data[s] = data[s].reindex(date_list)
            data[s]['validity_label']=data[s]['validity_label'].fillna(0)
            # data[s]=data[s].fillna(method="pad").fillna(0)
            data[s][feature_col]=data[s][feature_col].fillna(method="pad").fillna(0)
            data[s][label_col]=data[s][label_col].fillna(0)

        features = []
        for s in tqdm(stock_list):
            features.append(torch.tensor(np.array(data[s][feature_col])))
        features = torch.stack(features)

        labels = []
        for s in tqdm(stock_list):
            if data_type == 'ch_daily_processed':
                labels.append(torch.tensor(np.array(data[s][label_col])))
            elif data_type == 'ch_daily_origin':
                labels.append(torch.tensor(np.array(data[s][['pct_chg']])))
        labels = torch.stack(labels)

        print(
            f"feature tensor shape:{features.shape}, label tensor shape:{labels.shape}."
        )
        return date_list, features, labels, stock_list
    
    elif data_type=='ch_minute_origin':
        if data_type == 'ch_minute_origin':
            load_folder_path = os.path.join(parent_folder_path, 'CH', 'Minute_Origin_data')
        if not os.path.exists(load_folder_path):
            print("Load Folder Path not Exitst.")
            return
        
        del_stock_list = []
        data = {}
        for s in tqdm(stock_list):
            if not contain_bj and s[-2:] == 'BJ':
                del_stock_list.append(s)
                continue
            try:
                # print(os.path.join(load_folder_path, f"{normalize_code(s)}.pkl"))
                df = pd.read_pickle(os.path.join(load_folder_path, f"{normalize_code(s)}.pkl"))

            except Exception as e:
                print(f"Load stock {s} failed. {e}")
                del_stock_list.append(s)
                continue
            else:
                if end_date is not None:
                    df = df[df.index <= end_date]
                if start_date is not None:
                    df = df[df.index >= start_date]
                data[s] = df.copy()
        
        stock_list = list(data.keys())
        print(f"Use {len(stock_list)} stock, del {len(del_stock_list)} stock")
        avg_ret = pd.DataFrame({s: data[s]["close"]
                                for s in data}).fillna(0).mean(axis=1)
        date_list = list(avg_ret.index)

        for s in tqdm(stock_list):
            data[s] = data[s].reindex(date_list).fillna(method="pad").fillna(0)

        features = []
        for s in tqdm(stock_list):
            features.append(torch.tensor(np.array(data[s])))
        features = torch.stack(features)

        labels = []
        for s in tqdm(stock_list):
            if data_type == 'ch_minute_processed':
                labels.append(torch.tensor(np.array(data[s][label_col])))
            elif data_type == 'ch_minute_origin':
                labels.append(torch.tensor(np.array(data[s][['close']])))
        labels = torch.stack(labels)

        print(
            f"feature tensor shape:{features.shape}, label tensor shape:{labels.shape}."
        )
        return date_list, features, labels, stock_list,data

    elif data_type == 'reg_captital':
        load_folder_path = os.path.join(
            parent_folder_path, 'CH', 'Market_data')
        load_file_name = "reg_capital.pkl"
        reg_capital = load_market_info(
            load_file_name=load_file_name,
            load_folder_path=load_folder_path,
            stock_list=stock_list,
        )
        print(
            f"reg_capital shape:{reg_capital.shape}."
        )
        return reg_capital

    else:
        print('Wrong data type.')
        return


if __name__ == "__main__":
    # operator = "Get origin data from tushare"
    # operator = "ADF test"
    # operator = "process"
    # operator = "Load data"
    # operator = "Get market information"
    # operator == "Load market info"
    # company_info = ts_pro.stock_company(fields=["ts_code", "reg_capital"], ts_code='002550.SZ').set_index(
    #     "ts_code"
    # )
    # print(company_info)
    start_time = datetime.datetime.now()
    print(start_time)
    operators = ["Process"]
    current_file_path = os.path.abspath(__file__)
    parent_folder_path = os.path.dirname(current_file_path)
    for operator in operators:
        if operator == "Get origin data from tushare":
            save_folder_path = os.path.join(
                parent_folder_path, 'CH', 'Origin_data')
            stock_list = None
            start_date = None
            end_date = None
            reload_all = True
            get_origin_data_from_tushare(
                save_folder_path=save_folder_path,
                stock_list=stock_list,
                start_date=start_date,
                end_date=end_date,
                reload_all=False,
            )

        elif operator == "Plus Get origin data from tushare":
            save_folder_path = os.path.join(
                parent_folder_path, 'CH', 'Origin_data')
            stock_list = None
            start_date = '2023-01-01'
            end_date = '2024-01-01'
            plus_get_origin_data_from_tushare(
                save_folder_path=save_folder_path,
                stock_list=stock_list,
                start_date=start_date,
                end_date=end_date,
            )

        elif operator == "ADF test":
            folder_path = os.path.join(
                parent_folder_path, 'CH', 'Processed_data')
            file_name_list = os.listdir(folder_path)
            ADF_test(file_name_list=file_name_list, folder_path=folder_path)

        elif operator == "Process":
            folder_path = os.path.join(parent_folder_path, 'CH', 'Origin_data')
            file_name_list = os.listdir(folder_path)
            save_folder_path = os.path.join(
                parent_folder_path, 'CH', 'Processed_data')
            k = 10
            process_origin_data(
                folder_path=folder_path,
                file_name_list=file_name_list,
                save_folder_path=save_folder_path,
                k=k,
            )

        elif operator == "Load data":
            stock_list = None
            start_date = "2010-01-01"
            end_date = "2022-12-31"
            data, _, _, _ = load_data(
                data_type='ch_daily_processed',
                stock_list=stock_list,
                start_date=start_date,
                end_date=end_date,
            )

        elif operator == "Get market information":
            save_folder_path = os.path.join(
                parent_folder_path, 'CH', 'Market_data')
            save_file_name = "reg_capital.pkl"
            get_market_info_from_tushare(
                save_folder_path=save_folder_path, save_file_name=save_file_name
            )

        elif operator == "Load market info":
            load_folder_path = os.path.join(
                parent_folder_path, 'CH', 'Market_data')
            load_file_name = "reg_capital.pkl"
            a = load_market_info(
                load_file_name=load_file_name,
                load_folder_path=load_folder_path,
                stock_list=['002550.SZ', '002608.SZ', '002696.SZ', '002717.SZ', '002726.SZ', '002741.SZ', '002775.SZ', '002779.SZ', '002851.SZ', '002865.SZ', '002978.SZ', '003000.SZ', '003026.SZ', '003032.SZ', '003816.SZ', '300262.SZ', '300266.SZ', '300337.SZ', '300341.SZ', '300359.SZ', '300383.SZ', '300419.SZ', '300481.SZ', '300553.SZ', '300573.SZ', '300575.SZ', '300615.SZ', '300746.SZ', '300749.SZ', '300752.SZ', '300783.SZ', '300791.SZ', '300808.SZ', '300840.SZ', '300849.SZ', '300852.SZ', '300866.SZ', '300929.SZ', '300966.SZ', '601019.SH', '601086.SH', '601089.SH', '601595.SH', '601865.SH', '601921.SH', '603009.SH', '603013.SH', '603040.SH', '603132.SH', '603208.SH', '603213.SH', '603266.SH', '603389.SH', '603393.SH', '603707.SH', '603767.SH', '603789.SH', '603859.SH', '603936.SH', '603970.SH', '603983.SH', '603988.SH', '605155.SH', '688087.SH', '688165.SH', '688170.SH', '688190.SH', '688195.SH',     '688197.SH', '688260.SH', '688311.SH', '688313.SH', '688501.SH', '688560.SH', '688567.SH', '688596.SH', '688661.SH', '688718.SH'])


        elif operator == "Get origin minute data from joinquant":
            save_folder_path = os.path.join(
                parent_folder_path, 'CH', 'Minute_Origin_data')
            stock_list = 'all'
            start_date = '2022/01/01'
            end_date ='2024/01/01'
            reload_all = False
            get_minute_origin_data_from_joinquant(
                save_folder_path=save_folder_path,
                stock_list=stock_list,
                start_date=start_date,
                end_date=end_date,
                reload_all=reload_all
            )

    end_time = datetime.datetime.now()
    print(end_time)
