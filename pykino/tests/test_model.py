from pykino.models import gam
from pykino.concerto import pandas2R, R2pandas
import pandas as pd
import numpy as np
from scipy.special import logit
from scipy.stats import ttest_ind
import os
import pytest
from rpy2.rinterface_lib.embedded import RRuntimeError
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
stats = importr("stats")
base = importr("base")

np.random.seed(10)

os.environ[
    "KMP_DUPLICATE_LIB_OK"
] = "True"  # Recent versions of rpy2 sometimes cause the python kernel to die when running R code; this handles that

#@pytest.fixture
#def testdata():
#    file_path = "C:/Users/User/Desktop/paper/pykino/pykino/resources/testdata.csv"
#    df = pd.read_csv(file_path)
#    r_df=pandas2R(df)
#    return r_df

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'dem': [1, 2, 3, 4, 5],
        'dem48': [2, 3, 4, 5, 6],
        'dow': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'],
        'temp': [15, 16, 17, 18, 19],
        'temp95': [10, 11, 12, 13, 14],
        'tod': [0, 12, 24, 36, 48],
        'toy': [0.1, 0.2, 0.3, 0.4, 0.5]
    })
    return data

def test_gam_1(sample_data):
    model = gam()
    formula = "dem ~ dem48 + dow + temp + temp95 + tod + toy"
    
    # 拟合模型
    fit = model.gam(formula, sample_data)
    assert fit is not None

def test_gam_3(agg):
    model = gam()
    formula = "dem ~ dem48 + dow + temp + temp95 + tod + toy"
    
    # 拟合模型
    fit = model.gam(formula, agg)
    assert fit is not None


def test_gam_2(sample_data):
    model = gam()
    formula = "dem ~ dem48 + dow + s(temp) + s(temp95) + s(tod, bs='cc') + s(toy, bs='cc')"
    
    # 拟合模型
    fit = model.gam(formula, sample_data)
    assert fit is not None

def test_gam_4(agg):
    model = gam()
    formula = "dem ~ dem48 + dow + s(temp) + s(temp95) + s(tod, bs='cc') + s(toy, bs='cc')"
    
    # 拟合模型
    fit = model.gam(formula, agg)
    assert fit is not None

def test_bam(agg):
    model = gam()
    formula = "dem ~ dem48 + dow + s(temp) + s(temp95) + s(tod, bs='cc') + s(toy, bs='cc')"
    
    # 拟合模型
    fit = model.bam(formula, agg)
    assert fit is not None

def test_predict(agg):
    model = gam()
    formula = "dem ~ dem48 + dow + s(temp) + s(temp95) + s(tod, bs='cc') + s(toy, bs='cc')"
    
    # 拟合模型
    fit = model.gam(formula, agg)
    
    # 预测
    predictions = model.predict(fit, agg)
    assert predictions is not None
    assert predictions.size > 0  # Check if the array is not empty

#def test_plot(agg):
#    model = gam()
#    formula = "dem ~ dem48 + dow + s(temp) + s(temp95) + s(tod, bs='cc') + s(toy, bs='cc')"
    
    # 拟合模型
#    fit = model.gam(formula, agg)
    
    # 绘图
#    try:
#        model.plot(fit)
#    except Exception as e:
#        pytest.fail(f"Plotting failed with exception: {e}")

def test_summary(agg):
    # 创建模型对象
    model = gam()
    
    # 定义公式和数据
    formula = "dem ~ dem48 + dow + temp + temp95 + tod + toy"
    
    # 拟合模型
    fit = model.gam(formula, agg)
    
    # 调用 summary 方法
    summary_output = model.summary(fit)
    
    # 进行一些基本的检查（这部分需要根据你的实际需求进行调整）
    assert summary_output is not None
    assert len(summary_output) > 0  # Check if the summary output is not empty
    
    # 可以添加更多检查，以确保 summary 的内容符合预期