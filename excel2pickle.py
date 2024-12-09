import pandas as pd
from time import time

def use_time(func):
    """
    :param func: 设置时间装饰器
    :return: 返回func函数及wrapper对象
    """
    def wrapper(*args,**kwargs):
        start = time()
        result = func(*args,**kwargs)
        use = round(time() - start,3) #保留3位
        print("%s()用时: %s秒" %(func.__name__,use))
        return result
    return wrapper

@use_time
def read_excel(fn):
    df =pd.read_excel(fn)
    return df

@use_time
def write_to_pickle(df,path):
    df.to_pickle(path)

@use_time
def read_pickle_file(fn):
    df = pd.read_pickle(fn)
    return df

if __name__ == '__main__':
    file_path = './data/413719000/413719000.xls'
     
    #直接读取excel文件
    file01 = read_excel(file_path)
    print(file01)
     
    write_to_pickle(file01,"./data/413719000/new.pkl")
