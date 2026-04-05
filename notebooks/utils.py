import pandas as pd
import numpy as np

# [사용 방법]
# from utils import reduce_memory_usage
# df = pd.read_csv('../data/order_products__prior.csv')
# df = reduce_memory_usage(df, 'prior')

# 데이터 타입 최적화 함수
def reduce_memory_usage(df, file_name=''):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        
        # 1. 정수형(int)인 경우
        if str(col_type)[:3] == 'int':
            c_min = df[col].min()
            c_max = df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        
        # 2. 실수형(float)인 경우만 float32로 변경
        elif str(col_type)[:5] == 'float':
            df[col] = df[col].astype(np.float32)
            
        # 3. 그 외(object, string 등)는 무시하고 넘어감
        else:
            continue
            
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'메모리 최적화 완료: {start_mem:.2f}MB -> {end_mem:.2f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% 감소)')
    return df